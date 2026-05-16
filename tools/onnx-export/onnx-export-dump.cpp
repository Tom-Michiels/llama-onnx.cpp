// llama-onnx-export-dump
//
// Load a GGUF model, ask llama.cpp to build the prefill compute graph for it
// via `llama_graph_reserve()`, and serialise that graph (nodes + leafs +
// weight data) to a portable binary dump file (".gdump"). The companion
// Python script convert_gguf_to_onnx.py reads the dump and emits ONNX.
//
// This binary does not depend on ONNX or protobuf — it only knows about
// ggml/llama. All the ggml -> ONNX op translation happens in Python where
// it's easy to iterate against onnxruntime.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "../src/llama-ext.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cinttypes>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr uint32_t GDUMP_MAGIC   = 0x504D4447u; // "GDMP" (little-endian)
constexpr uint32_t GDUMP_VERSION = 1;

constexpr uint32_t FLAG_HAS_DATA  = 1u << 0;
constexpr uint32_t FLAG_IS_INPUT  = 1u << 1;
constexpr uint32_t FLAG_IS_OUTPUT = 1u << 2;
constexpr uint32_t FLAG_IS_LEAF   = 1u << 3;

template <typename T>
void write_pod(std::ostream & out, const T & v) {
    out.write(reinterpret_cast<const char *>(&v), sizeof(T));
}

void write_bytes(std::ostream & out, const void * p, size_t n) {
    out.write(reinterpret_cast<const char *>(p), (std::streamsize) n);
}

// Flat, topologically-ordered view of a cgraph. Tensors come in two flavours:
//
//   - "leafs": tensors with no sources (model weights, graph inputs, etc.).
//     `ggml_cgraph::leafs` already lists them.
//   - "nodes": computation results. `ggml_cgraph::nodes` lists them in the
//     order they will be evaluated.
//
// Building one combined topo-sorted list is then just leafs ++ nodes, plus
// a pass that picks up any source we somehow missed (rare, but the graph
// reservation sometimes references tensors that are neither in leafs nor
// nodes — typically intermediate views).
struct collected {
    std::vector<ggml_tensor *>                tensors;
    std::unordered_map<ggml_tensor *, size_t> index;

    size_t add(ggml_tensor * t) {
        auto it = index.find(t);
        if (it != index.end()) return it->second;
        size_t idx = tensors.size();
        tensors.push_back(t);
        index[t] = idx;
        return idx;
    }

    // After the main leafs+nodes pass, sweep for any tensor referenced as a
    // source but not yet added. These are usually `view`/`reshape` intermediates
    // that ggml elides from the node list because they share storage with their
    // source. We re-run an iterative topo sort so sources come before consumers.
    void absorb_missing_sources() {
        const std::vector<ggml_tensor *> seed = tensors;
        tensors.clear();
        tensors.reserve(seed.size() * 2);
        index.clear();

        // Iterative DFS: a frame stores the tensor and the index of the next
        // child to visit. We push the tensor as visited only after all its
        // children are emitted.
        enum class state : uint8_t { entered, finished };
        struct frame { ggml_tensor * t; size_t next_src; };
        std::vector<frame> stack;
        std::unordered_map<ggml_tensor *, state> seen;
        seen.reserve(seed.size() * 2);

        auto push_unseen = [&](ggml_tensor * t) {
            if (t == nullptr) return;
            auto it = seen.find(t);
            if (it == seen.end()) {
                seen[t] = state::entered;
                stack.push_back({t, 0});
            }
        };

        for (auto * root : seed) {
            push_unseen(root);
            while (!stack.empty()) {
                frame & f = stack.back();
                if (f.next_src < GGML_MAX_SRC && f.t->src[f.next_src] != nullptr) {
                    ggml_tensor * child = f.t->src[f.next_src];
                    f.next_src++;
                    auto it = seen.find(child);
                    if (it == seen.end()) {
                        seen[child] = state::entered;
                        stack.push_back({child, 0});
                    }
                    continue;
                }
                // All children visited — emit this tensor.
                if (seen[f.t] != state::finished) {
                    seen[f.t] = state::finished;
                    index[f.t] = tensors.size();
                    tensors.push_back(f.t);
                }
                stack.pop_back();
            }
        }
    }

    bool is_leaf(const ggml_tensor * t) const {
        for (size_t s = 0; s < GGML_MAX_SRC; s++) {
            if (t->src[s] != nullptr) return false;
        }
        return t->op == GGML_OP_NONE;
    }
};

// Total number of bytes a tensor's data occupies. Skips unsupported (quantised)
// types — the Python side currently only handles floating-point coefficients.
// On-disk Q4_0 / Q8_0 / Q4_K / ... tensors get re-packed by the CPU backend
// (see ggml/src/ggml-cpu/repack.cpp) for SIMD-friendly access, so `t->data`
// no longer matches the canonical GGUF byte layout. For quantised leafs we
// therefore re-read the original bytes from the GGUF file via the
// gguf_context. For native float types `t->data` is already canonical.
struct gguf_data_reader {
    gguf_context * ctx = nullptr;
    std::FILE * fp = nullptr;
    size_t data_offset = 0;

    bool open(const std::string & path) {
        gguf_init_params p{ /*no_alloc=*/true, /*ctx=*/nullptr };
        ctx = gguf_init_from_file(path.c_str(), p);
        if (!ctx) return false;
        data_offset = gguf_get_data_offset(ctx);
        fp = std::fopen(path.c_str(), "rb");
        return fp != nullptr;
    }

    ~gguf_data_reader() {
        if (fp) std::fclose(fp);
        if (ctx) gguf_free(ctx);
    }

    // Read the canonical bytes for `name` into `buf`. Returns false if the
    // tensor isn't in this GGUF or has a different size than expected.
    bool read(const char * name, void * buf, size_t expected_nbytes) const {
        if (!ctx || !fp) return false;
        const int64_t id = gguf_find_tensor(ctx, name);
        if (id < 0) return false;
        const size_t size = gguf_get_tensor_size(ctx, id);
        if (size != expected_nbytes) return false;
        const size_t off = data_offset + gguf_get_tensor_offset(ctx, id);
        if (std::fseek(fp, (long) off, SEEK_SET) != 0) return false;
        return std::fread(buf, 1, size, fp) == size;
    }
};

void dump_tensor(std::ofstream & f, ggml_tensor * t, const collected & all, bool is_output,
                 const gguf_data_reader * raw = nullptr) {
    const std::string name = t->name;
    const uint32_t name_len = (uint32_t) name.size();
    write_pod(f, name_len);
    write_bytes(f, name.data(), name_len);

    const uint32_t op   = (uint32_t) t->op;
    const uint32_t type = (uint32_t) t->type;

    uint32_t flags = 0;
    const bool leaf = all.is_leaf(t);
    if (leaf) flags |= FLAG_IS_LEAF;
    if (is_output) flags |= FLAG_IS_OUTPUT;

    // Leafs with backing memory are model weights (any ggml type, including
    // quantised); leafs without backing memory are graph inputs (token ids,
    // positions, KV-cache slots, ...). Non-leaf nodes never carry their own
    // data. The Python side dequantises quantised tensors on load.
    const bool has_data = leaf && t->data != nullptr;
    if (has_data) flags |= FLAG_HAS_DATA;
    if (leaf && !has_data) flags |= FLAG_IS_INPUT;

    write_pod(f, op);
    write_pod(f, type);
    write_pod(f, flags);

    for (int i = 0; i < 4; i++) write_pod(f, (int64_t) t->ne[i]);
    for (int i = 0; i < 4; i++) write_pod(f, (int64_t) t->nb[i]);

    const uint32_t op_params_size = (uint32_t) sizeof(t->op_params);
    write_pod(f, op_params_size);
    write_bytes(f, t->op_params, op_params_size);

    uint32_t n_src = 0;
    for (size_t s = 0; s < GGML_MAX_SRC; s++) {
        if (t->src[s] == nullptr) break;
        n_src++;
    }
    write_pod(f, n_src);
    for (uint32_t s = 0; s < n_src; s++) {
        ggml_tensor * src = t->src[s];
        auto it = all.index.find(src);
        if (it == all.index.end()) {
            LOG_ERR("internal error: source tensor %s not interned\n", src->name);
            std::exit(1);
        }
        const uint32_t idx = (uint32_t) it->second;
        write_pod(f, idx);
    }

    if (has_data) {
        // ggml_nbytes() returns the actual byte length regardless of quant type.
        const uint64_t nbytes = (uint64_t) ggml_nbytes(t);
        write_pod(f, nbytes);
        // For quantised types, t->data may have been re-packed by the CPU
        // backend for SIMD; read the canonical layout from the GGUF file
        // instead. Float / integer types are not re-packed, so for them
        // we can copy t->data directly.
        const bool quantised = !(t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 ||
                                 t->type == GGML_TYPE_BF16 || t->type == GGML_TYPE_F64 ||
                                 t->type == GGML_TYPE_I8  || t->type == GGML_TYPE_I16 ||
                                 t->type == GGML_TYPE_I32 || t->type == GGML_TYPE_I64);
        if (quantised && raw != nullptr) {
            std::vector<uint8_t> buf(nbytes);
            if (raw->read(t->name, buf.data(), nbytes)) {
                write_bytes(f, buf.data(), nbytes);
                return;
            }
            LOG_WRN("could not read canonical bytes for %s from GGUF; falling back to repacked t->data\n", t->name);
        }
        write_bytes(f, t->data, nbytes);
    }
}

} // namespace

int main(int argc, char ** argv) {
    common_params params;
    params.out_file = "model.gdump";
    // The dumper inspects the graph once with a fixed shape; the prefill graph
    // is built for these many tokens. The Python translator currently emits a
    // model with concrete shapes, so pick a value that matches how the user
    // intends to call the model.
    params.n_batch  = 32;
    params.n_ubatch = 32;
    params.n_ctx    = 512;

    // ``--fixture <path>``: after dumping the graph, also run a real
    // llama_decode() with deterministic input tokens and serialise the
    // (post-decode) inputs the graph consumes plus the reference logits.
    // The Python side can then feed those exact inputs through onnxruntime
    // and compare logits — that's the bit-for-bit verification harness.
    std::string fixture_path;
    for (int i = 1; i + 1 < argc; i++) {
        if (std::string(argv[i]) == "--fixture") {
            fixture_path = argv[i + 1];
            // Remove this pair from argv so common_params_parse doesn't see it.
            for (int j = i; j + 2 <= argc; j++) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            break;
        }
    }

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_EXPORT_GRAPH_OPS)) {
        return 1;
    }

    // Force a CPU-only setup: the dumper does not actually compute anything
    // (we never call llama_decode), but using CPU keeps the graph close to
    // what onnxruntime will run.
    ggml_backend_dev_t cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    params.devices = { cpu_device, nullptr };
    params.fit_params = false;
    params.n_gpu_layers = 0;
    params.warmup = false;

    auto init_result = common_init_from_params(params);
    llama_context * ctx = init_result->context();
    if (!ctx) {
        LOG_ERR("failed to initialise llama context\n");
        return 1;
    }

    const llama_model * model = llama_get_model(ctx);
    if (!model) {
        LOG_ERR("failed to get llama_model\n");
        return 1;
    }

    const uint32_t n_seqs   = 1;
    const uint32_t n_tokens = std::min<uint32_t>(llama_n_ctx(ctx), llama_n_ubatch(ctx));

    ggml_cgraph * gf = llama_graph_reserve(ctx, n_tokens, n_seqs, n_tokens);
    if (!gf) {
        LOG_ERR("failed to reserve prefill graph (n_tokens=%u n_seqs=%u)\n", n_tokens, n_seqs);
        return 1;
    }
    LOG_INF("reserved graph: %d nodes, %d leafs\n", gf->n_nodes, gf->n_leafs);

    // Collect leafs first (so their indices are stable and small) then nodes
    // in evaluation order. A final sweep catches any "view of a view"-style
    // intermediate that ggml doesn't list explicitly.
    collected all;
    for (int i = 0; i < gf->n_leafs; i++) {
        if (gf->leafs[i]) all.add(gf->leafs[i]);
    }
    for (int i = 0; i < gf->n_nodes; i++) {
        if (gf->nodes[i]) all.add(gf->nodes[i]);
    }
    all.absorb_missing_sources();
    LOG_INF("collected %zu tensors (leafs+nodes incl. unreferenced views)\n", all.tensors.size());

    // The graph's "output" is the last node (everything after that has been
    // consumed somewhere or is dead). For llama this is the logits tensor.
    ggml_tensor * output_node = gf->n_nodes > 0 ? gf->nodes[gf->n_nodes - 1] : nullptr;

    // -------------------- write the dump --------------------

    std::ofstream f(params.out_file, std::ios::binary);
    if (!f.is_open()) {
        LOG_ERR("cannot open output file %s\n", params.out_file.c_str());
        return 1;
    }

    write_pod(f, GDUMP_MAGIC);
    write_pod(f, GDUMP_VERSION);

    // Small metadata block: architecture string + the (n_tokens, n_seqs) we
    // built the graph for. This lets the Python side pick reasonable input
    // shapes without having to re-read the GGUF.
    {
        char buf[64] = {};
        llama_model_meta_val_str(model, "general.architecture", buf, sizeof(buf));
        const std::string arch = buf[0] ? buf : "unknown";
        const uint32_t arch_len = (uint32_t) arch.size();
        write_pod(f, arch_len);
        write_bytes(f, arch.data(), arch_len);
        write_pod(f, n_tokens);
        write_pod(f, n_seqs);
    }

    const uint64_t n_tensors = (uint64_t) all.tensors.size();
    write_pod(f, n_tensors);

    LOG_INF("dumping %llu tensors to %s\n",
            (unsigned long long) n_tensors, params.out_file.c_str());

    // Open the GGUF a second time so we can read the canonical (non-repacked)
    // bytes for any quantised tensors.
    gguf_data_reader raw;
    if (!raw.open(params.model.path)) {
        LOG_WRN("could not open %s for raw tensor reads; quantised weights may be re-packed\n",
                params.model.path.c_str());
    }

    size_t total_bytes_written = 0;
    for (size_t i = 0; i < all.tensors.size(); i++) {
        ggml_tensor * t = all.tensors[i];
        const bool is_output = (t == output_node);
        dump_tensor(f, t, all, is_output, &raw);

        if (all.is_leaf(t) && t->data != nullptr) {
            total_bytes_written += ggml_nbytes(t);
        }
    }

    f.flush();
    LOG_INF("done: %zu bytes of float weight data written\n", total_bytes_written);

    // -------------------- optional verification fixture --------------------

    if (!fixture_path.empty()) {
        // Build a deterministic prompt: tokens [0, 1, 2, ..., n_tokens-1] mod
        // vocab_size, in a single sequence. That keeps the test reproducible
        // and works even with the "no_vocab" tokenizer the synthetic GGUFs
        // use.
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        std::vector<llama_token> tokens(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            tokens[i] = (llama_token)(i % (uint32_t) n_vocab);
        }

        llama_batch batch = llama_batch_init((int32_t) n_tokens, /*embd=*/0, /*n_seq_max=*/1);
        batch.n_tokens = (int32_t) n_tokens;
        for (uint32_t i = 0; i < n_tokens; i++) {
            batch.token   [i] = tokens[i];
            batch.pos     [i] = (llama_pos) i;
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i][0] = 0;
            batch.logits  [i] = 1;  // ask for logits at every position
        }

        LOG_INF("running llama_decode for fixture: n_tokens=%u\n", n_tokens);
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("llama_decode failed\n");
            return 1;
        }

        // Open the fixture file.
        std::ofstream fx(fixture_path, std::ios::binary);
        if (!fx.is_open()) {
            LOG_ERR("cannot open fixture file %s\n", fixture_path.c_str());
            return 1;
        }

        constexpr uint32_t FIXTURE_MAGIC = 0x54584647u; // "GFXT"
        constexpr uint32_t FIXTURE_VERSION = 2;
        write_pod(fx, FIXTURE_MAGIC);
        write_pod(fx, FIXTURE_VERSION);
        write_pod(fx, n_tokens);
        write_pod(fx, (uint32_t) n_vocab);

        // The token IDs we fed in (the Python side synthesises the rest of
        // the graph's inputs — positions, KV-cache indices, the causal mask
        // — from these and the gdump's graph structure).
        for (uint32_t i = 0; i < n_tokens; i++) {
            const int32_t tok = (int32_t) batch.token[i];
            write_pod(fx, tok);
        }

        // Reference logits: contiguous [n_tokens, n_vocab] fp32 array (we
        // requested logits at every position).
        const float * logits = llama_get_logits(ctx);
        const uint64_t logits_nbytes = (uint64_t) n_tokens * (uint64_t) n_vocab * sizeof(float);
        write_pod(fx, logits_nbytes);
        write_bytes(fx, logits, logits_nbytes);

        llama_batch_free(batch);
        fx.flush();
        LOG_INF("wrote fixture: %s (%u tokens, %d-vocab logits)\n",
                fixture_path.c_str(), n_tokens, n_vocab);
    }
    return 0;
}
