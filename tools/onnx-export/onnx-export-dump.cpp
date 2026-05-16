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
size_t tensor_data_nbytes(const ggml_tensor * t) {
    return ggml_nbytes(t);
}

bool is_float_type(ggml_type tt) {
    return tt == GGML_TYPE_F32 || tt == GGML_TYPE_F16 || tt == GGML_TYPE_BF16 || tt == GGML_TYPE_F64;
}

void dump_tensor(std::ofstream & f, ggml_tensor * t, const collected & all, bool is_output, bool include_data) {
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

    // Leafs with backing memory are model weights; leafs without are graph
    // inputs (token ids, positions, KV-cache slots, ...). Non-leaf nodes
    // never carry their own data.
    const bool has_data = leaf && t->data != nullptr && include_data && is_float_type(t->type);
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
        const uint64_t nbytes = (uint64_t) tensor_data_nbytes(t);
        write_pod(f, nbytes);
        write_bytes(f, t->data, nbytes);
    } else if (leaf && t->data != nullptr && !is_float_type(t->type)) {
        LOG_WRN("leaf %s has data but unsupported (non-float) type %s; skipping data\n",
                t->name, ggml_type_name(t->type));
        const uint64_t nbytes = 0;
        write_pod(f, nbytes);
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

    size_t total_bytes_written = 0;
    for (size_t i = 0; i < all.tensors.size(); i++) {
        ggml_tensor * t = all.tensors[i];
        const bool is_output = (t == output_node);
        dump_tensor(f, t, all, is_output, /*include_data=*/true);

        if (all.is_leaf(t) && t->data != nullptr && is_float_type(t->type)) {
            total_bytes_written += tensor_data_nbytes(t);
        }
    }

    f.flush();
    LOG_INF("done: %zu bytes of float weight data written\n", total_bytes_written);
    return 0;
}
