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
#include "../mtmd/clip.h"
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
#include <random>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

        // Iterative DFS: each `frame` stores the tensor and the index of the
        // next source to descend into. We add a tensor to `tensors` (i.e.
        // mark it finished) only after all its sources have been emitted.
        struct frame { ggml_tensor * t; size_t next_src; };
        std::vector<frame> stack;
        std::unordered_set<ggml_tensor *> finished;
        finished.reserve(seed.size() * 2);
        std::unordered_set<ggml_tensor *> on_stack;
        on_stack.reserve(seed.size() * 2);

        auto push_unseen = [&](ggml_tensor * t) {
            if (t == nullptr) return;
            if (finished.count(t) || on_stack.count(t)) return;
            on_stack.insert(t);
            stack.push_back({t, 0});
        };

        for (auto * root : seed) {
            push_unseen(root);
            while (!stack.empty()) {
                frame & f = stack.back();
                if (f.next_src < GGML_MAX_SRC && f.t->src[f.next_src] != nullptr) {
                    ggml_tensor * child = f.t->src[f.next_src];
                    f.next_src++;
                    push_unseen(child);
                    continue;
                }
                // All children visited — emit this tensor.
                if (!finished.count(f.t)) {
                    finished.insert(f.t);
                    index[f.t] = tensors.size();
                    tensors.push_back(f.t);
                }
                on_stack.erase(f.t);
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

// ----------------------------------------------------------------------
// Intermediate-activation capture
//
// When --fixture is supplied, we install a ggml backend-scheduler eval
// callback (the same hook used by examples/eval-callback) that fires once
// per ggml node. For tensors whose name matches one of a fixed set of
// "interesting" prefixes (attn_norm-N, Qcur-N, ...), we copy the tensor
// contents out (converting to fp32 if needed) and remember them keyed by
// the ggml name. After llama_decode returns, we append them to the fixture
// file. The Python side adds matching extra outputs to the ONNX model so
// we can compare layer-by-layer.
// ----------------------------------------------------------------------

struct intermediate_capture {
    // ggml name -> raw fp32 row-major data, length-prefixed by shape.
    struct entry {
        std::string name;
        std::vector<int64_t> ne;  // length 4 (same convention as ggml)
        std::vector<float>   data;
    };
    std::vector<entry>             entries;
    std::vector<std::regex>        filters;

    bool matches(const char * name) const {
        if (filters.empty()) return false;
        for (const auto & f : filters) {
            if (std::regex_match(name, f)) return true;
        }
        return false;
    }
};

bool intermediate_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cap = static_cast<intermediate_capture *>(user_data);
    if (ask) {
        return cap->matches(t->name);
    }
    if (!cap->matches(t->name)) {
        return true;
    }

    // Read the tensor contents into a host buffer. CPU backend tensors
    // are already host-accessible, but we go through ggml_backend_tensor_get
    // unconditionally to keep this generic.
    const size_t nb = ggml_nbytes(t);
    std::vector<uint8_t> raw(nb);
    ggml_backend_tensor_get(t, raw.data(), 0, nb);

    intermediate_capture::entry e;
    e.name = t->name;
    e.ne = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };

    const int64_t n_elem = (int64_t) ggml_nelements(t);
    e.data.resize((size_t) n_elem);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(e.data.data(), raw.data(), n_elem * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = reinterpret_cast<const ggml_fp16_t *>(raw.data());
        for (int64_t i = 0; i < n_elem; i++) {
            e.data[i] = ggml_fp16_to_fp32(src[i]);
        }
    } else if (t->type == GGML_TYPE_BF16) {
        const ggml_bf16_t * src = reinterpret_cast<const ggml_bf16_t *>(raw.data());
        for (int64_t i = 0; i < n_elem; i++) {
            e.data[i] = ggml_bf16_to_fp32(src[i]);
        }
    } else {
        // Skip unsupported dtypes (e.g. quantised intermediates — shouldn't happen for the
        // tensors we're capturing).
        return true;
    }
    // Multiple ggml nodes can share the same name (e.g. "Qcur-0" sits on the
    // MUL_MAT, the RESHAPE and the ROPE result). We want the LAST one in
    // topo order, which corresponds to the cb()-site in src/models/<arch>.cpp.
    // The eval-callback fires per-node in topo order, so just overwrite any
    // earlier entry with the same name.
    for (auto & existing : cap->entries) {
        if (existing.name == e.name) {
            existing = std::move(e);
            return true;
        }
    }
    cap->entries.push_back(std::move(e));
    return true;
}

// Serialise a single cgraph (already-reserved, never executed) to ``out_path``
// using the gdump v1 layout: magic + version, arch + (n_tokens, n_seqs)
// metadata, then a flat list of tensor records. ``raw_gguf_path`` is used to
// re-read canonical bytes for quantised weights (the CPU backend re-packs
// them after load, so t->data is no longer canonical); pass an empty string
// to skip that fallback.
//
// Returns true on success, false on any IO error.
bool dump_cgraph_to_file(ggml_cgraph * gf,
                         const std::string & arch,
                         uint32_t n_tokens, uint32_t n_seqs,
                         const std::string & out_path,
                         const std::string & raw_gguf_path) {
    if (!gf) {
        LOG_ERR("dump_cgraph_to_file: cgraph is null\n");
        return false;
    }

    collected all;
    for (int i = 0; i < gf->n_leafs; i++) {
        if (gf->leafs[i]) all.add(gf->leafs[i]);
    }
    for (int i = 0; i < gf->n_nodes; i++) {
        if (gf->nodes[i]) all.add(gf->nodes[i]);
    }
    all.absorb_missing_sources();
    LOG_INF("collected %zu tensors (leafs+nodes incl. unreferenced views)\n", all.tensors.size());

    ggml_tensor * output_node = gf->n_nodes > 0 ? gf->nodes[gf->n_nodes - 1] : nullptr;

    std::ofstream f(out_path, std::ios::binary);
    if (!f.is_open()) {
        LOG_ERR("cannot open output file %s\n", out_path.c_str());
        return false;
    }

    write_pod(f, GDUMP_MAGIC);
    write_pod(f, GDUMP_VERSION);

    const uint32_t arch_len = (uint32_t) arch.size();
    write_pod(f, arch_len);
    write_bytes(f, arch.data(), arch_len);
    write_pod(f, n_tokens);
    write_pod(f, n_seqs);

    const uint64_t n_tensors = (uint64_t) all.tensors.size();
    write_pod(f, n_tensors);

    LOG_INF("dumping %llu tensors to %s\n",
            (unsigned long long) n_tensors, out_path.c_str());

    gguf_data_reader raw;
    const bool have_raw = !raw_gguf_path.empty() && raw.open(raw_gguf_path);
    if (!raw_gguf_path.empty() && !have_raw) {
        LOG_WRN("could not open %s for raw tensor reads; quantised weights may be re-packed\n",
                raw_gguf_path.c_str());
    }

    size_t total_bytes_written = 0;
    for (size_t i = 0; i < all.tensors.size(); i++) {
        ggml_tensor * t = all.tensors[i];
        const bool is_output = (t == output_node);
        dump_tensor(f, t, all, is_output, have_raw ? &raw : nullptr);
        if (all.is_leaf(t) && t->data != nullptr) {
            total_bytes_written += ggml_nbytes(t);
        }
    }
    f.flush();
    LOG_INF("done: %zu bytes of weight data written to %s\n", total_bytes_written, out_path.c_str());
    return true;
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
    //
    // ``--vision-dump <path>``: build the clip (vision/audio) compute graph
    // for the GGUF and serialise it to its own .gdump file. The vision graph
    // is independent from the LM graph; when this flag is set the dumper
    // will additionally build/dump the vision encoder.
    //
    // ``--include-vision``: convenience flag that places the vision dump
    // next to the main output (out.gdump -> out.vision.gdump).
    //
    // ``--vision-only``: skip the LM dump entirely; useful for vision-only
    // mmproj GGUFs that have no language model to reserve a graph for.
    std::string fixture_path;
    std::string fixture_prompt;
    std::string vision_dump_path;
    std::string vision_fixture_path;
    bool include_vision = false;
    bool vision_only = false;
    for (int i = 1; i < argc; ) {
        const std::string a = argv[i];
        if (a == "--include-vision") {
            include_vision = true;
            for (int j = i; j + 1 <= argc; j++) argv[j] = argv[j + 1];
            argc -= 1;
            continue;
        }
        if (a == "--vision-only") {
            vision_only = true;
            include_vision = true;
            for (int j = i; j + 1 <= argc; j++) argv[j] = argv[j + 1];
            argc -= 1;
            continue;
        }
        if (i + 1 < argc && (a == "--fixture" || a == "--fixture-prompt" ||
                             a == "--vision-dump" || a == "--vision-fixture")) {
            if (a == "--fixture") {
                fixture_path = argv[i + 1];
            } else if (a == "--fixture-prompt") {
                fixture_prompt = argv[i + 1];
            } else if (a == "--vision-fixture") {
                vision_fixture_path = argv[i + 1];
                include_vision = true;
            } else {
                vision_dump_path = argv[i + 1];
                include_vision = true;
            }
            for (int j = i; j + 2 <= argc; j++) argv[j] = argv[j + 2];
            argc -= 2;
            continue;
        }
        i++;
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

    // If we're going to run a fixture, install a backend-scheduler eval
    // callback to capture certain named intermediate tensors during
    // llama_decode. Filters match the names that src/models/llama.cpp
    // sets via cb(...): attn_norm-N, Qcur-N, Kcur-N, Vcur-N, attn_out-N,
    // ffn_inp-N, ffn_norm-N, ffn_out-N, l_out-N, result_norm, result_output.
    intermediate_capture capture;
    if (!fixture_path.empty()) {
        // Match names ending in -<layer> for per-layer tensors, plus the
        // un-suffixed global ones. We anchor with regex_match.
        const char * patterns[] = {
            "attn_norm-[0-9]+",
            "Qcur-[0-9]+", "Kcur-[0-9]+", "Vcur-[0-9]+",
            "attn_out-[0-9]+",
            "ffn_inp-[0-9]+",
            "ffn_norm-[0-9]+",
            "ffn_out-[0-9]+",
            "l_out-[0-9]+",
            "result_norm", "result_output",
            // Internal ggml-named tensors that surface useful info on the
            // attention path (final K/V copy to cache, RoPE intermediates).
            // We don't rely on these but they're cheap to keep.
            "Qcur-[0-9]+ \\(reshaped\\)",
            "Kcur-[0-9]+ \\(reshaped\\)",
            "Vcur-[0-9]+ \\(reshaped\\)",
            "kqv_out-[0-9]+",
        };
        for (const char * p : patterns) {
            capture.filters.emplace_back(p, std::regex::optimize);
        }
        params.cb_eval           = intermediate_eval_cb;
        params.cb_eval_user_data = &capture;
    }

    // ------------------------------------------------------------------
    // Detect modality. A vision-only mmproj GGUF has no LM, so calling
    // common_init_from_params would fail; we need to skip the LM path in
    // that case. clip_get_cap reads the GGUF header without loading tensors.
    // ------------------------------------------------------------------
    clip_cap cap = {};
    if (include_vision) {
        cap = clip_get_cap(params.model.path.c_str());
        LOG_INF("clip_get_cap(%s): has_vision=%d has_audio=%d\n",
                params.model.path.c_str(), (int) cap.has_vision, (int) cap.has_audio);
        if (!cap.has_vision && !cap.has_audio) {
            LOG_WRN("--include-vision/--vision-dump requested but GGUF has no vision/audio encoder\n");
            include_vision = false;
        }
    }

    // ------------------------------------------------------------------
    // LM dump (skipped when --vision-only or when the GGUF is vision-only).
    // ------------------------------------------------------------------
    common_init_result_ptr init_result_holder;
    llama_context * ctx = nullptr;
    const llama_model * model = nullptr;
    uint32_t lm_n_tokens = 0;
    uint32_t lm_n_seqs   = 1;

    if (!vision_only) {
        init_result_holder = common_init_from_params(params);
        ctx = init_result_holder ? init_result_holder->context() : nullptr;
        if (!ctx) {
            LOG_ERR("failed to initialise llama context\n");
            return 1;
        }

        model = llama_get_model(ctx);
        if (!model) {
            LOG_ERR("failed to get llama_model\n");
            return 1;
        }

        lm_n_tokens = std::min<uint32_t>(llama_n_ctx(ctx), llama_n_ubatch(ctx));

        ggml_cgraph * gf = llama_graph_reserve(ctx, lm_n_tokens, lm_n_seqs, lm_n_tokens);
        if (!gf) {
            LOG_ERR("failed to reserve prefill graph (n_tokens=%u n_seqs=%u)\n", lm_n_tokens, lm_n_seqs);
            return 1;
        }
        LOG_INF("reserved graph: %d nodes, %d leafs\n", gf->n_nodes, gf->n_leafs);

        char arch_buf[64] = {};
        llama_model_meta_val_str(model, "general.architecture", arch_buf, sizeof(arch_buf));
        const std::string arch = arch_buf[0] ? arch_buf : "unknown";

        if (!dump_cgraph_to_file(gf, arch, lm_n_tokens, lm_n_seqs,
                                 params.out_file, params.model.path)) {
            return 1;
        }
    } else {
        LOG_INF("--vision-only: skipping LM graph dump\n");
    }

    // ------------------------------------------------------------------
    // Vision dump (--include-vision / --vision-dump / --vision-only).
    //
    // The .gdump produced here uses the same v1 format as the LM dump. The
    // Python side reads it through gdump.load(...) just like an LM dump
    // and translates ggml ops to ONNX through the same pipeline.
    // ------------------------------------------------------------------
    if (include_vision) {
        if (vision_dump_path.empty()) {
            // Default: place the vision dump next to the main output. If the
            // user's outfile is "out.gdump" we use "out.vision.gdump".
            const std::string & lm = params.out_file;
            std::string base = lm;
            const size_t dot = lm.rfind('.');
            if (dot != std::string::npos && dot > 0) {
                base = lm.substr(0, dot);
                vision_dump_path = base + ".vision" + lm.substr(dot);
            } else {
                vision_dump_path = lm + ".vision";
            }
        }

        clip_context_params cparams = {};
        cparams.use_gpu = false;
        cparams.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
        cparams.image_min_tokens = -1;
        cparams.image_max_tokens = -1;
        cparams.warmup = false;
        cparams.cb_eval = nullptr;
        cparams.cb_eval_user_data = nullptr;

        clip_init_result clip_res = clip_init(params.model.path.c_str(), cparams);
        if (!clip_res.ctx_v && !clip_res.ctx_a) {
            LOG_ERR("clip_init failed for %s\n", params.model.path.c_str());
            return 1;
        }

        // Pick the first available encoder context. Most mmproj GGUFs only
        // carry one modality at a time, but the audio path uses the same
        // dump-graph machinery as vision.
        clip_ctx * clip_ctx_for_dump = clip_res.ctx_v ? clip_res.ctx_v : clip_res.ctx_a;
        clip_dump_graph * dg = clip_dump_graph_create(clip_ctx_for_dump);
        if (!dg) {
            LOG_ERR("clip_dump_graph_create failed\n");
            clip_free(clip_res.ctx_v);
            clip_free(clip_res.ctx_a);
            return 1;
        }

        ggml_cgraph * vg = clip_dump_graph_cgraph(dg);
        LOG_INF("clip graph: %d nodes, %d leafs\n", vg->n_nodes, vg->n_leafs);

        // We tag the arch with a "clip-" prefix so the Python side can tell
        // apart a vision dump from an LM dump. n_tokens for the clip graph
        // is per-encoder so we encode 0 — the Python translator only uses
        // it as an informational hint anyway.
        const std::string vision_arch = std::string("clip-") +
            (clip_ctx_for_dump == clip_res.ctx_v ? "vision" : "audio");

        const bool ok = dump_cgraph_to_file(vg, vision_arch, 0, 0,
                                            vision_dump_path, params.model.path);
        clip_dump_graph_free(dg);
        if (!ok) {
            clip_free(clip_res.ctx_v);
            clip_free(clip_res.ctx_a);
            return 1;
        }

        // -------------- optional vision verification fixture ----------------
        //
        // Mirrors the LM fixture flow: feed a deterministic input through the
        // clip encoder and serialise (input_pixels, output_features) so the
        // Python side can compare the ONNX-translated graph against the
        // ggml/clip.cpp CPU reference.
        if (!vision_fixture_path.empty()) {
            // Vision fixtures only make sense for the vision modality (audio
            // would need a different input shape and entry point).
            if (clip_ctx_for_dump != clip_res.ctx_v) {
                LOG_ERR("--vision-fixture requires a vision encoder, but only the audio one is present\n");
                clip_free(clip_res.ctx_v);
                clip_free(clip_res.ctx_a);
                return 1;
            }

            const int32_t img_size = clip_get_image_size(clip_ctx_for_dump);
            const int H = (int) img_size;
            const int W = (int) img_size;
            const int C = 3;
            const size_t n_pixels = (size_t) C * (size_t) H * (size_t) W;

            // Deterministic input: mt19937 seeded with 42, uniform [-1, 1].
            // Layout matches what clip_encode_float_image expects: a flat
            // (H, W, C) interleaved RGB float buffer. We also write that
            // exact buffer to the fixture so the ONNX side can feed the same
            // bytes (the ONNX vision encoder will be wrapped with the same
            // HWC->CHW unrolling that lives inside clip_image_batch_encode,
            // OR the comparison driver can reshape to (C, H, W) if its model
            // expects that layout — both layouts have the same elements).
            std::vector<float> img_data(n_pixels);
            {
                std::mt19937 rng(42);
                std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
                for (size_t i = 0; i < n_pixels; i++) {
                    img_data[i] = dist(rng);
                }
            }

            const size_t out_nbytes = clip_embd_nbytes(clip_ctx_for_dump);
            const size_t n_out = out_nbytes / sizeof(float);
            std::vector<float> out_feat(n_out, 0.0f);

            LOG_INF("running clip_encode_float_image for vision fixture: H=%d W=%d C=%d (out=%zu floats)\n",
                    H, W, C, n_out);
            if (!clip_encode_float_image(clip_ctx_for_dump, /*n_threads=*/1,
                                         img_data.data(), H, W, out_feat.data())) {
                LOG_ERR("clip_encode_float_image failed\n");
                clip_free(clip_res.ctx_v);
                clip_free(clip_res.ctx_a);
                return 1;
            }

            // The number of output tokens depends on the projector type / image
            // size; recover it from out_nbytes and the projection embedding
            // dimension so we can write a meaningful 2D shape.
            const int n_mmproj = clip_n_mmproj_embd(clip_ctx_for_dump);
            const int64_t n_out_tokens = (n_mmproj > 0) ? (int64_t)(n_out / (size_t) n_mmproj) : 0;

            std::ofstream fx(vision_fixture_path, std::ios::binary);
            if (!fx.is_open()) {
                LOG_ERR("cannot open vision fixture file %s\n", vision_fixture_path.c_str());
                clip_free(clip_res.ctx_v);
                clip_free(clip_res.ctx_a);
                return 1;
            }

            constexpr uint32_t FIXTURE_MAGIC   = 0x54584647u; // "GFXT"
            constexpr uint32_t FIXTURE_VERSION = 4;
            write_pod(fx, FIXTURE_MAGIC);
            write_pod(fx, FIXTURE_VERSION);
            // 4-byte kind tag, NUL-padded. v3 has no tag (implicit "lm").
            const char kind[4] = { 'v', 'i', 's', '\0' };
            write_bytes(fx, kind, 4);

            // input shape: (H, W, C) interleaved RGB (matches img_data layout).
            const uint32_t in_rank = 3;
            write_pod(fx, in_rank);
            write_pod(fx, (uint64_t) H);
            write_pod(fx, (uint64_t) W);
            write_pod(fx, (uint64_t) C);
            write_bytes(fx, img_data.data(), img_data.size() * sizeof(float));

            // output_name: name of the last node in the clip cgraph (the
            // projected visual features). Read it back from the cgraph we
            // already dumped.
            const char * out_name = "vision_features";
            if (vg->n_nodes > 0 && vg->nodes[vg->n_nodes - 1] && vg->nodes[vg->n_nodes - 1]->name[0]) {
                out_name = vg->nodes[vg->n_nodes - 1]->name;
            }
            const uint32_t name_len = (uint32_t) std::strlen(out_name);
            write_pod(fx, name_len);
            write_bytes(fx, out_name, name_len);

            // output shape: (n_out_tokens, n_mmproj) when we know it,
            // otherwise just a flat 1-D shape.
            if (n_out_tokens > 0 && (size_t)(n_out_tokens * n_mmproj) == n_out) {
                const uint32_t out_rank = 2;
                write_pod(fx, out_rank);
                write_pod(fx, (uint64_t) n_out_tokens);
                write_pod(fx, (uint64_t) n_mmproj);
            } else {
                const uint32_t out_rank = 1;
                write_pod(fx, out_rank);
                write_pod(fx, (uint64_t) n_out);
            }
            write_bytes(fx, out_feat.data(), out_feat.size() * sizeof(float));

            fx.flush();
            LOG_INF("wrote vision fixture: %s (input=%dx%dx%d, output=%zu floats, name=\"%s\")\n",
                    vision_fixture_path.c_str(), H, W, C, n_out, out_name);
        }

        clip_free(clip_res.ctx_v);
        clip_free(clip_res.ctx_a);
    } else if (!vision_fixture_path.empty()) {
        LOG_ERR("--vision-fixture requires --include-vision or --vision-only\n");
        return 1;
    }

    // -------------------- optional verification fixture --------------------

    if (!fixture_path.empty()) {
        if (!ctx || !model) {
            LOG_ERR("--fixture requires an LM (cannot use with --vision-only)\n");
            return 1;
        }
        const uint32_t n_tokens = lm_n_tokens;
        // Build the input token sequence. Default: deterministic synthetic
        // [0, 1, ..., n_tokens-1] mod vocab_size (works for the "no_vocab"
        // tokenizer synthetic GGUFs use). With --fixture-prompt, tokenise
        // the supplied text via the model's vocab; truncate from the right
        // if longer than n_tokens, or pad with token 0 if shorter.
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        std::vector<llama_token> tokens(n_tokens);
        if (!fixture_prompt.empty()) {
            std::vector<llama_token> toks = common_tokenize(ctx, fixture_prompt,
                /*add_special=*/true, /*parse_special=*/false);
            LOG_INF("tokenised fixture prompt: %zu tokens (truncating/padding to %u)\n",
                    toks.size(), n_tokens);
            for (uint32_t i = 0; i < n_tokens; i++) {
                tokens[i] = i < toks.size() ? toks[i] : (llama_token) 0;
            }
        } else {
            for (uint32_t i = 0; i < n_tokens; i++) {
                tokens[i] = (llama_token)(i % (uint32_t) n_vocab);
            }
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
        // v3 appends a block of captured intermediate tensors after the logits.
        // Backwards-compatible: when zero intermediates were captured, the
        // appended block is just (u32)0. v3 loaders need to accept v2.
        constexpr uint32_t FIXTURE_VERSION = 3;
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

        // ---- v3 block: captured intermediate tensors ----
        // Layout:
        //   u32 n_entries
        //   per entry:
        //     u32 name_len
        //     bytes name (utf-8)
        //     i64[4] ne
        //     u64 n_elements
        //     f32[n_elements] data
        const uint32_t n_entries = (uint32_t) capture.entries.size();
        write_pod(fx, n_entries);
        for (const auto & e : capture.entries) {
            const uint32_t name_len = (uint32_t) e.name.size();
            write_pod(fx, name_len);
            write_bytes(fx, e.name.data(), name_len);
            for (int i = 0; i < 4; i++) write_pod(fx, (int64_t) e.ne[i]);
            const uint64_t n_elements = (uint64_t) e.data.size();
            write_pod(fx, n_elements);
            write_bytes(fx, e.data.data(), n_elements * sizeof(float));
        }
        LOG_INF("captured %u intermediate tensors\n", n_entries);

        llama_batch_free(batch);
        fx.flush();
        LOG_INF("wrote fixture: %s (%u tokens, %d-vocab logits)\n",
                fixture_path.c_str(), n_tokens, n_vocab);
    }
    return 0;
}
