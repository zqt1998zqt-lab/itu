// Use the frozen input reader and diagnostic ABI for a per-hop exact A/B.
#define main itu_frozen_probe_main
#include "../validation-root/probe.cpp"
#undef main

#include <algorithm>
#include <array>
#include <memory>

namespace {
constexpr std::size_t kStateBytes = 2 * 1024 * 1024;

struct Bytes {
    explicit Bytes(std::size_t n) : data(n), stream(fmemopen(data.data(), n, "wb")) {
        if (!stream) throw std::runtime_error("fmemopen failed");
    }
    ~Bytes() { fclose(stream); }
    void rewind_file() {
        if (fseek(stream, 0, SEEK_SET)) throw std::runtime_error("state rewind failed");
    }
    std::size_t length() {
        if (fflush(stream) || ferror(stream)) throw std::runtime_error("state flush failed");
        auto n = ftell(stream);
        if (n < 0) throw std::runtime_error("state offset failed");
        return std::size_t(n);
    }
    std::vector<unsigned char> data;
    FILE* stream;
};

struct Provider {
    explicit Provider(const char* path) {
        lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!lib) throw std::runtime_error(dlerror());
    }
    template<class T> T get(const char* name) {
        auto* p = dlsym(lib, name);
        if (!p) throw std::runtime_error(std::string("missing symbol ") + name);
        return reinterpret_cast<T>(p);
    }
    ~Provider() { dlclose(lib); }
    void* lib;
};

struct Pipeline {
    Pipeline(Provider& p, const char* model, const char* mode) {
        if (setenv("ITU_LVQ_FUSION", mode, 1)) throw std::runtime_error("setenv failed");
        auto new_options = p.get<std::uintptr_t (*)()>("localvqe_options_new");
        auto options = new_options();
        if (!options ||
            p.get<int (*)(std::uintptr_t, const char*)>("localvqe_options_set_model_path")(options, model) ||
            p.get<int (*)(std::uintptr_t, int)>("localvqe_options_set_threads")(options, 1))
            throw std::runtime_error("invalid options");
        ctx = p.get<std::uintptr_t (*)(std::uintptr_t)>("localvqe_new_with_options")(options);
        p.get<void (*)(std::uintptr_t)>("localvqe_options_free")(options);
        if (!ctx || p.get<int (*)(std::uintptr_t, int, float)>("localvqe_set_noise_gate")(ctx, 0, -90.f))
            throw std::runtime_error(std::string("model setup failed: ") + mode);
        process = p.get<decltype(process)>("localvqe_process_frame_f32");
        dump = p.get<decltype(dump)>("itu_lvq_probe_dump");
        destroy = p.get<decltype(destroy)>("localvqe_free");
    }
    ~Pipeline() { destroy(ctx); }
    void step(const float* mic, const float* ref, bool emit_meta) {
        if (process(ctx, mic, ref, 256, pcm.data())) throw std::runtime_error("LocalVQE step failed");
        if (!std::all_of(pcm.begin(), pcm.end(), [](float v) { return std::isfinite(v); }))
            throw std::runtime_error("non-finite PCM");
        state.rewind_file();
        const int wrote = dump(ctx, state.stream, emit_meta ? metadata.stream : nullptr);
        if (wrote <= 0 || std::size_t(wrote) != state.length())
            throw std::runtime_error("state dump invalid");
        if (extent && extent != std::size_t(wrote)) throw std::runtime_error("state shape changed");
        extent = std::size_t(wrote);
    }
    std::uintptr_t ctx = 0;
    int (*process)(std::uintptr_t, const float*, const float*, int, float*) = nullptr;
    int (*dump)(std::uintptr_t, FILE*, FILE*) = nullptr;
    void (*destroy)(std::uintptr_t) = nullptr;
    Bytes state{kStateBytes}, metadata{32768};
    std::size_t extent = 0;
    std::array<float, 256> pcm{};
};
}

int main(int argc, char** argv) {
    try {
        if (argc != 8) throw std::runtime_error(
            "usage: state-probe BASE.so CANDIDATE.so model mono-mic.wav mono-ref.wav mode exact|negative");
        const auto mic = read_wave(argv[4]), ref = read_wave(argv[5]);
        constexpr std::size_t frames = 750;
        if (mic.pcm.size() != frames * 256 || mic.pcm.size() != ref.pcm.size())
            throw std::runtime_error("expected frozen 750-frame mono files");
        const std::string policy = argv[7];
        if (policy != "exact" && policy != "negative") throw std::runtime_error("unknown policy");
        Provider base_library(argv[1]), candidate_library(argv[2]);
        Pipeline base(base_library, argv[3], "baseline");
        Pipeline candidate(candidate_library, argv[3], argv[6]);
        std::size_t tensors = 0;
        for (std::size_t i = 0; i < frames; ++i) {
            base.step(mic.pcm.data() + i * 256, ref.pcm.data() + i * 256, i == 0);
            candidate.step(mic.pcm.data() + i * 256, ref.pcm.data() + i * 256, i == 0);
            if (base.extent != candidate.extent) throw std::runtime_error("state extent mismatch");
            if (i == 0) {
                const auto size = base.metadata.length();
                if (size != candidate.metadata.length() ||
                    std::memcmp(base.metadata.data.data(), candidate.metadata.data.data(), size))
                    throw std::runtime_error("state metadata mismatch");
                tensors = std::count(base.metadata.data.begin(),
                                     base.metadata.data.begin() + size, '\n') - 1;
                if (tensors != 77) throw std::runtime_error("state probe lost required tensors");
            }
            if (policy == "negative" && i == 12) candidate.state.data[128] ^= 1;
            auto differs = std::mismatch(base.state.data.begin(), base.state.data.begin() + base.extent,
                                         candidate.state.data.begin());
            if (differs.first != base.state.data.begin() + base.extent ||
                std::memcmp(base.pcm.data(), candidate.pcm.data(), base.pcm.size() * sizeof(float))) {
                std::fprintf(stderr, "STATE_MISMATCH mode=%s frame=%zu first_state_byte=%zu pcm_exact=%d\n",
                    argv[6], i, std::size_t(differs.first - base.state.data.begin()),
                    int(std::memcmp(base.pcm.data(), candidate.pcm.data(), base.pcm.size()*sizeof(float)) == 0));
                return 3;
            }
        }
        std::printf("STATE_PASS mode=%s frames=%zu tensors=%zu bytes_per_frame=%zu pcm_samples=%zu\n",
            argv[6], frames, tensors, base.extent, frames * 256);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 2;
    }
}
