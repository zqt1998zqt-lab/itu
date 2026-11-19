
#include "dispatch_audit.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static bool stub_result = true;
static unsigned int stub_calls;

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "%s:%d: ggml_abort called by verification stub\n", file, line);
    (void) fmt;
    abort();
}

bool llamafile_sgemm(
        const struct ggml_compute_params * params,
        int64_t m,
        int64_t n,
        int64_t k,
        const void * A,
        int64_t lda,
        const void * B,
        int64_t ldb,
        void * C,
        int64_t ldc,
        int atype,
        int btype,
        int ctype) {
    (void) params;
    (void) m;
    (void) n;
    (void) k;
    (void) A;
    (void) lda;
    (void) B;
    (void) ldb;
    (void) C;
    (void) ldc;
    (void) atype;
    (void) btype;
    (void) ctype;
    ++stub_calls;
    return stub_result;
}

static bool invoke(
        const struct ggml_compute_params * params,
        int64_t k) {
    return itu_lvq_sgemm_audit(
        params,
        1,
        1,
        k,
        NULL,
        k,
        NULL,
        k,
        NULL,
        1,
        GGML_TYPE_F32,
        GGML_TYPE_F32,
        GGML_TYPE_F32);
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODE REPORT_PATH\n", argv[0]);
        return 2;
    }
    const char * const mode = argv[1];
    const char * const report = argv[2];

    struct ggml_compute_params params;
    memset(&params, 0, sizeof(params));
    params.ith = 0;
    params.nth = 1;

    if (strcmp(mode, "off") == 0) {
        (void) unsetenv("ITU_LVQ_SGEMM_REPORT");
        stub_result = true;
        if (!invoke(&params, 8) || !invoke(&params, 8)) {
            return 20;
        }
        if (stub_calls != 2u) {
            return 21;
        }
        if (access(report, F_OK) == 0) {
            return 22;
        }
        return 0;
    }

    if (setenv("ITU_LVQ_SGEMM_REPORT", report, 1) != 0) {
        return 3;
    }

    if (strcmp(mode, "zero") == 0) {
        return 0;
    }

    if (strcmp(mode, "decline") == 0) {
        stub_result = false;
    } else {
        stub_result = true;
    }

    if (strcmp(mode, "invalid_nth") == 0) {
        params.nth = 2;
        if (!invoke(&params, 8)) {
            return 30;
        }
    } else if (strcmp(mode, "overflow") == 0) {
        for (int64_t k = 1; k <= 257; ++k) {
            if (!invoke(&params, k)) {
                return 31;
            }
        }
    } else {
        const bool expected = strcmp(mode, "decline") != 0;
        if (invoke(&params, 8) != expected || invoke(&params, 8) != expected) {
            return 32;
        }
    }

    if (strcmp(mode, "existing") != 0 && access(report, F_OK) == 0) {
        return 33;
    }
    return 0;
}
