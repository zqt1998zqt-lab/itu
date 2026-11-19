#ifndef ITU_LVQ_DISPATCH_AUDIT_H
#define ITU_LVQ_DISPATCH_AUDIT_H

/*
 * LocalVQE FP32 llamafile-SGEMM hit audit.
 *
 * ITU_LVQ_SGEMM_REPORT is sampled on the first wrapper invocation. When it is
 * absent or empty, calls go straight to llamafile_sgemm. When it is set, the
 * wrapper calls llamafile_sgemm exactly once, records the real return value,
 * and emits one CSV at explicit dump or destructor time. The report is opened
 * only if at least one audited attempt occurred, using O_CREAT|O_EXCL.
 *
 * The single-process/single-thread full-chain contract is nth == 1. Calls with
 * any other nth are retained in the totals but mark invalid_nth. At most 256
 * distinct (m,n,k,Atype,Btype,Ctype) shapes are retained; additional attempts
 * are counted in the summary's unrecorded_attempts field.
 */

#include "ggml-cpu-impl.h"
#include "llamafile/sgemm.h"

#include <fcntl.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ITU_LVQ_SGEMM_MAX_SHAPES 256u
#define ITU_LVQ_SGEMM_PATH_CAP 4096u

struct itu_lvq_sgemm_shape {
    int64_t m;
    int64_t n;
    int64_t k;
    int atype;
    int btype;
    int ctype;
    uint64_t attempt;
    uint64_t success;
    uint64_t decline;
    uint64_t invalid_nth;
    unsigned char counter_overflow;
};

static unsigned char itu_lvq_sgemm_lock_word;
static unsigned char itu_lvq_sgemm_initialized;
static unsigned char itu_lvq_sgemm_enabled;
static unsigned char itu_lvq_sgemm_dump_state;
static char itu_lvq_sgemm_report_path[ITU_LVQ_SGEMM_PATH_CAP];
static struct itu_lvq_sgemm_shape
    itu_lvq_sgemm_shapes[ITU_LVQ_SGEMM_MAX_SHAPES];
static size_t itu_lvq_sgemm_shape_count;
static uint64_t itu_lvq_sgemm_total_attempt;
static uint64_t itu_lvq_sgemm_total_success;
static uint64_t itu_lvq_sgemm_total_decline;
static uint64_t itu_lvq_sgemm_total_invalid_nth;
static uint64_t itu_lvq_sgemm_unrecorded_attempts;
static unsigned char itu_lvq_sgemm_counter_overflow;

static void itu_lvq_sgemm_lock_acquire(void) {
    while (__atomic_test_and_set(&itu_lvq_sgemm_lock_word, __ATOMIC_ACQUIRE)) {
    }
}

static void itu_lvq_sgemm_lock_release(void) {
    __atomic_clear(&itu_lvq_sgemm_lock_word, __ATOMIC_RELEASE);
}

static void itu_lvq_sgemm_add_u64(
        uint64_t * value,
        uint64_t increment,
        unsigned char * overflow) {
    if (*value > UINT64_MAX - increment) {
        *overflow = 1;
        return;
    }
    *value += increment;
}

static bool itu_lvq_sgemm_audit_enabled(void) {
    if (__atomic_load_n(&itu_lvq_sgemm_initialized, __ATOMIC_ACQUIRE)) {
        return __atomic_load_n(&itu_lvq_sgemm_enabled, __ATOMIC_ACQUIRE) != 0;
    }

    const char * report = getenv("ITU_LVQ_SGEMM_REPORT");
    itu_lvq_sgemm_lock_acquire();
    if (!__atomic_load_n(&itu_lvq_sgemm_initialized, __ATOMIC_RELAXED)) {
        const size_t length = report == NULL ? 0u : strlen(report);
        if (length > 0u && length < ITU_LVQ_SGEMM_PATH_CAP) {
            memcpy(itu_lvq_sgemm_report_path, report, length + 1u);
            __atomic_store_n(&itu_lvq_sgemm_enabled, 1u, __ATOMIC_RELAXED);
        }
        __atomic_store_n(&itu_lvq_sgemm_initialized, 1u, __ATOMIC_RELEASE);
    }
    itu_lvq_sgemm_lock_release();
    return __atomic_load_n(&itu_lvq_sgemm_enabled, __ATOMIC_ACQUIRE) != 0;
}

static struct itu_lvq_sgemm_shape * itu_lvq_sgemm_find_or_add_shape(
        int64_t m,
        int64_t n,
        int64_t k,
        int atype,
        int btype,
        int ctype) {
    size_t index = 0;
    for (; index < itu_lvq_sgemm_shape_count; ++index) {
        struct itu_lvq_sgemm_shape * const shape = &itu_lvq_sgemm_shapes[index];
        if (shape->m == m &&
            shape->n == n &&
            shape->k == k &&
            shape->atype == atype &&
            shape->btype == btype &&
            shape->ctype == ctype) {
            return shape;
        }
    }
    if (index >= ITU_LVQ_SGEMM_MAX_SHAPES) {
        return NULL;
    }

    struct itu_lvq_sgemm_shape * const shape = &itu_lvq_sgemm_shapes[index];
    memset(shape, 0, sizeof(*shape));
    shape->m = m;
    shape->n = n;
    shape->k = k;
    shape->atype = atype;
    shape->btype = btype;
    shape->ctype = ctype;
    itu_lvq_sgemm_shape_count = index + 1u;
    return shape;
}

static void itu_lvq_sgemm_record(
        const struct ggml_compute_params * params,
        int64_t m,
        int64_t n,
        int64_t k,
        int atype,
        int btype,
        int ctype,
        bool result) {
    if (!itu_lvq_sgemm_audit_enabled()) {
        return;
    }

    itu_lvq_sgemm_lock_acquire();
    itu_lvq_sgemm_add_u64(
        &itu_lvq_sgemm_total_attempt,
        1u,
        &itu_lvq_sgemm_counter_overflow);

    const bool invalid_nth = params == NULL || params->nth != 1;
    if (invalid_nth) {
        itu_lvq_sgemm_add_u64(
            &itu_lvq_sgemm_total_invalid_nth,
            1u,
            &itu_lvq_sgemm_counter_overflow);
    }
    if (result) {
        itu_lvq_sgemm_add_u64(
            &itu_lvq_sgemm_total_success,
            1u,
            &itu_lvq_sgemm_counter_overflow);
    } else {
        itu_lvq_sgemm_add_u64(
            &itu_lvq_sgemm_total_decline,
            1u,
            &itu_lvq_sgemm_counter_overflow);
    }

    struct itu_lvq_sgemm_shape * const shape =
        itu_lvq_sgemm_find_or_add_shape(m, n, k, atype, btype, ctype);
    if (shape == NULL) {
        itu_lvq_sgemm_add_u64(
            &itu_lvq_sgemm_unrecorded_attempts,
            1u,
            &itu_lvq_sgemm_counter_overflow);
    } else {
        itu_lvq_sgemm_add_u64(
            &shape->attempt,
            1u,
            &shape->counter_overflow);
        if (invalid_nth) {
            itu_lvq_sgemm_add_u64(
                &shape->invalid_nth,
                1u,
                &shape->counter_overflow);
        }
        if (result) {
            itu_lvq_sgemm_add_u64(
                &shape->success,
                1u,
                &shape->counter_overflow);
        } else {
            itu_lvq_sgemm_add_u64(
                &shape->decline,
                1u,
                &shape->counter_overflow);
        }
    }
    itu_lvq_sgemm_lock_release();
}

static inline bool itu_lvq_sgemm_audit(
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
    if (!itu_lvq_sgemm_audit_enabled()) {
        return llamafile_sgemm(
            params, m, n, k, A, lda, B, ldb, C, ldc, atype, btype, ctype);
    }

    const bool result = llamafile_sgemm(
        params, m, n, k, A, lda, B, ldb, C, ldc, atype, btype, ctype);
    itu_lvq_sgemm_record(params, m, n, k, atype, btype, ctype, result);
    return result;
}

void itu_lvq_sgemm_audit_dump(void);

void itu_lvq_sgemm_audit_dump(void) {
    itu_lvq_sgemm_lock_acquire();
    if (itu_lvq_sgemm_dump_state != 0u ||
        itu_lvq_sgemm_total_attempt == 0u ||
        itu_lvq_sgemm_enabled == 0u) {
        itu_lvq_sgemm_lock_release();
        return;
    }
    itu_lvq_sgemm_dump_state = 1u;

    int flags = O_WRONLY | O_CREAT | O_EXCL;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    const int fd = open(itu_lvq_sgemm_report_path, flags, 0666);
    if (fd < 0) {
        itu_lvq_sgemm_dump_state = 2u;
        itu_lvq_sgemm_lock_release();
        return;
    }

    FILE * const stream = fdopen(fd, "w");
    if (stream == NULL) {
        (void) close(fd);
        itu_lvq_sgemm_dump_state = 2u;
        itu_lvq_sgemm_lock_release();
        return;
    }

    (void) fputs(
        "row_type,slot,m,n,k,atype,btype,ctype,attempt,success,decline,"
        "invalid_nth,unrecorded_attempts,counter_overflow\n",
        stream);
    for (size_t index = 0; index < itu_lvq_sgemm_shape_count; ++index) {
        const struct itu_lvq_sgemm_shape * const shape =
            &itu_lvq_sgemm_shapes[index];
        (void) fprintf(
            stream,
            "shape,%zu,%" PRId64 ",%" PRId64 ",%" PRId64
            ",%d,%d,%d,%" PRIu64 ",%" PRIu64 ",%" PRIu64
            ",%" PRIu64 ",0,%u\n",
            index,
            shape->m,
            shape->n,
            shape->k,
            shape->atype,
            shape->btype,
            shape->ctype,
            shape->attempt,
            shape->success,
            shape->decline,
            shape->invalid_nth,
            shape->counter_overflow != 0u ? 1u : 0u);
    }
    (void) fprintf(
        stream,
        "summary,,,,,,,,%" PRIu64 ",%" PRIu64 ",%" PRIu64
        ",%" PRIu64 ",%" PRIu64 ",%u\n",
        itu_lvq_sgemm_total_attempt,
        itu_lvq_sgemm_total_success,
        itu_lvq_sgemm_total_decline,
        itu_lvq_sgemm_total_invalid_nth,
        itu_lvq_sgemm_unrecorded_attempts,
        itu_lvq_sgemm_counter_overflow != 0u ? 1u : 0u);
    (void) fclose(stream);

    itu_lvq_sgemm_dump_state = 2u;
    itu_lvq_sgemm_lock_release();
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((destructor))
static void itu_lvq_sgemm_audit_destructor(void) {
    itu_lvq_sgemm_audit_dump();
}
#endif

#ifdef __cplusplus
}
#endif

#endif
