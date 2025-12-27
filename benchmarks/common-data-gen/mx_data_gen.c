#include <stdlib.h>
#include "mx_data_gen.h"
#include "rvv_mx.h"

uint64_t random_float(double min, double max) {
    double val = min + (max - min) * ((double) rand() / (double) RAND_MAX);
    return *(uint64_t *) &val;
}

// Narrowing conversion

uint64_t fp64_to_fp32(uint64_t a) {
    uint64_t res;
	asm volatile("fmv.d.x f0, %0" :: "r"(a));
	asm volatile("fcvt.s.d f1, f0");
	asm volatile("fmv.x.s %0, f1" : "=r"(res));
	return res & 0xFFFFFFFF;
}

uint64_t fp32_to_fp16(uint64_t a) {
	uint64_t res;
	asm volatile("fmv.s.x f0, %0" :: "r"(a));
	asm volatile("fcvt.h.s f1, f0");
	asm volatile("fmv.x.h %0, f1" : "=r"(res));
	return res & 0xFFFF;
}

uint64_t fp32_to_bf16(uint64_t a) {
	uint64_t res;
	asm volatile("fmv.s.x f0, %0" :: "r"(a));
    FCVT_BF16_S(F1, F0);
	asm volatile("fmv.x.h %0, f1" : "=r"(res));
	return res & 0xFFFF;
}

uint64_t bf16_to_e4m3(uint64_t a) {
    uint64_t res;
    VSETVLI_ALTFMT_X0(1, SEW_E16, LMUL_M1, 0);
    asm volatile("vmv.s.x v0, %0" :: "r"(a));
    VSETVLI_ALTFMT_X0(1, SEW_E8, LMUL_M1, 0);
    VFNCVTBF16_F_F_W(V8, V0);
    asm volatile("vmv.x.s %0, v8" : "=r"(res));
    return res & 0xFF;
}

uint64_t bf16_to_e5m2(uint64_t a) {
    uint64_t res;
    VSETVLI_ALTFMT_X0(1, SEW_E16, LMUL_M1, 0);
    asm volatile("vmv.s.x v0, %0" :: "r"(a));
    VSETVLI_ALTFMT_X0(1, SEW_E8, LMUL_M1, 1);
    VFNCVTBF16_F_F_W(V8, V0);
    asm volatile("vmv.x.s %0, v8" : "=r"(res));
    return res & 0xFF;
}

// Generation

uint64_t gen_fp32(GenMode mode, double min, double max) {
    switch (mode) {
        case GM_INF:
            return 0x7f800000;
        case GM_NINF:
            return 0xff800000;
        case GM_NAN:
            return 0x7fc00000;
        case GM_NNAN:
            return 0xffc00000;
        case GM_ZERO:
            return 0x00000000;
        case GM_NZERO:
            return 0x80000000;
        case GM_RAND:
            return fp64_to_fp32(random_float(min, max));
    }
}

uint64_t gen_bf16(GenMode mode, double min, double max) {
    switch (mode) {
        case GM_INF:
            return 0x7f80;
        case GM_NINF:
            return 0xff80;
        case GM_NAN:
            return 0x7fc0;
        case GM_NNAN:
            return 0xffc0;
        case GM_ZERO:
            return 0x0000;
        case GM_NZERO:
            return 0x8000;
        case GM_RAND:
            return fp32_to_bf16(fp64_to_fp32(random_float(min, max)));
    }
}