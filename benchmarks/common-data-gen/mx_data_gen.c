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

uint64_t bf16_to_e4m3_sat(uint64_t a) {
    uint64_t res;
    VSETVLI_ALTFMT_X0(1, SEW_E16, LMUL_M1, 0);
    asm volatile("vmv.s.x v0, %0" :: "r"(a));
    VSETVLI_ALTFMT_X0(1, SEW_E8, LMUL_M1, 0);
    VFNCVTBF16_SAT_F_F_W(V8, V0);
    asm volatile("vmv.x.s %0, v8" : "=r"(res));
    return res & 0xFF;
}

uint64_t bf16_to_e5m2_sat(uint64_t a) {
    uint64_t res;
    VSETVLI_ALTFMT_X0(1, SEW_E16, LMUL_M1, 0);
    asm volatile("vmv.s.x v0, %0" :: "r"(a));
    VSETVLI_ALTFMT_X0(1, SEW_E8, LMUL_M1, 1);
    VFNCVTBF16_SAT_F_F_W(V8, V0);
    asm volatile("vmv.x.s %0, v8" : "=r"(res));
    return res & 0xFF;
}

// Widening conversion

uint64_t fp16_to_fp32(uint64_t a) {
	uint64_t res;
	asm volatile("fmv.h.x f0, %0" :: "r"(a));
	asm volatile("fcvt.s.h f1, f0");
	asm volatile("fmv.x.s %0, f1" : "=r"(res));
	return res & 0xFFFFFFFF;
}

uint64_t bf16_to_fp32(uint64_t a) {
	uint64_t res;
	asm volatile("fmv.h.x f0, %0" :: "r"(a));
    FCVT_S_BF16(F1, F0);
	asm volatile("fmv.x.s %0, f1" : "=r"(res));
	return res & 0xFFFFFFFF;
}

uint64_t e4m3_to_bf16(uint64_t a) {
    uint64_t res;
    VSETVLI_ALTFMT_X0(1, SEW_E8, LMUL_M1, 0);
    asm volatile("vmv.s.x v0, %0" :: "r"(a));
    VFWCVTBF16_F_F_V(V8, V0);
    VSETVLI_ALTFMT_X0(1, SEW_E16, LMUL_M1, 0);
    asm volatile("vmv.x.s %0, v8" : "=r"(res));
    return res & 0xFFFF;
}

uint64_t e5m2_to_bf16(uint64_t a) {
    uint64_t res;
    VSETVLI_ALTFMT_X0(1, SEW_E8, LMUL_M1, 1);
    asm volatile("vmv.s.x v0, %0" :: "r"(a));
    VFWCVTBF16_F_F_V(V8, V0);
    VSETVLI_ALTFMT_X0(1, SEW_E16, LMUL_M1, 0);
    asm volatile("vmv.x.s %0, v8" : "=r"(res));
    return res & 0xFFFF;
}

// FMA Binary

#define OP_BINARY(name, op, inst, isew, osew, esew, alt, mask) \
    uint64_t name ## _ ## op(uint64_t a, uint64_t b) { \
        uint64_t res; \
        VSETVLI_ALTFMT_X0(1, isew, LMUL_M1, 0); \
        asm volatile("vmv.s.x v0, %0" :: "r"(a)); \
        asm volatile("vmv.s.x v16, %0" :: "r"(b)); \
        VSETVLI_ALTFMT_X0(1, esew, LMUL_M1, alt); \
        asm volatile(inst " v8, v0, v16"); \
        VSETVLI_ALTFMT_X0(1, osew, LMUL_M1, 0); \
        asm volatile("vmv.x.s %0, v8" : "=r"(res)); \
        return res & mask; \
    }

#define OPS_FMA_BINARY(name, sew, wsew, alt, mask, wmask) \
    OP_BINARY(name, mul, "vfmul.vv", sew, sew, sew, alt, mask) \
    OP_BINARY(name, add, "vfadd.vv", sew, sew, sew, alt, mask) \
    OP_BINARY(name, sub, "vfsub.vv", sew, sew, sew, alt, mask) \
    OP_BINARY(name, wmul, "vfwmul.vv", sew, wsew, sew, alt, wmask) \
    OP_BINARY(name, wadd, "vfwadd.vv", sew, wsew, sew, alt, wmask) \
    OP_BINARY(name, wsub, "vfwsub.vv", sew, wsew, sew, alt, wmask)

OPS_FMA_BINARY(fp32, SEW_E32, SEW_E64, 0, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF)
OPS_FMA_BINARY(fp16, SEW_E16, SEW_E32, 0, 0xFFFF, 0xFFFFFFFF)
OPS_FMA_BINARY(bf16, SEW_E16, SEW_E32, 1, 0xFFFF, 0xFFFFFFFF)

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

uint64_t gen_fp16(GenMode mode, double min, double max) {
    switch (mode) {
        case GM_INF:
            return 0x7c00;
        case GM_NINF:
            return 0xfc00;
        case GM_NAN:
            return 0x7e00;
        case GM_NNAN:
            return 0xfe00;
        case GM_ZERO:
            return 0x0000;
        case GM_NZERO:
            return 0x8000;
        case GM_RAND:
            return fp32_to_fp16(fp64_to_fp32(random_float(min, max)));
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

uint64_t gen_e4m3(GenMode mode, double min, double max) {
    switch (mode) {
        case GM_INF:
            return 0x7f;
        case GM_NINF:
            return 0xff;
        case GM_NAN:
            return 0x7f;
        case GM_NNAN:
            return 0xff;
        case GM_ZERO:
            return 0x00;
        case GM_NZERO:
            return 0x80;
        case GM_RAND:
            return bf16_to_e4m3(fp32_to_bf16(fp64_to_fp32(random_float(min, max))));
    }
}

uint64_t gen_e5m2(GenMode mode, double min, double max) {
    switch (mode) {
        case GM_INF:
            return 0x7c;
        case GM_NINF:
            return 0xfc;
        case GM_NAN:
            return 0x7f;
        case GM_NNAN:
            return 0xff;
        case GM_ZERO:
            return 0x00;
        case GM_NZERO:
            return 0x80;
        case GM_RAND:
            return bf16_to_e5m2(fp32_to_bf16(fp64_to_fp32(random_float(min, max))));
    }
}