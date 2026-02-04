#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "rvv_mx.h"
#include "bme.h"

#define STALL(n) for (int __i = 0; __i < n; __i ++) { asm volatile("nop"); }

// #define VDOTSET_VV(rd, as) asm volatile(".insn r 0x77, 0x0, 0x00, " rd ", x0, " as)
// #define VDOTSETZERO_VV(as) asm volatile(".insn r 0x77, 0x0, 0x00, x0, x1, " as)
// #define VDOTSETZEROBC_VV() asm volatile(".insn r 0x77, 0x0, 0x00, x0, x3, x0")
// #define VDOTWB_VV(rd, as) asm volatile(".insn r 0x77, 0x0, 0x02, " rd ", x0, " as)
// #define VQLDOTUA_VV(as, rs2, rs1) asm volatile(".insn r 0x77, 0x0, 0x4c, " as ", " rs1 ", " rs2)
// #define VQLDOTSA_VV(as, rs2, rs1) asm volatile(".insn r 0x77, 0x0, 0x4e, " as ", " rs1 ", " rs2)

// #define OPMVIN(md, vs2, rs1) asm volatile(".insn r 0x57, 0x6, 0x55, " md ", %0, " vs2 : : "r"(rs1));
// #define OPMVOUT(vd, ms2, rs1) asm volatile(".insn r 0x57, 0x6, 0x5d, " vd ", %0, " ms2 : : "r"(rs1));
// #define OPMVINBCAST(md, vs2) asm volatile(".insn r 0x57, 0x6, 0x59, " md ", x0, " vs2);
// #define OPMACC(md, vs2, vs1) asm volatile(".insn r 0x57, 0x2, 0x51, " md ", " vs1 ", " vs2);

extern const size_t N;
extern uint8_t a[] __attribute__((aligned(64)));
extern uint8_t b[] __attribute__((aligned(64)));
extern uint32_t r[] __attribute__((aligned(64)));
extern uint32_t rt[] __attribute__((aligned(64)));

#define BLOCK_SIZE 16

void matmul_opu() {
    int cycles_start;
    int cycles_end;
    uint32_t res[N * N];
    memset(res, 0, N * N * sizeof(uint32_t));
    int vl;
    asm volatile("csrr %0, cycle" : "=r"(cycles_start));

    VSETVLI_ALTFMT(vl, N, SEW_E8, LMUL_M1, 0);
    for (int i = 0; i < N; i += vl) {
        for (int j = 0; j < N; j += vl) {
            VSETVLI_ALTFMT_X0(vl, SEW_E32, LMUL_M4, 0);
            asm volatile("vmv.v.i v0, 0");
            OPMVINBCAST("x1", "x0");
            VSETVLI_ALTFMT_X0(vl, SEW_E8, LMUL_M1, 1);
            for (int k = 0; k < N; k ++) {
                asm volatile("vle8.v v0, (%0)" :: "r"(a + i + k * N));
                asm volatile("vle8.v v1, (%0)" :: "r"(b + j + k * N));
                OPFMACC("x1", "x1", "x0");
            }
            VSETVLI_ALTFMT_X0(vl, SEW_E32, LMUL_M4, 0);
            for (int l = 0; l < vl; l ++) {
                OPMVOUT("x0", "x1", l);
                asm volatile("vle32.v v4, (%0)" :: "r"(res + (i + l) * N + j));
                asm volatile("vadd.vv v0, v0, v4");
                asm volatile("vle32.v v0, (%0)" :: "r"(res + (i + l) * N + j));
            }
        }
    }

    asm volatile("fence");
    asm volatile("csrr %0, cycle" : "=r"(cycles_end));
    printf("Cycles (OPU): %d\n", cycles_end - cycles_start);
    // for (int i = 0; i < N * N; i ++) {
    //     if (res[i] != rt[i]) {
    //         printf("Bad value at index %d: got %d, expected %d\n", i, res[i], r[i]);
    //         exit(1);
    //     }
    // }
}

void matmul_vector_inner() {
    int cycles_start;
    int cycles_end;
    uint32_t res[N * N];
    memset(res, 0, N * N * sizeof(uint32_t));
    int vl;
    asm volatile("csrr %0, cycle" : "=r"(cycles_start));

    VSETVLI_ALTFMT_X0(1, SEW_E32, LMUL_M1, 0);
    asm volatile("vmv.v.i v28, 0");
    VSETVLI_ALTFMT(vl, N, SEW_E8, LMUL_M1, 0);
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int ii = 0; ii < BLOCK_SIZE; ii ++) {
                for (int jj = 0; jj < BLOCK_SIZE; jj ++) {
                    VSETVLI_ALTFMT_X0(vl, SEW_E16, LMUL_M2, 0);
                    asm volatile("vmv.v.i v0, 0");
                    VSETVLI_ALTFMT_X0(vl, SEW_E8, LMUL_M1, 0);
                    for (int k = 0; k < N; k += vl) {
                        asm volatile("vle8.v v16, (%0)" :: "r"(a + (i + ii) * N + k));
                        asm volatile("vle8.v v24, (%0)" :: "r"(b + (j + jj) * N + k));
                        asm volatile("vwmaccu.vv v0, v16, v24");
                    }
                    VSETVLI_ALTFMT_X0(vl, SEW_E16, LMUL_M2, 0);
                    asm volatile("vwredsumu.vs v16, v0, v28");
                    VSETVLI_ALTFMT_X0(1, SEW_E32, LMUL_M1, 0);
                    int e0;
                    asm volatile("vmv.x.s %0, v16" : "=r"(e0));
                    res[(i + ii) * N + j + jj] = e0;
                    VSETVLI_ALTFMT_X0(vl, SEW_E8, LMUL_M1, 0);
                }
            }
        }
    }

    asm volatile("fence");
    asm volatile("csrr %0, cycle" : "=r"(cycles_end));
    printf("Cycles (Vector Inner): %d\n", cycles_end - cycles_start);
    for (int i = 0; i < N * N; i ++) {
        if (res[i] != r[i]) {
            printf("Bad value at index %d: got %d, expected %d\n", i, res[i], r[i]);
            exit(1);
        }
    }
}

void matmul_scalar() {
    int cycles_start;
    int cycles_end;
    uint32_t res[N * N];
    memset(res, 0, N * N * sizeof(uint32_t));
    asm volatile("csrr %0, cycle" : "=r"(cycles_start));

    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int k = 0; k < N; k ++) {
                for (int ii = 0; ii < BLOCK_SIZE; ii ++) {
                    for (int jj = 0; jj < BLOCK_SIZE; jj ++) {
                        res[(i + ii) * N + j + jj] += a[(i + ii) * N + k] * b[(j + jj) * N + k];
                    }
                }
            }
        }
    }

    asm volatile("fence");
    asm volatile("csrr %0, cycle" : "=r"(cycles_end));
    printf("Cycles (Scalar): %d\n", cycles_end - cycles_start);
    for (int i = 0; i < N * N; i ++) {
        if (res[i] != r[i]) {
            printf("Bad value at index %d: got %d, expected %d\n", i, res[i], r[i]);
            exit(1);
        }
    }
}

int main() {

    matmul_opu();
    matmul_vector_inner();
    matmul_scalar();
    exit(0);

    int res;
    int a = 128;
    int vl;
    int cycles_start;
    int cycles_end;

    VSETVLI_ALTFMT_X0(a, SEW_E32, LMUL_M2, 0);
    // vd
    asm volatile("vmv.v.i v24, 1");
    VSETVLI_ALTFMT(vl, a, SEW_E8, LMUL_M1, 0);
    // vs2
    asm volatile("vmv.v.i v8, 2");
    asm volatile("vmv.v.i v9, 3");
    asm volatile("vmv.v.i v10, 4");
    asm volatile("vmv.v.i v11, 5");
    asm volatile("vmv.v.i v12, 6");
    asm volatile("vmv.v.i v13, 7");
    asm volatile("vmv.v.i v14, 8");
    asm volatile("vmv.v.i v15, 9");
    // vs1
    VSETVLI_ALTFMT_X0(a, SEW_E8, LMUL_M4, 0);
    asm volatile("vmv.v.i v16, 3");

    asm volatile("csrr %0, cycle" : "=r"(cycles_start));

    VSETVLI_ALTFMT_X0(8, SEW_E32, LMUL_M1, 0);
    VDOTSET_VV("x24", "x1");
    VDOTSETZERO_VV("x0");
    VSETVLI_ALTFMT_X0(a, SEW_E8, LMUL_M1, 0);
    VSETVLI_ALTFMT_X0(8, SEW_E32, LMUL_M1, 0);
    VDOTWB_VV("x24", "x0");
    VDOTWB_VV("x16", "x1");

    // exit(0);

    asm volatile("vmv.x.s x0, v24"); // Wait for writeback
    asm volatile("fence");
    asm volatile("csrr %0, cycle" : "=r"(cycles_end));

    for (int i = 0; i < 8; i ++) {
        asm volatile("vmv.x.s %0, v16" : "=r"(res));
        printf("Result %d: %d\n", i, res);
        asm volatile("vslidedown.vi v16, v16, 1");
    }

    // exit(0);
    printf("Cycles: %d\n", cycles_end - cycles_start);
    printf("VL: %d\n", vl);
    for (int i = 0; i < 8; i ++) {
        asm volatile("vmv.x.s %0, v24" : "=r"(res));
        printf("Result %d: %d\n", i, res);
        asm volatile("vslidedown.vi v24, v24, 1");
    }

    // VSETVLI_ALTFMT_X0(a, SEW_E8, LMUL_M2, 1);
    // asm volatile("vmv.v.i v8, 5");
    // asm volatile("vmv.v.i v16, 3");
    // VQLDOTSA_VV("x24", "x8", "x16");
    // // STALL(100);
    // asm volatile("vmv.v.i v16, 4");
    // VQLDOTUA_VV("x24", "x8", "x16");
    // asm volatile("vsetvli zero, a0, e32, m1");
    // asm volatile("vmv.x.s %0, v24" : "=r"(res));
    // printf("Result: %d\n", res);

    return 0;
}