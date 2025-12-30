#include <stdio.h>
#include <stdint.h>
#include "../../common-data-gen/mx_data_gen.h"

#define COUNT 128
#define SPECIAL_COUNT 20

void test(size_t nsize, size_t wsize, char *name, double min, double max, mx_generator gen, mx_op_unary op) {

	uint64_t vals[COUNT];
	uint64_t vals_out[COUNT];

	for (size_t i = 0; i < COUNT; i ++) {
		vals[i] = gen(GM_RAND, min, max);
	}

	vals[0] = gen(GM_INF, min, max);
	vals[1] = gen(GM_NAN, min, max);

	vals[2] = gen(GM_NINF, min, max);
	// vals[3] = gen(GM_NNAN, min, max);

	vals[4] = gen(GM_ZERO, min, max);
	vals[5] = gen(GM_NZERO, min, max);

    for (size_t i = 0; i < COUNT; i ++) {
		vals_out[i] = op(vals[i]);
	}

    printf(".global %s\n", name);
    printf(".balign 64\n");
    printf("%s:\n", name);
	for (size_t i = 0; i < COUNT / (4 / wsize); i ++) {
		printf("    .word 0x");
		for (int j = 4 / wsize - 1; j >= 0; j --) {
            printf("%0*X", wsize * 2, vals[i * (4 / wsize) + j]);
		}
		printf("\n");
	}

    printf(".global %s_out\n", name);
    printf(".balign 64\n");
    printf("%s_out:\n", name);
	for (size_t i = 0; i < COUNT / (4 / nsize); i ++) {
		printf("    .word 0x");
		for (int j = 4 / nsize - 1; j >= 0; j --) {
			printf("%0*X", nsize * 2, vals_out[i * (4 / nsize) + j]);
		}
		printf("\n");
	}
}

int main() {

	printf(".section .data,\"aw\",@progbits\n");

    printf(".global N\n");
    printf(".balign 8\n");
    printf("N:\n");
    printf("    .word 0x%0*X\n", 8, COUNT);
    printf("    .word 0x00000000\n");

    test(2, 4, "fp16_narrow", -1e2, 1e2, gen_fp32, fp32_to_fp16);
    test(2, 4, "bf16_narrow", -1e15, 1e15, gen_fp32, fp32_to_bf16);
    test(1, 2, "e5m2_narrow", -1e2, 1e2, gen_bf16, bf16_to_e5m2);
    test(1, 2, "e4m3_narrow", -3e1, 3e1, gen_bf16, bf16_to_e4m3);

    test(1, 2, "e5m2_narrow_sat", -1e2, 1e2, gen_bf16, bf16_to_e5m2_sat);
    test(1, 2, "e4m3_narrow_sat", -3e1, 3e1, gen_bf16, bf16_to_e4m3_sat);

	test(4, 2, "fp16_widen", -1e2, 1e2, gen_fp16, fp16_to_fp32);
    test(4, 2, "bf16_widen", -1e15, 1e15, gen_bf16, bf16_to_fp32);
    test(2, 1, "e5m2_widen", -1e2, 1e2, gen_e5m2, e5m2_to_bf16);
    test(2, 1, "e4m3_widen", -3e1, 3e1, gen_e4m3, e4m3_to_bf16);

	return 0;
}