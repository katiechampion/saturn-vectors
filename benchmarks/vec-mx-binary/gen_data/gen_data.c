#include <stdio.h>
#include <stdint.h>
#include "../../common-data-gen/mx_data_gen.h"

#define COUNT 128
#define SPECIAL_COUNT 20

void test(size_t nsize, size_t wsize, char *name, double min, double max, double sn_min, double sn_max, mx_generator gen, mx_op_binary op) {

	uint64_t vals_a[COUNT];
	uint64_t vals_b[COUNT];
	uint64_t vals_out[COUNT];

	for (size_t i = 0; i < COUNT; i ++) {
		vals_a[i] = gen(GM_RAND, min, max);
		vals_b[i] = gen(GM_RAND, min, max);
	}

	for (size_t i = 9; i < SPECIAL_COUNT; i ++) {
		vals_a[i] = gen(GM_RAND, sn_min, sn_max);
		vals_b[i] = gen(GM_RAND, sn_min, sn_max);
	}

	vals_a[0] = gen(GM_INF, min, max);
	vals_a[1] = gen(GM_NAN, min, max);

	vals_b[2] = gen(GM_INF, min, max);
	vals_b[3] = gen(GM_NAN, min, max);

	vals_a[4] = gen(GM_NINF, min, max);

	vals_a[5] = gen(GM_ZERO, min, max);
	vals_a[6] = gen(GM_NZERO, min, max);

	vals_a[7] = gen(GM_INF, min, max);
	vals_b[7] = gen(GM_NAN, min, max);

	vals_a[8] = gen(GM_INF, min, max);
	vals_b[8] = gen(GM_ZERO, min, max);

    for (size_t i = 0; i < COUNT; i ++) {
		vals_out[i] = op(vals_a[i], vals_b[i]);
	}

    printf(".global %s_a\n", name);
    printf(".balign 64\n");
    printf("%s_a:\n", name);
	for (size_t i = 0; i < COUNT / (4 / wsize); i ++) {
		printf("    .word 0x");
		for (int j = 4 / wsize - 1; j >= 0; j --) {
            printf("%0*X", wsize * 2, vals_a[i * (4 / wsize) + j]);
		}
		printf("\n");
	}

	printf(".global %s_b\n", name);
    printf(".balign 64\n");
    printf("%s_b:\n", name);
	for (size_t i = 0; i < COUNT / (4 / wsize); i ++) {
		printf("    .word 0x");
		for (int j = 4 / wsize - 1; j >= 0; j --) {
            printf("%0*X", wsize * 2, vals_b[i * (4 / wsize) + j]);
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

#define TESTS_FMA_BINARY(nsize, wsize, name, min, max, sn_min, sn_max, gen, ops_name) \
	test(nsize, nsize, name "_mul", min, max, sn_min, sn_max, gen, ops_name ## _mul); \
	test(nsize, nsize, name "_add", min, max, sn_min, sn_max, gen, ops_name ## _add); \
	test(nsize, nsize, name "_sub", min, max, sn_min, sn_max, gen, ops_name ## _sub); \
	test(wsize, nsize, name "_wmul", min, max, sn_min, sn_max, gen, ops_name ## _wmul); \
	test(wsize, nsize, name "_wadd", min, max, sn_min, sn_max, gen, ops_name ## _wadd); \
	test(wsize, nsize, name "_wsub", min, max, sn_min, sn_max, gen, ops_name ## _wsub);

int main() {

	printf(".section .data,\"aw\",@progbits\n");

    printf(".global N\n");
    printf(".balign 8\n");
    printf("N:\n");
    printf("    .word 0x%0*X\n", 8, COUNT);
    printf("    .word 0x00000000\n");

	TESTS_FMA_BINARY(2, 4, "fp16", -1e2, 1e2, -1e-6, 1e-6, gen_fp16, fp16)
	TESTS_FMA_BINARY(2, 4, "bf16", -1e15, 1e15, -2e-38, 2e-38, gen_bf16, bf16)

	/*
	Section from old tests for reference for decent min/max values
	
	test<fp32<rm>, 4, fp32name, -1e2, 1e2, -1e-6, 1e-6, false, fp32<rm>, rm_name>(); \
	test<fp16<rm>, 2, fp16name, -1e2, 1e2, -1e-6, 1e-6, false, fp16<rm>, rm_name>(); \
	test<bf16<rm>, 2, bf16name, -1e15, 1e15, -2e-38, 2e-38, false, bf16<rm>, rm_name>(); \
	test<ofp8e5m2<rm>, 1, ofp8e5m2name, -1e2, 1e2, -1e-4, 1e-4, false, ofp8e5m2<rm>, rm_name>(); \
	test<ofp8e4m3<rm>, 1, ofp8e4m3name, -3e1, 3e1, -1e-1, 1e-1, false, ofp8e4m3<rm>, rm_name>(); \
	test<fp16<rm>, 2, fp16Wname, -1e2, 1e2, -1e-6, 1e-6, true, fp32<rm>, rm_name>(); \
	test<bf16<rm>, 2, bf16Wname, -1e15, 1e15, -2e-38, 2e-38, true, fp32<rm>, rm_name>(); \
	test<ofp8e5m2<rm>, 1, ofp8e5m2Wname, -1e2, 1e2, -1e-4, 1e-4, true, bf16<rm>, rm_name>(); \
	test<ofp8e4m3<rm>, 1, ofp8e4m3Wname, -1e2, 1e2, -1e-3, 1e-3, true, bf16<rm>, rm_name>(); \
	*/

	return 0;
}