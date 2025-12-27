#include <stdint.h>

typedef enum GenMode {
	GM_INF,
	GM_NINF,
	GM_NAN,
	GM_NNAN,
	GM_ZERO,
	GM_NZERO,
	GM_RAND
} GenMode;

typedef uint64_t (*mx_generator)(GenMode, double, double);
typedef uint64_t (*mx_op_unary)(uint64_t);

uint64_t random_float(double min, double max);

uint64_t fp64_to_fp32(uint64_t a);
uint64_t fp32_to_fp16(uint64_t a);
uint64_t fp32_to_bf16(uint64_t a);
uint64_t bf16_to_e4m3(uint64_t a);
uint64_t bf16_to_e5m2(uint64_t a);

uint64_t gen_fp32(GenMode mode, double min, double max);
uint64_t gen_bf16(GenMode mode, double min, double max);