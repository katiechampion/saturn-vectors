#!/usr/bin/env bash

# g++ gen_data.cpp -I ../common/LoFloat/src -I ../vec-mx-fma -std=c++20 -Wno-overflow -o gen_data 
# ./gen_data > data.S
# rm gen_data

ISA=rv64imafdcv_zfh
SPIKE_ISA=${ISA}_zfbfmin_zvfofp8min

riscv64-unknown-elf-gcc gen_data.c ../../common-data-gen/mx_data_gen.c -I ../../common -I ../../common-data-gen -march=$ISA -o gen_data
spike --isa=$SPIKE_ISA ${LOG:+-l --log=$LOG} pk gen_data > ../data.S
rm gen_data