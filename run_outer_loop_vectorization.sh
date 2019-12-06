#!/bin/bash

printf "\n2048 x 2048 matrix multiplication with outer loop vectorization and the -O3 flag. The command being run is:\n\n"

printf "time ../../../build/bin/mlir-opt matmul.mlir -outer-loop-vectorize -virtual-vec-size 1024 --test-convert-vector-to-loops -convert-linalg-to-llvm | ../../../build/bin/mlir-cpu-runner -O3 -shared-libs=../../../build/lib/libcblas.so,../../../build/lib/libcblas_interface.so,../../../build/lib/libmlir_runner_utils.so
"
time ../../../build/bin/mlir-opt matmul.mlir -outer-loop-vectorize -virtual-vec-size 1024 --test-convert-vector-to-loops -convert-linalg-to-llvm | ../../../build/bin/mlir-cpu-runner -O3 -shared-libs=../../../build/lib/libcblas.so,../../../build/lib/libcblas_interface.so,../../../build/lib/libmlir_runner_utils.so



printf "\n\n\n2048 x 2048 matrix multiplication with the -O3 flag. The command being run is:\n\n"
printf "time ../../../build/bin/mlir-opt matmul.mlir -convert-linalg-to-llvm | ../../../build/bin/mlir-cpu-runner -O3 -shared-libs=../../../build/lib/libcblas.so,../../../build/lib/libcblas_interface.so,../../../build/lib/libmlir_runner_utils.so
"
time ../../../build/bin/mlir-opt matmul.mlir -convert-linalg-to-llvm | ../../../build/bin/mlir-cpu-runner -O3 -shared-libs=../../../build/lib/libcblas.so,../../../build/lib/libcblas_interface.so,../../../build/lib/libmlir_runner_utils.so

