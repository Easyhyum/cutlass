# CMake generated Testfile for 
# Source directory: /workspace/examples/63_hopper_gemm_with_weight_prefetch
# Build directory: /workspace/custum/examples/63_hopper_gemm_with_weight_prefetch
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[ctest_examples_63_hopper_gemm_with_weight_prefetch]=] "/workspace/custum/examples/63_hopper_gemm_with_weight_prefetch/63_hopper_gemm_with_weight_prefetch" "--m=8192" "--n=64" "--k=8192" "--iterations=0")
set_tests_properties([=[ctest_examples_63_hopper_gemm_with_weight_prefetch]=] PROPERTIES  DISABLED "OFF" WORKING_DIRECTORY "./bin" _BACKTRACE_TRIPLES "/workspace/custum/examples/63_hopper_gemm_with_weight_prefetch/ctest/ctest_examples_63_hopper_gemm_with_weight_prefetch/CTestTestfile.ctest_examples_63_hopper_gemm_with_weight_prefetch.cmake;85;add_test;/workspace/custum/examples/63_hopper_gemm_with_weight_prefetch/ctest/ctest_examples_63_hopper_gemm_with_weight_prefetch/CTestTestfile.ctest_examples_63_hopper_gemm_with_weight_prefetch.cmake;0;;/workspace/CMakeLists.txt;1043;include;/workspace/examples/CMakeLists.txt;71;cutlass_add_executable_tests;/workspace/examples/63_hopper_gemm_with_weight_prefetch/CMakeLists.txt;31;cutlass_example_add_executable;/workspace/examples/63_hopper_gemm_with_weight_prefetch/CMakeLists.txt;0;")
