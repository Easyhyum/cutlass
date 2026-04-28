# CMake generated Testfile for 
# Source directory: /workspace/test/unit/gemm/threadblock
# Build directory: /workspace/custum/test/unit/gemm/threadblock
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[ctest_unit_gemm_threadblock]=] "/workspace/custum/test/unit/gemm/threadblock/cutlass_test_unit_gemm_threadblock" "--gtest_output=xml:test_unit_gemm_threadblock.gtest.xml")
set_tests_properties([=[ctest_unit_gemm_threadblock]=] PROPERTIES  DISABLED "OFF" WORKING_DIRECTORY "./bin" _BACKTRACE_TRIPLES "/workspace/custum/test/unit/gemm/threadblock/ctest/ctest_unit_gemm_threadblock/CTestTestfile.ctest_unit_gemm_threadblock.cmake;85;add_test;/workspace/custum/test/unit/gemm/threadblock/ctest/ctest_unit_gemm_threadblock/CTestTestfile.ctest_unit_gemm_threadblock.cmake;0;;/workspace/CMakeLists.txt;1043;include;/workspace/test/unit/CMakeLists.txt;118;cutlass_add_executable_tests;/workspace/test/unit/gemm/threadblock/CMakeLists.txt;29;cutlass_test_unit_add_executable;/workspace/test/unit/gemm/threadblock/CMakeLists.txt;0;")
