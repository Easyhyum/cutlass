# CMake generated Testfile for 
# Source directory: /workspace/test/unit/gemm/thread/host
# Build directory: /workspace/custum/test/unit/gemm/thread/host
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[ctest_unit_gemm_thread_host]=] "/workspace/custum/test/unit/gemm/thread/host/cutlass_test_unit_gemm_thread_host" "--gtest_output=xml:test_unit_gemm_thread_host.gtest.xml")
set_tests_properties([=[ctest_unit_gemm_thread_host]=] PROPERTIES  DISABLED "OFF" WORKING_DIRECTORY "./bin" _BACKTRACE_TRIPLES "/workspace/custum/test/unit/gemm/thread/host/ctest/ctest_unit_gemm_thread_host/CTestTestfile.ctest_unit_gemm_thread_host.cmake;85;add_test;/workspace/custum/test/unit/gemm/thread/host/ctest/ctest_unit_gemm_thread_host/CTestTestfile.ctest_unit_gemm_thread_host.cmake;0;;/workspace/CMakeLists.txt;1043;include;/workspace/test/unit/CMakeLists.txt;118;cutlass_add_executable_tests;/workspace/test/unit/gemm/thread/host/CMakeLists.txt;29;cutlass_test_unit_add_executable;/workspace/test/unit/gemm/thread/host/CMakeLists.txt;0;")
