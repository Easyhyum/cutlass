# CMake generated Testfile for 
# Source directory: /workspace/test/unit/cute/turing
# Build directory: /workspace/custum/test/unit/cute/turing
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[ctest_unit_cute_turing]=] "/workspace/custum/test/unit/cute/turing/cutlass_test_unit_cute_turing" "--gtest_output=xml:test_unit_cute_turing.gtest.xml")
set_tests_properties([=[ctest_unit_cute_turing]=] PROPERTIES  DISABLED "OFF" WORKING_DIRECTORY "./bin" _BACKTRACE_TRIPLES "/workspace/custum/test/unit/cute/turing/ctest/ctest_unit_cute_turing/CTestTestfile.ctest_unit_cute_turing.cmake;85;add_test;/workspace/custum/test/unit/cute/turing/ctest/ctest_unit_cute_turing/CTestTestfile.ctest_unit_cute_turing.cmake;0;;/workspace/CMakeLists.txt;1043;include;/workspace/test/unit/CMakeLists.txt;118;cutlass_add_executable_tests;/workspace/test/unit/cute/turing/CMakeLists.txt;29;cutlass_test_unit_add_executable;/workspace/test/unit/cute/turing/CMakeLists.txt;0;")
