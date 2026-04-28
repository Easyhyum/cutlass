# CMake generated Testfile for 
# Source directory: /workspace/test/unit/substrate
# Build directory: /workspace/custum/test/unit/substrate
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[ctest_unit_substrate]=] "/workspace/custum/test/unit/substrate/cutlass_test_unit_substrate" "--gtest_output=xml:test_unit_substrate.gtest.xml")
set_tests_properties([=[ctest_unit_substrate]=] PROPERTIES  DISABLED "OFF" WORKING_DIRECTORY "./bin" _BACKTRACE_TRIPLES "/workspace/custum/test/unit/substrate/ctest/ctest_unit_substrate/CTestTestfile.ctest_unit_substrate.cmake;85;add_test;/workspace/custum/test/unit/substrate/ctest/ctest_unit_substrate/CTestTestfile.ctest_unit_substrate.cmake;0;;/workspace/CMakeLists.txt;1043;include;/workspace/test/unit/CMakeLists.txt;118;cutlass_add_executable_tests;/workspace/test/unit/substrate/CMakeLists.txt;29;cutlass_test_unit_add_executable;/workspace/test/unit/substrate/CMakeLists.txt;0;")
