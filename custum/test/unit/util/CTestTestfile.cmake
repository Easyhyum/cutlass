# CMake generated Testfile for 
# Source directory: /workspace/test/unit/util
# Build directory: /workspace/custum/test/unit/util
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[ctest_unit_util]=] "/workspace/custum/test/unit/util/cutlass_test_unit_util" "--gtest_output=xml:test_unit_util.gtest.xml")
set_tests_properties([=[ctest_unit_util]=] PROPERTIES  DISABLED "OFF" WORKING_DIRECTORY "./bin" _BACKTRACE_TRIPLES "/workspace/custum/test/unit/util/ctest/ctest_unit_util/CTestTestfile.ctest_unit_util.cmake;85;add_test;/workspace/custum/test/unit/util/ctest/ctest_unit_util/CTestTestfile.ctest_unit_util.cmake;0;;/workspace/CMakeLists.txt;1043;include;/workspace/test/unit/CMakeLists.txt;118;cutlass_add_executable_tests;/workspace/test/unit/util/CMakeLists.txt;29;cutlass_test_unit_add_executable;/workspace/test/unit/util/CMakeLists.txt;0;")
