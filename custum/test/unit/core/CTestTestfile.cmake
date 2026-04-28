# CMake generated Testfile for 
# Source directory: /workspace/test/unit/core
# Build directory: /workspace/custum/test/unit/core
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[ctest_unit_core]=] "/workspace/custum/test/unit/core/cutlass_test_unit_core" "--gtest_output=xml:test_unit_core.gtest.xml")
set_tests_properties([=[ctest_unit_core]=] PROPERTIES  DISABLED "OFF" WORKING_DIRECTORY "./bin" _BACKTRACE_TRIPLES "/workspace/custum/test/unit/core/ctest/ctest_unit_core/CTestTestfile.ctest_unit_core.cmake;85;add_test;/workspace/custum/test/unit/core/ctest/ctest_unit_core/CTestTestfile.ctest_unit_core.cmake;0;;/workspace/CMakeLists.txt;1043;include;/workspace/test/unit/CMakeLists.txt;118;cutlass_add_executable_tests;/workspace/test/unit/core/CMakeLists.txt;29;cutlass_test_unit_add_executable;/workspace/test/unit/core/CMakeLists.txt;0;")
