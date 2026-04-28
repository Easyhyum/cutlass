# CMake generated Testfile for 
# Source directory: /workspace/test/unit/layout
# Build directory: /workspace/custum/test/unit/layout
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[ctest_unit_layout]=] "/workspace/custum/test/unit/layout/cutlass_test_unit_layout" "--gtest_output=xml:test_unit_layout.gtest.xml")
set_tests_properties([=[ctest_unit_layout]=] PROPERTIES  DISABLED "OFF" WORKING_DIRECTORY "./bin" _BACKTRACE_TRIPLES "/workspace/custum/test/unit/layout/ctest/ctest_unit_layout/CTestTestfile.ctest_unit_layout.cmake;85;add_test;/workspace/custum/test/unit/layout/ctest/ctest_unit_layout/CTestTestfile.ctest_unit_layout.cmake;0;;/workspace/CMakeLists.txt;1043;include;/workspace/test/unit/CMakeLists.txt;118;cutlass_add_executable_tests;/workspace/test/unit/layout/CMakeLists.txt;29;cutlass_test_unit_add_executable;/workspace/test/unit/layout/CMakeLists.txt;0;")
