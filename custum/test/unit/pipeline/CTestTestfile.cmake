# CMake generated Testfile for 
# Source directory: /workspace/test/unit/pipeline
# Build directory: /workspace/custum/test/unit/pipeline
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[ctest_unit_pipeline]=] "/workspace/custum/test/unit/pipeline/cutlass_test_unit_pipeline" "--gtest_output=xml:test_unit_pipeline.gtest.xml")
set_tests_properties([=[ctest_unit_pipeline]=] PROPERTIES  DISABLED "OFF" WORKING_DIRECTORY "./bin" _BACKTRACE_TRIPLES "/workspace/custum/test/unit/pipeline/ctest/ctest_unit_pipeline/CTestTestfile.ctest_unit_pipeline.cmake;85;add_test;/workspace/custum/test/unit/pipeline/ctest/ctest_unit_pipeline/CTestTestfile.ctest_unit_pipeline.cmake;0;;/workspace/CMakeLists.txt;1043;include;/workspace/test/unit/CMakeLists.txt;118;cutlass_add_executable_tests;/workspace/test/unit/pipeline/CMakeLists.txt;43;cutlass_test_unit_add_executable;/workspace/test/unit/pipeline/CMakeLists.txt;0;")
