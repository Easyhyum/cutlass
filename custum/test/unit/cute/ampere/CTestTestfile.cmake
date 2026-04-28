# CMake generated Testfile for 
# Source directory: /workspace/test/unit/cute/ampere
# Build directory: /workspace/custum/test/unit/cute/ampere
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[ctest_unit_cute_ampere]=] "/workspace/custum/test/unit/cute/ampere/cutlass_test_unit_cute_ampere" "--gtest_output=xml:test_unit_cute_ampere.gtest.xml")
set_tests_properties([=[ctest_unit_cute_ampere]=] PROPERTIES  DISABLED "OFF" WORKING_DIRECTORY "./bin" _BACKTRACE_TRIPLES "/workspace/custum/test/unit/cute/ampere/ctest/ctest_unit_cute_ampere/CTestTestfile.ctest_unit_cute_ampere.cmake;85;add_test;/workspace/custum/test/unit/cute/ampere/ctest/ctest_unit_cute_ampere/CTestTestfile.ctest_unit_cute_ampere.cmake;0;;/workspace/CMakeLists.txt;1043;include;/workspace/test/unit/CMakeLists.txt;118;cutlass_add_executable_tests;/workspace/test/unit/cute/ampere/CMakeLists.txt;29;cutlass_test_unit_add_executable;/workspace/test/unit/cute/ampere/CMakeLists.txt;0;")
add_test([=[ctest_unit_cute_ampere_tiled_cp_async]=] "/workspace/custum/test/unit/cute/ampere/cutlass_test_unit_cute_ampere_tiled_cp_async" "--gtest_output=xml:test_unit_cute_ampere_tiled_cp_async.gtest.xml")
set_tests_properties([=[ctest_unit_cute_ampere_tiled_cp_async]=] PROPERTIES  DISABLED "OFF" WORKING_DIRECTORY "./bin" _BACKTRACE_TRIPLES "/workspace/custum/test/unit/cute/ampere/ctest/ctest_unit_cute_ampere_tiled_cp_async/CTestTestfile.ctest_unit_cute_ampere_tiled_cp_async.cmake;85;add_test;/workspace/custum/test/unit/cute/ampere/ctest/ctest_unit_cute_ampere_tiled_cp_async/CTestTestfile.ctest_unit_cute_ampere_tiled_cp_async.cmake;0;;/workspace/CMakeLists.txt;1043;include;/workspace/test/unit/CMakeLists.txt;118;cutlass_add_executable_tests;/workspace/test/unit/cute/ampere/CMakeLists.txt;37;cutlass_test_unit_add_executable;/workspace/test/unit/cute/ampere/CMakeLists.txt;0;")
