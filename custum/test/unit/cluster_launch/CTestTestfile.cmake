# CMake generated Testfile for 
# Source directory: /workspace/test/unit/cluster_launch
# Build directory: /workspace/custum/test/unit/cluster_launch
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[ctest_unit_cluster_launch]=] "/workspace/custum/test/unit/cluster_launch/cutlass_test_unit_cluster_launch" "--gtest_output=xml:test_unit_cluster_launch.gtest.xml")
set_tests_properties([=[ctest_unit_cluster_launch]=] PROPERTIES  DISABLED "OFF" WORKING_DIRECTORY "./bin" _BACKTRACE_TRIPLES "/workspace/custum/test/unit/cluster_launch/ctest/ctest_unit_cluster_launch/CTestTestfile.ctest_unit_cluster_launch.cmake;85;add_test;/workspace/custum/test/unit/cluster_launch/ctest/ctest_unit_cluster_launch/CTestTestfile.ctest_unit_cluster_launch.cmake;0;;/workspace/CMakeLists.txt;1043;include;/workspace/test/unit/CMakeLists.txt;118;cutlass_add_executable_tests;/workspace/test/unit/cluster_launch/CMakeLists.txt;29;cutlass_test_unit_add_executable;/workspace/test/unit/cluster_launch/CMakeLists.txt;0;")
