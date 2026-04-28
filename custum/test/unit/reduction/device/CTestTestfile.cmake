# CMake generated Testfile for 
# Source directory: /workspace/test/unit/reduction/device
# Build directory: /workspace/custum/test/unit/reduction/device
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[ctest_unit_reduction_device]=] "/workspace/custum/test/unit/reduction/device/cutlass_test_unit_reduction_device" "--gtest_output=xml:test_unit_reduction_device.gtest.xml")
set_tests_properties([=[ctest_unit_reduction_device]=] PROPERTIES  DISABLED "OFF" WORKING_DIRECTORY "./bin" _BACKTRACE_TRIPLES "/workspace/custum/test/unit/reduction/device/ctest/ctest_unit_reduction_device/CTestTestfile.ctest_unit_reduction_device.cmake;85;add_test;/workspace/custum/test/unit/reduction/device/ctest/ctest_unit_reduction_device/CTestTestfile.ctest_unit_reduction_device.cmake;0;;/workspace/CMakeLists.txt;1043;include;/workspace/test/unit/CMakeLists.txt;118;cutlass_add_executable_tests;/workspace/test/unit/reduction/device/CMakeLists.txt;29;cutlass_test_unit_add_executable;/workspace/test/unit/reduction/device/CMakeLists.txt;0;")
