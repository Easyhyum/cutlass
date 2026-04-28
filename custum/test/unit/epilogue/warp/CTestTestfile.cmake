# CMake generated Testfile for 
# Source directory: /workspace/test/unit/epilogue/warp
# Build directory: /workspace/custum/test/unit/epilogue/warp
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[ctest_unit_epilogue_warp]=] "/workspace/custum/test/unit/epilogue/warp/cutlass_test_unit_epilogue_warp" "--gtest_output=xml:test_unit_epilogue_warp.gtest.xml")
set_tests_properties([=[ctest_unit_epilogue_warp]=] PROPERTIES  DISABLED "OFF" WORKING_DIRECTORY "./bin" _BACKTRACE_TRIPLES "/workspace/custum/test/unit/epilogue/warp/ctest/ctest_unit_epilogue_warp/CTestTestfile.ctest_unit_epilogue_warp.cmake;85;add_test;/workspace/custum/test/unit/epilogue/warp/ctest/ctest_unit_epilogue_warp/CTestTestfile.ctest_unit_epilogue_warp.cmake;0;;/workspace/CMakeLists.txt;1043;include;/workspace/test/unit/CMakeLists.txt;118;cutlass_add_executable_tests;/workspace/test/unit/epilogue/warp/CMakeLists.txt;29;cutlass_test_unit_add_executable;/workspace/test/unit/epilogue/warp/CMakeLists.txt;0;")
