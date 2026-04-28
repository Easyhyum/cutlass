export CUDA_VISIBLE_DEVICES=3

apt-get update && apt-get install -y libxcb1
python setup_bf16_sm80_kernel.py build_ext --inplace
python setup_bf16_custom.py build_ext --inplace
python setup_bf16_gemm.py build_ext --inplace
python setup_wmma_sleep.py build_ext --inplace