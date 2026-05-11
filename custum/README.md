export CUDA_VISIBLE_DEVICES=3

apt-get update && apt-get install -y libxcb1
python setup_bf16_sm80_kernel.py build_ext --inplace
python setup_bf16_custom.py build_ext --inplace
python setup_bf16_gemm.py build_ext --inplace
python setup_wmma_sleep.py build_ext --inplace

nvidia-smi -i 3   --query-gpu=timestamp,clocks.current.sm,power.draw.instant,power.draw.average,power.limit,clocks_event_reasons.sw_power_cap,clocks_event_reasons_counters.sw_power_cap   --format=csv   -lms 100 > gpu_power_log_5.csv

# ``nvidia-smi -q -d POWER`` 의 Power Samples Max 는 --query-gpu 로는 제공되지 않음.
# 로그에 찍힌 power.draw.instant 기준 누적 최대 열 추가: gpu_power_log_augment.py
#   python3 gpu_power_log_augment.py gpu_power_log_5.csv -o gpu_power_log_5_max.csv
#   nvidia-smi ... | python3 gpu_power_log_augment.py > out.csv