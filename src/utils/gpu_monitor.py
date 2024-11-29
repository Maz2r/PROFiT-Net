# src/utils/gpu_monitor.py

import pynvml

def log_gpu_usage():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    print(f"GPU Memory Used: {mem_info.used / (1024 ** 2):.2f} MB, GPU Utilization: {gpu_util.gpu}%")
    pynvml.nvmlShutdown()
