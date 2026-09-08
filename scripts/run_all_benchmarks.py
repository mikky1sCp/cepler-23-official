# scripts/run_all_benchmarks.py
import subprocess
import sys
import os

# Исправляем пути – запускаем скрипты из папки scripts/
scripts = [
    "benchmark_flops.py",
    "benchmark_scale_flops.py",
    "benchmark_time.py"
]

for script in scripts:
    script_path = os.path.join("scripts", script)
    print(f"\n===== Running {script_path} =====")
    subprocess.run([sys.executable, script_path], check=True)