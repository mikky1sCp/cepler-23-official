import subprocess
import sys
import os

scripts = [
    "benchmark_flops.py",
    "benchmark_scale_flops.py",
    "benchmark_time.py"
]

for script in scripts:
    script_path = os.path.join("scripts", script)
    print(f"\n===== Running {script_path} =====")
    subprocess.run([sys.executable, script_path], check=True)
