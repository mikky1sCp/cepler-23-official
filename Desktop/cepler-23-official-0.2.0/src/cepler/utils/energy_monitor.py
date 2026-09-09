import pynvml  # используем pynvml (из nvidia-ml-py)
import time
import torch

class EnergyMonitor:
    def __init__(self, device_id=0):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        self.device_id = device_id
        self._start_energy = None
        self._start_time = None
        self.total_energy_joules = 0.0

    def start(self):
        pynvml.nvmlInit()
        self._start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        self._start_time = time.time()
        self.total_energy_joules = 0.0

    def stop(self):
        if self._start_energy is None:
            raise RuntimeError("Monitor not started. Call start() first.")
        end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        end_time = time.time()
        energy_joules = (end_energy - self._start_energy) / 1000.0
        elapsed_sec = end_time - self._start_time
        avg_power = energy_joules / elapsed_sec if elapsed_sec > 0 else 0.0
        self.total_energy_joules = energy_joules
        self.elapsed_sec = elapsed_sec
        self.avg_power_watts = avg_power
        return {
            'energy_joules': energy_joules,
            'energy_wh': energy_joules / 3600.0,
            'elapsed_sec': elapsed_sec,
            'avg_power_w': avg_power,
            'peak_power_w': self._get_peak_power()
        }

    def _get_peak_power(self):
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            return power
        except:
            return 0.0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @staticmethod
    def print_report(report):
        print("===== Energy Report =====")
        print(f"Total energy: {report['energy_wh']:.4f} Wh")
        print(f"Elapsed time: {report['elapsed_sec']:.2f} s")
        print(f"Average power: {report['avg_power_w']:.2f} W")
        print(f"Peak power (approx): {report['peak_power_w']:.2f} W")
        print("=========================")