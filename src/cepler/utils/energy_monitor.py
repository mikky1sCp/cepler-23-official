import time
import torch

class EnergyMonitor:

    def __init__(self, device_id=0):
        self.device_id = device_id
        self._start_energy = None
        self._start_time = None
        self.total_energy_joules = 0.0
        self._nvml_available = False
        self._pynvml = None
        self.handle = None

        if torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                self._nvml_available = True
                self._pynvml = pynvml
                print("EnergyMonitor: NVML initialized successfully.")
            except Exception as e:
                print(f"EnergyMonitor: NVML init failed: {e}. Energy monitoring disabled.")
        else:
            print("EnergyMonitor: CUDA not available. Energy monitoring disabled.")

    def start(self):
        if not self._nvml_available:
            self._start_time = time.time()
            return
        self._start_energy = self._pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        self._start_time = time.time()

    def stop(self):
        if not self._nvml_available:
            end_time = time.time()
            return {
                'energy_joules': 0.0,
                'energy_wh': 0.0,
                'elapsed_sec': end_time - self._start_time,
                'avg_power_w': 0.0,
                'peak_power_w': 0.0
            }
        if self._start_energy is None:
            raise RuntimeError("Monitor not started. Call start() first.")
        end_energy = self._pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
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
        if not self._nvml_available:
            return 0.0
        try:
            power = self._pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
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