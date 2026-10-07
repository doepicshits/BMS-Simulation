"""
Battery Management System (BMS) Simulator
------------------------------------------
Simulates a lithium-ion battery pack and monitors:
- Pack voltage and current
- State of Charge (SOC)
- State of Health (SOH)
- Cell temperature
- Thermal management / cooling fan
- Over-voltage, under-voltage, over-current and over-temperature protection

Author: Nivedan Kumar Yadav
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
import tracemalloc

@dataclass
class BatteryParameters:
    # Condensed dataclass using semicolons for vertical brevity
    cells_in_series: int = 4; cells_in_parallel: int = 2
    nominal_cell_voltage: float = 3.7; max_cell_voltage: float = 4.2; min_cell_voltage: float = 3.0
    nominal_cell_capacity_ah: float = 2.5; internal_resistance_ohm: float = 0.045
    max_discharge_current_a: float = 10.0; max_charge_current_a: float = 6.0
    ambient_temperature_c: float = 25.0; fan_on_temperature_c: float = 38.0
    maximum_temperature_c: float = 50.0; minimum_temperature_c: float = 0.0
    thermal_mass_j_per_c: float = 850.0; passive_cooling_w_per_c: float = 1.8; fan_cooling_w_per_c: float = 8.0
    capacity_fade_per_cycle: float = 0.0005; resistance_growth_per_cycle: float = 0.0008

class BMSSimulator:
    def __init__(self, p: BatteryParameters):
        self.p = p
        self.cap = p.nominal_cell_capacity_ah * p.cells_in_parallel
        self.v_nom, self.v_max, self.v_min = [v * p.cells_in_series for v in (p.nominal_cell_voltage, p.max_cell_voltage, p.min_cell_voltage)]
        self.r_base = p.internal_resistance_ohm * p.cells_in_series / p.cells_in_parallel
        self.soc, self.soh, self.t, self.cycles = 80.0, 100.0, p.ambient_temperature_c, 0.0
        self.fan, self.prot, self.history = False, False, []

    def simulate_step(self, t_s, i_a, dt):
        p = self.p
        
        # 1. Update SOC & SOH
        self.soc = np.clip(self.soc - (i_a * dt / 3600.0) / (self.cap * self.soh / 100.0) * 100, 0, 100)
        eq_cyc = (abs(i_a) * dt / 3600.0) / (2 * self.cap)
        self.cycles += eq_cyc
        self.soh = np.clip(self.soh - eq_cyc * p.capacity_fade_per_cycle * 100 * (1.0 + max(0, (self.t - 35) * 0.015)), 50, 100)
        
        # 2. Update Temperature
        r = self.r_base * (1 + (100 - self.soh) / 100.0)
        self.fan = True if self.t >= p.fan_on_temperature_c else (False if self.t <= p.fan_on_temperature_c - 3 else self.fan)
        cool = p.passive_cooling_w_per_c + (p.fan_cooling_w_per_c if self.fan else 0)
        self.t += ((i_a**2 * r) - cool * (self.t - p.ambient_temperature_c)) * dt / p.thermal_mass_j_per_c
        
        # 3. Calculate Voltage
        soc_n = self.soc / 100.0
        v = np.clip((3.0 + 1.15*soc_n + 0.05*np.sin(np.pi*soc_n)) * p.cells_in_series - i_a*r, self.v_min, self.v_max)
        
        # 4. Diagnostics & Protection
        checks = [(v <= self.v_min, "UNDER-VOLTAGE PROTECTION", True), (v >= self.v_max, "OVER-VOLTAGE PROTECTION", True),
                  (i_a > p.max_discharge_current_a, "OVER-CURRENT DURING DISCHARGE", True), 
                  (i_a < -p.max_charge_current_a, "OVER-CURRENT DURING CHARGING", True),
                  (self.t >= p.maximum_temperature_c, "OVER-TEMPERATURE PROTECTION", True), 
                  (self.t <= p.minimum_temperature_c, "LOW-TEMPERATURE PROTECTION", True),
                  (self.soc <= 5, "CRITICAL LOW SOC", False), (self.soh <= 70, "BATTERY HEALTH LOW", False)]
        
        w = [msg for c, msg, _ in checks if c]
        self.prot = any(trip for c, _, trip in checks if c)
        
        self.history.append({"time_s": t_s, "time_min": t_s/60, "current_a": i_a, "voltage_v": v, "power_w": v*i_a,
                             "soc_percent": self.soc, "soh_percent": self.soh, "temperature_c": self.t, 
                             "fan_status": int(self.fan), "protection_active": int(self.prot), 
                             "cycle_count": self.cycles, "warnings": "; ".join(w) if w else "Normal"})

def print_summary(res, bms):
    f = res.iloc[-1]
    print(f"\n{'='*55}\n{' '*16}BMS SIMULATION SUMMARY\n{'='*55}")
    print(f"Pack Configuration       : {bms.p.cells_in_series}S{bms.p.cells_in_parallel}P\nNominal Pack Voltage     : {bms.v_nom:.2f} V\nNominal Pack Capacity    : {bms.cap:.2f} Ah\n{'-'*55}")
    print(f"Final Voltage            : {f['voltage_v']:.2f} V\nFinal Current            : {f['current_a']:.2f} A\nFinal SOC                : {f['soc_percent']:.2f} %\nFinal SOH                : {f['soh_percent']:.2f} %\nFinal Temperature        : {f['temperature_c']:.2f} °C")
    print(f"Equivalent Cycle Count   : {f['cycle_count']:.4f}\nCooling Fan Status       : {'ON' if f['fan_status'] else 'OFF'}\nProtection Status        : {'ACTIVE' if f['protection_active'] else 'NORMAL'}\n{'='*55}")
    w = res[res["warnings"] != "Normal"]
    print(f"\nBMS WARNINGS DETECTED:\n{w[['time_min', 'warnings']].tail(10).to_string(index=False)}" if len(w) else "\nNo BMS protection warnings detected.")

def plot_results(df):
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(3, 2, figsize=(15, 13))
    fig.suptitle("Battery Management System (BMS) Simulation Dashboard", fontsize=16, fontweight="bold")
    
    cfg = [(0, 0, 'voltage_v', 'tab:blue', 'Pack Voltage', 'Voltage (V)', []),
           (0, 1, 'current_a', 'tab:orange', 'Current Profile', 'Current (A)', [(0, 'black', '-')]),
           (1, 0, 'soc_percent', 'tab:green', 'State of Charge (SOC)', 'SOC (%)', [(20, 'red', '--')]),
           (1, 1, 'soh_percent', 'tab:purple', 'State of Health (SOH)', 'SOH (%)', [(80, 'red', '--')]),
           (2, 0, 'temperature_c', 'tab:red', 'Thermal Management', 'Temperature (°C)', [(38, 'orange', '--'), (50, 'red', '--')])]
    
    for r, c, y, clr, title, ylab, hlines in cfg:
        ax[r,c].plot(df["time_min"], df[y], color=clr, lw=2)
        ax[r,c].set(title=title, xlabel="Time (minutes)", ylabel=ylab)
        for y_v, c_v, ls in hlines: ax[r,c].axhline(y_v, color=c_v, linestyle=ls)
        
    ax[0,1].text(0.02, 0.90, "Positive = Discharge\nNegative = Charge", transform=ax[0,1].transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    a = ax[2,1]
    a.step(df["time_min"], df["fan_status"], where="post", color="tab:cyan", lw=2, label="Cooling fan")
    a.step(df["time_min"], df["protection_active"], where="post", color="tab:red", lw=2, label="Protection active")
    a.set(title="BMS Control Outputs", xlabel="Time (minutes)", ylabel="Status", yticks=[0,1], yticklabels=["OFF", "ON"])
    a.legend(); plt.tight_layout(); plt.savefig("bms_simulation_dashboard.png", dpi=300); plt.show()

if __name__ == "__main__":
    # Start tracking memory allocations
    tracemalloc.start()
    
    bms = BMSSimulator(BatteryParameters())
    
    # drive cycle loop with vectorized numpy selection
    t_vals = np.arange(0, 3600, 1)
    c = t_vals % 600
    i_vals = np.select([c<120, c<240, c<360, c<420, c<520], [2., 6., 9., -3., 4.], default=1.)
    
    for t, i in zip(t_vals, i_vals): 
        bms.simulate_step(t, i, 1)
    
    res = pd.DataFrame(bms.history)
    res.to_csv("bms_simulation_results.csv", index=False)
    print_summary(res, bms)
    
    # Snapshot memory BEFORE Matplotlib renders (to see core logic size)
    current, peak = tracemalloc.get_traced_memory()
    print(f"\n[Memory Profiler] Peak memory BEFORE plotting: {peak / 1024 / 1024:.2f} MB")
    
    plot_results(res)
    
    # Snapshot memory AFTER Matplotlib
    current_final, peak_final = tracemalloc.get_traced_memory()
    print(f"[Memory Profiler] Peak memory AFTER plotting: {peak_final / 1024 / 1024:.2f} MB")
    
    tracemalloc.stop()
    
    print("\nFiles generated successfully:\n1. bms_simulation_results.csv\n2. bms_simulation_dashboard.png")
