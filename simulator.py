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


@dataclass
class BatteryParameters:
    # Pack configuration
    cells_in_series: int = 4
    cells_in_parallel: int = 2

    # Cell electrical parameters
    nominal_cell_voltage: float = 3.7
    max_cell_voltage: float = 4.2
    min_cell_voltage: float = 3.0
    nominal_cell_capacity_ah: float = 2.5
    internal_resistance_ohm: float = 0.045

    # Current limits
    max_discharge_current_a: float = 10.0
    max_charge_current_a: float = 6.0

    # Temperature limits
    ambient_temperature_c: float = 25.0
    fan_on_temperature_c: float = 38.0
    maximum_temperature_c: float = 50.0
    minimum_temperature_c: float = 0.0

    # Thermal model
    thermal_mass_j_per_c: float = 850.0
    passive_cooling_w_per_c: float = 1.8
    fan_cooling_w_per_c: float = 8.0

    # Ageing model
    capacity_fade_per_cycle: float = 0.0005
    resistance_growth_per_cycle: float = 0.0008


class BMSSimulator:
    def __init__(self, parameters: BatteryParameters):
        self.p = parameters

        self.nominal_pack_capacity_ah = (
            self.p.nominal_cell_capacity_ah * self.p.cells_in_parallel
        )

        self.nominal_pack_voltage = (
            self.p.nominal_cell_voltage * self.p.cells_in_series
        )

        self.max_pack_voltage = (
            self.p.max_cell_voltage * self.p.cells_in_series
        )

        self.min_pack_voltage = (
            self.p.min_cell_voltage * self.p.cells_in_series
        )

        self.base_pack_resistance = (
            self.p.internal_resistance_ohm
            * self.p.cells_in_series
            / self.p.cells_in_parallel
        )

        self.soc = 80.0
        self.soh = 100.0
        self.temperature = self.p.ambient_temperature_c
        self.cycle_count = 0.0
        self.fan_status = False
        self.protection_active = False

        self.history = []

    def ocv_from_soc(self, soc):
        """
        Approximate lithium-ion Open Circuit Voltage (OCV) from SOC.
        Output is pack voltage.
        """
        soc = np.clip(soc, 0, 100)
        normalized_soc = soc / 100.0

        cell_ocv = (
            3.0
            + 1.15 * normalized_soc
            + 0.05 * np.sin(np.pi * normalized_soc)
        )

        return cell_ocv * self.p.cells_in_series

    def calculate_pack_resistance(self):
        """
        Internal resistance rises as SOH decreases.
        """
        degradation_factor = 1 + ((100 - self.soh) / 100.0)
        return self.base_pack_resistance * degradation_factor

    def calculate_terminal_voltage(self, current_a):
        """
        Current convention:
        Positive current  = discharge
        Negative current = charge
        """
        ocv = self.ocv_from_soc(self.soc)
        resistance = self.calculate_pack_resistance()

        terminal_voltage = ocv - (current_a * resistance)

        return np.clip(
            terminal_voltage,
            self.min_pack_voltage,
            self.max_pack_voltage
        )

    def update_soc(self, current_a, dt_seconds):
        """
        SOC estimation using Coulomb Counting.

        SOC_new = SOC_old - (I * dt / Capacity) * 100

        Positive current discharges the battery.
        Negative current charges the battery.
        """
        available_capacity_ah = (
            self.nominal_pack_capacity_ah * (self.soh / 100.0)
        )

        charge_change_ah = current_a * dt_seconds / 3600.0

        soc_change = (
            charge_change_ah / available_capacity_ah
        ) * 100.0

        self.soc -= soc_change
        self.soc = np.clip(self.soc, 0, 100)

    def update_soh(self, current_a, dt_seconds):
        """
        Simple degradation model:
        - Battery capacity decreases with equivalent full cycles.
        - Internal resistance increases as SOH falls.
        """
        throughput_ah = abs(current_a) * dt_seconds / 3600.0

        equivalent_cycle_increment = (
            throughput_ah / (2 * self.nominal_pack_capacity_ah)
        )

        self.cycle_count += equivalent_cycle_increment

        capacity_loss = (
            equivalent_cycle_increment
            * self.p.capacity_fade_per_cycle
            * 100
        )

        temperature_factor = 1.0

        if self.temperature > 35:
            temperature_factor += (self.temperature - 35) * 0.015

        self.soh -= capacity_loss * temperature_factor
        self.soh = np.clip(self.soh, 50, 100)

    def update_temperature(self, current_a, dt_seconds):
        """
        Lumped thermal model.

        Heat generated:
            Q = I^2 * R

        Temperature change:
            dT = (generated heat - cooling heat) * dt / thermal mass
        """
        resistance = self.calculate_pack_resistance()

        heat_generation_w = (current_a ** 2) * resistance

        if self.temperature >= self.p.fan_on_temperature_c:
            self.fan_status = True
        elif self.temperature <= self.p.fan_on_temperature_c - 3:
            self.fan_status = False

        cooling_coefficient = self.p.passive_cooling_w_per_c

        if self.fan_status:
            cooling_coefficient += self.p.fan_cooling_w_per_c

        heat_loss_w = (
            cooling_coefficient
            * (self.temperature - self.p.ambient_temperature_c)
        )

        net_heat_w = heat_generation_w - heat_loss_w

        temperature_change = (
            net_heat_w * dt_seconds / self.p.thermal_mass_j_per_c
        )

        self.temperature += temperature_change

    def check_protection(self, voltage, current_a):
        """
        Checks BMS safety thresholds.
        """
        warnings = []
        self.protection_active = False

        if voltage <= self.min_pack_voltage:
            warnings.append("UNDER-VOLTAGE PROTECTION")
            self.protection_active = True

        if voltage >= self.max_pack_voltage:
            warnings.append("OVER-VOLTAGE PROTECTION")
            self.protection_active = True

        if current_a > self.p.max_discharge_current_a:
            warnings.append("OVER-CURRENT DURING DISCHARGE")
            self.protection_active = True

        if abs(current_a) > self.p.max_charge_current_a and current_a < 0:
            warnings.append("OVER-CURRENT DURING CHARGING")
            self.protection_active = True

        if self.temperature >= self.p.maximum_temperature_c:
            warnings.append("OVER-TEMPERATURE PROTECTION")
            self.protection_active = True

        if self.temperature <= self.p.minimum_temperature_c:
            warnings.append("LOW-TEMPERATURE PROTECTION")
            self.protection_active = True

        if self.soc <= 5:
            warnings.append("CRITICAL LOW SOC")

        if self.soh <= 70:
            warnings.append("BATTERY HEALTH LOW")

        return warnings

    def simulate_step(self, time_s, current_a, dt_seconds):
        """
        Runs one complete BMS simulation step.
        """
        self.update_soc(current_a, dt_seconds)
        self.update_soh(current_a, dt_seconds)
        self.update_temperature(current_a, dt_seconds)

        voltage = self.calculate_terminal_voltage(current_a)
        power_w = voltage * current_a

        warnings = self.check_protection(voltage, current_a)

        self.history.append({
            "time_s": time_s,
            "time_min": time_s / 60,
            "current_a": current_a,
            "voltage_v": voltage,
            "power_w": power_w,
            "soc_percent": self.soc,
            "soh_percent": self.soh,
            "temperature_c": self.temperature,
            "fan_status": int(self.fan_status),
            "protection_active": int(self.protection_active),
            "cycle_count": self.cycle_count,
            "warnings": "; ".join(warnings) if warnings else "Normal"
        })

    def get_results(self):
        return pd.DataFrame(self.history)


def create_drive_cycle(total_time_s, dt_seconds):
    """
    Creates a realistic repeating load profile.

    Positive values: battery discharge
    Negative values: battery charging / regenerative braking
    """
    time = np.arange(0, total_time_s, dt_seconds)
    current_profile = []

    for t in time:
        cycle_time = t % 600

        if cycle_time < 120:
            current = 2.0

        elif cycle_time < 240:
            current = 6.0

        elif cycle_time < 360:
            current = 9.0

        elif cycle_time < 420:
            current = -3.0

        elif cycle_time < 520:
            current = 4.0

        else:
            current = 1.0

        current_profile.append(current)

    return time, np.array(current_profile)


def print_summary(results, simulator):
    final = results.iloc[-1]

    print("\n" + "=" * 55)
    print("          BMS SIMULATION SUMMARY")
    print("=" * 55)
    print(f"Pack Configuration       : {simulator.p.cells_in_series}S{simulator.p.cells_in_parallel}P")
    print(f"Nominal Pack Voltage     : {simulator.nominal_pack_voltage:.2f} V")
    print(f"Nominal Pack Capacity    : {simulator.nominal_pack_capacity_ah:.2f} Ah")
    print("-" * 55)
    print(f"Final Voltage            : {final['voltage_v']:.2f} V")
    print(f"Final Current            : {final['current_a']:.2f} A")
    print(f"Final SOC                : {final['soc_percent']:.2f} %")
    print(f"Final SOH                : {final['soh_percent']:.2f} %")
    print(f"Final Temperature        : {final['temperature_c']:.2f} °C")
    print(f"Equivalent Cycle Count   : {final['cycle_count']:.4f}")
    print(f"Cooling Fan Status       : {'ON' if final['fan_status'] else 'OFF'}")
    print(f"Protection Status        : {'ACTIVE' if final['protection_active'] else 'NORMAL'}")
    print("=" * 55)

    warning_rows = results[results["warnings"] != "Normal"]

    if len(warning_rows) > 0:
        print("\nBMS WARNINGS DETECTED:")
        print(warning_rows[["time_min", "warnings"]].tail(10).to_string(index=False))
    else:
        print("\nNo BMS protection warnings detected.")


def plot_results(results):
    """
    Creates dashboard-style plots and saves them as an image.
    """
    plt.style.use("seaborn-v0_8-darkgrid")

    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    fig.suptitle(
        "Battery Management System (BMS) Simulation Dashboard",
        fontsize=16,
        fontweight="bold"
    )

    axes[0, 0].plot(
        results["time_min"],
        results["voltage_v"],
        color="tab:blue",
        linewidth=2
    )
    axes[0, 0].set_title("Pack Voltage")
    axes[0, 0].set_xlabel("Time (minutes)")
    axes[0, 0].set_ylabel("Voltage (V)")

    axes[0, 1].plot(
        results["time_min"],
        results["current_a"],
        color="tab:orange",
        linewidth=2
    )
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Current Profile")
    axes[0, 1].set_xlabel("Time (minutes)")
    axes[0, 1].set_ylabel("Current (A)")
    axes[0, 1].text(
        0.02,
        0.90,
        "Positive = Discharge\nNegative = Charge",
        transform=axes[0, 1].transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    axes[1, 0].plot(
        results["time_min"],
        results["soc_percent"],
        color="tab:green",
        linewidth=2
    )
    axes[1, 0].axhline(20, color="red", linestyle="--", label="Low SOC threshold")
    axes[1, 0].set_title("State of Charge (SOC)")
    axes[1, 0].set_xlabel("Time (minutes)")
    axes[1, 0].set_ylabel("SOC (%)")
    axes[1, 0].legend()

    axes[1, 1].plot(
        results["time_min"],
        results["soh_percent"],
        color="tab:purple",
        linewidth=2
    )
    axes[1, 1].axhline(80, color="red", linestyle="--", label="Health threshold")
    axes[1, 1].set_title("State of Health (SOH)")
    axes[1, 1].set_xlabel("Time (minutes)")
    axes[1, 1].set_ylabel("SOH (%)")
    axes[1, 1].legend()

    axes[2, 0].plot(
        results["time_min"],
        results["temperature_c"],
        color="tab:red",
        linewidth=2,
        label="Battery temperature"
    )
    axes[2, 0].axhline(
        38,
        color="orange",
        linestyle="--",
        label="Fan ON threshold"
    )
    axes[2, 0].axhline(
        50,
        color="red",
        linestyle="--",
        label="Maximum temperature"
    )
    axes[2, 0].set_title("Thermal Management")
    axes[2, 0].set_xlabel("Time (minutes)")
    axes[2, 0].set_ylabel("Temperature (°C)")
    axes[2, 0].legend()

    axes[2, 1].step(
        results["time_min"],
        results["fan_status"],
        where="post",
        color="tab:cyan",
        linewidth=2,
        label="Cooling fan"
    )
    axes[2, 1].step(
        results["time_min"],
        results["protection_active"],
        where="post",
        color="tab:red",
        linewidth=2,
        label="Protection active"
    )
    axes[2, 1].set_title("BMS Control Outputs")
    axes[2, 1].set_xlabel("Time (minutes)")
    axes[2, 1].set_ylabel("Status")
    axes[2, 1].set_yticks([0, 1])
    axes[2, 1].set_yticklabels(["OFF", "ON"])
    axes[2, 1].legend()

    plt.tight_layout()
    plt.savefig("bms_simulation_dashboard.png", dpi=300)
    plt.show()


def main():
    parameters = BatteryParameters()
    bms = BMSSimulator(parameters)

    simulation_time_s = 3600
    dt_seconds = 1

    time_values, current_values = create_drive_cycle(
        simulation_time_s,
        dt_seconds
    )

    for time_s, current_a in zip(time_values, current_values):
        bms.simulate_step(time_s, current_a, dt_seconds)

    results = bms.get_results()

    results.to_csv("bms_simulation_results.csv", index=False)

    print_summary(results, bms)
    plot_results(results)

    print("\nFiles generated successfully:")
    print("1. bms_simulation_results.csv")
    print("2. bms_simulation_dashboard.png")


if __name__ == "__main__":
    main()
