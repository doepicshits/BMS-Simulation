# Battery Management System (BMS) Simulator using Python

A Python-based mini project that simulates a lithium-ion Battery Management System (BMS). The program monitors battery voltage, current, temperature, State of Charge (SOC), State of Health (SOH), and thermal protection status.

## Features

- Lithium-ion battery pack simulation
- Configurable 4S2P battery pack
- SOC estimation using Coulomb Counting
- SOH estimation based on capacity fade
- Internal resistance growth model
- Battery voltage estimation using an OCV-resistance model
- Thermal model using \(I^2R\) heat generation
- Passive cooling and cooling-fan thermal management
- BMS protection checks:
  - Over-voltage protection
  - Under-voltage protection
  - Over-current protection
  - Over-temperature protection
  - Low SOC warning
  - Low SOH warning
- Automatic CSV data logging
- Dashboard graph generation

## Battery Pack Configuration

| Parameter | Value |
|---|---:|
| Cell chemistry | Lithium-ion |
| Pack arrangement | 4S2P |
| Nominal cell voltage | 3.7 V |
| Maximum cell voltage | 4.2 V |
| Minimum cell voltage | 3.0 V |
| Cell capacity | 2.5 Ah |
| Nominal pack voltage | 14.8 V |
| Nominal pack capacity | 5.0 Ah |

## Theory

### State of Charge (SOC)

SOC indicates the available battery capacity as a percentage.


Where:

- (I) is the battery current in amperes
- Positive current indicates discharge
- Negative current indicates charging
- (C available) is the battery capacity adjusted according to SOH

### State of Health (SOH)

SOH measures the usable battery capacity compared with its original rated capacity.


SOH = (C current/C nominal)*100


The simulator reduces SOH gradually as the battery experiences charge-discharge throughput and high temperatures.

### Thermal Model

Heat generated due to internal resistance is:


Q generated = I^2R

The thermal model calculates the temperature rise based on generated heat, cooling losses, ambient temperature, and the cooling-fan state.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/bms-simulator.git
cd bms-simulator
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the simulation:

```bash
python bms_simulator.py
```

## Output

After simulation, the project generates:

- `bms_simulation_results.csv` — complete time-series BMS data
- `bms_simulation_dashboard.png` — voltage, current, SOC, SOH, temperature, fan, and protection plots

## Future Improvements

- Extended Kalman Filter (EKF) based SOC estimation
- Individual cell monitoring and cell balancing
- Real sensor integration using ESP32 and INA219
- MQTT/IoT cloud dashboard
- Machine-learning-based SOH prediction
- Real EV drive-cycle datasets
- Streamlit web dashboard

## Author

Nivedan Kumar Yadav  
B.E. Electronics and Communication Engineering  
Sant Longowal Institute of Engineering and Technology (SLIET)