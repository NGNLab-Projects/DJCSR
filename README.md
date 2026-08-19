# Resource optimization in O-RAN via D3QN

![O-RAN xApp](https://img.shields.io/badge/O--RAN-Near--RT%20RIC%20xApp-blue.svg)
![FlexRIC](https://img.shields.io/badge/FlexRIC-E2AP%20v2.0-orange.svg)
![OpenAirInterface](https://img.shields.io/badge/OAI-5G%20NR-red.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

An Open RAN (O-RAN) Near-Real-Time RIC xApp for **Dynamic Subcarrier Allocation** in 5G/6G Integrated Sensing and Communication (ISAC) networks using **Deep Dueling Double Q-Networks (D3QN)**, integrated with **OpenAirInterface (OAI)** and **FlexRIC**.

---

## 📌 Overview

As 5G-Advanced and 6G networks evolve toward **Integrated Sensing and Communication (ISAC)**, base stations (gNBs) must dynamically share spectrum and radio resource blocks (PRBs) between target radar sensing and multi-tenant communication services (**eMBB** and **URLLC**).

This repository contains two main sub-trees:
1. **`Jupyter_Python/`**: Standalone D3QN reinforcement learning simulation suite.
2. **`OAI_FlexRIC_ISAC_xAPP/`**: Live OpenAirInterface (OAI) & FlexRIC E2AP xApp deployment scripts.

---

## 📂 Repository Structure

```
PRB_Allocation_O-RAN_xAPP/
├── Jupyter_Python/
│   ├── agent.py              # D3QN Agent (Dueling Network, Replay Buffer)
│   ├── environment.py        # 5G ISAC Network Simulation Environment
│   └── train.py              # Training loop & evaluation script
├── OAI_FlexRIC_ISAC_xAPP/
│   ├── generate_traffic.sh   # iperf3 & ping traffic generator script
│   ├── isac_xapp.py          # OAI FlexRIC E2 xApp D3QN Controller
│   └── launch_topology.py    # OAI 5G NR RFSimulator Multi-gNB & UE launcher
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation
```

---

## 🛠️ System Parameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **gNB Topology** | 3 Base Stations | Line layout spaced 500m apart |
| **Antennas ($N_t$)** | 8 per gNB | Uniform Linear Array (ULA) |
| **Carrier Frequency ($f_c$)** | 3.5 GHz | 3GPP NR Band n78 |
| **Subcarrier Spacing ($SCS$)** | 30 kHz | Numerology $\mu = 1$ |
| **Total Subcarriers ($N_{sc}$)** | 780 | Total bandwidth $\approx 23.4$ MHz |
| **URLLC Users** | 5 UEs | Latency target: $<10$ ms |
| **eMBB Users** | 5 UEs | High throughput demand |

---

## 💻 Quick Start

### 1. Standalone Python Simulation (`Jupyter_Python/`)

To train the D3QN xApp agent in a simulated environment:

```bash
cd Jupyter_Python
python train.py
```

### 2. OAI FlexRIC Deployment (`OAI_FlexRIC_ISAC_xAPP/`)

To run the live xApp on an OpenAirInterface Near-RT RIC testbed:

```bash
cd OAI_FlexRIC_ISAC_xAPP

# 1. Launch network topology (3 gNBs + 10 UEs)
python launch_topology.py

# 2. Start traffic generation
bash generate_traffic.sh

# 3. Run the ISAC D3QN xApp Controller
python isac_xapp.py
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
