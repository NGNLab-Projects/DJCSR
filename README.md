# PRB Allocation O-RAN xAPP via D3QN

![O-RAN xApp](https://img.shields.io/badge/O--RAN-Near--RT%20RIC%20xApp-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

An Open RAN (O-RAN) Near-Real-Time RIC xApp for **Dynamic Physical Resource Block (PRB) Allocation** in 5G/6G Integrated Sensing and Communication (ISAC) networks using **Deep Dueling Double Q-Networks (D3QN)**.

---

## 📌 Overview

As 5G-Advanced and 6G networks evolve toward **Integrated Sensing and Communication (ISAC)**, base stations (gNBs) must dynamically share spectrum and radio resource blocks (PRBs) between target radar sensing and multi-tenant communication services (**eMBB** and **URLLC**).

This xApp runs on the **Near-RT RIC (RAN Intelligent Controller)** to optimize resource allocation policies over the E2 interface, ensuring:
- **Radar Target Sensing**: Maximizing the Fisher Information Matrix (FIM) for high-precision range/velocity tracking.
- **eMBB Slicing**: Maximizing aggregate data throughput.
- **URLLC Slicing**: Guaranteeing sub-10ms ultra-reliable low-latency communications.

---

## 🚀 Key Features

- **D3QN Reinforcement Learning**: Combines Dueling Q-Network architecture with Double DQN updating to eliminate Q-value overestimation.
- **ISAC Simulation Environment**: 3GPP-aligned numerology (Band n78, 30 kHz SCS, 23.4 MHz bandwidth) with multi-gNB Uniform Linear Arrays (ULA).
- **Dynamic PRB Splitting**: Adaptive partitioning of subcarriers across radar sensing, URLLC, and eMBB slices.
- **Custom Multi-Objective Reward Function**: Balanced formulation combining FIM, throughput, and latency constraints.

---

## 📂 Repository Structure

```
PRB_Allocation_O-RAN_xAPP/
├── environment.py   # 5G ISAC Network Environment (Channel models, FIM, QoS metrics)
├── agent.py         # D3QN Agent implementation (Dueling Deep Q-Network, Replay Buffer)
├── train.py         # Training loop & performance metric evaluation script
├── .gitignore       # Git ignore rules for Python & PyTorch artifacts
└── README.md        # Project documentation
```

---

## 🛠️ System Parameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **gNB Topology** | 3 Base Stations | Line layout spaced 500m apart |
| **Antennas ($N_t$)** | 8 per gNB | Uniform Linear Array (ULA) with Maximum Ratio Transmission (MRT) |
| **Carrier Frequency ($f_c$)** | 3.5 GHz | 3GPP NR Band n78 |
| **Subcarrier Spacing ($SCS$)** | 30 kHz | Numerology $\mu = 1$ |
| **Total Subcarriers ($N_{sc}$)** | 780 | Total bandwidth $\approx 23.4$ MHz |
| **URLLC Users** | 5 UEs | Latency target: $<10$ ms |
| **eMBB Users** | 5 UEs | High throughput demand |

---

## 💻 Quick Start

### 1. Requirements

Ensure Python 3.9+ is installed along with the required libraries:

```bash
pip install numpy torch matplotlib
```

### 2. Run Training

To train the D3QN xApp agent and evaluate resource allocation performance:

```bash
python train.py
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
