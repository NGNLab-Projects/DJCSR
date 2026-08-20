# Resource Optimization in O-RAN via D3QN

![O-RAN xApp](https://img.shields.io/badge/O--RAN-Near--RT%20RIC%20xApp-blue.svg)
![FlexRIC](https://img.shields.io/badge/FlexRIC-E2AP%20v2.0-orange.svg)
![OpenAirInterface](https://img.shields.io/badge/OAI-5G%20NR-red.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

An Open RAN (O-RAN) Near-Real-Time RIC xApp for **Dynamic Subcarrier Allocation** in 5G/6G Integrated Sensing and Communication (ISAC) networks using **Deep Dueling Double Q-Networks (D3QN)**, integrated with **OpenAirInterface (OAI)** and **FlexRIC**.

---

## 📌 Overview

As 5G-Advanced and 6G networks evolve toward **Integrated Sensing and Communication (ISAC)**, base stations (gNBs) must dynamically share spectrum and subcarriers between target radar sensing and multi-tenant communication services (**eMBB** and **URLLC**).

This repository contains two main sub-trees:
1. **`Train Model`**: Standalone D3QN reinforcement learning simulation suite.
2. **`OAI_FlexRIC`**: Live OpenAirInterface (OAI) & FlexRIC E2AP xApp deployment scripts.

---

## 📂 Repository Structure

```
PRB_Allocation_O-RAN_xAPP
├── Train Model
│   ├── agent.py              # D3QN Agent (Dueling Network, Replay Buffer)
│   ├── environment.py        # 5G ISAC Network Simulation Environment
│   └── train.py              # Training loop & evaluation script
├── OAI_FlexRIC
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
| **URLLC Users** | 5 UEs | Latency target: $<5$ ms |
| **eMBB Users** | 5 UEs | High throughput demand |

---

## System Requirements
The requirements.txt file includes all required libraries for running DJCSR (1. Train Model) and the FlexRIC Python xApp (2. OAI_FlexRIC):

## 1. Python 3 Requirements

   Deep Reinforcement Learning & ML Frameworks: 
	torch>=2.0.0, torchvision, torchaudio, gymnasium>=0.28.1, gym>=0.26.0

   Numerical & Signal Processing: 
	numpy>=1.22.0,<2.0.0, scipy>=1.8.0

   Data Logging & Analysis: 
	pandas>=1.4.0, h5py>=3.6.0

   Visualization: 
	matplotlib>=3.5.0, seaborn>=0.11.2, tensorboard>=2.10.0

   FlexRIC SWIG/Python Bindings & Utilities: 
	cffi>=1.15.0, pyyaml>=6.0, pydantic>=1.10.0, requests>=2.28.0

## 2. How to Install OAI & FlexRIC
   System Build Prerequisites (Ubuntu 20.04/22.04 LTS)
   ```bash
   sudo apt update && sudo apt install -y \
    build-essential cmake ninja-build gcc g++ git bison flex swig \
    python3-dev python3-pip python3-numpy libsctp-dev libasn1c-dev ccache pcre2-utils
   ```
   Installing FlexRIC (O-RAN Near-RT RIC)
   ```bash
   cd ~
   git clone https://gitlab.eurecom.fr/oai/flexric.git
   cd flexric
   mkdir -p build && cd build
   cmake -DXAPP_MULTACELL=ON -DXAPP_PYTHON3=ON ..
   make -j$(nproc)
   sudo make install
   ```
   Installing OpenAirInterface 5G (OAI gNB with E2 Agent)
   ```bash
   cd ~
   git clone https://gitlab.eurecom.fr/oai/openairinterface5g.git
   cd openairinterface5g/cmake_targets
   ./build_oai -I
   ./build_oai --gNB -w SIMU --e2 -c
   ```
## 3. Necessary Documents & Reference Links
   OAI FlexRIC Repository: FlexRIC GitLab
   OAI FlexRIC Python xApp Guide: FlexRIC Python 3 xApp Docs
   OpenAirInterface 5G E2 Agent Manual: OAI E2AP README
   
  **O-RAN Alliance Working Group 3 (WG3) Specifications:**
      **O-RAN.WG3.E2AP:** E2 Application Protocol Architecture
      **O-RAN.WG3.E2SM-KPM:** Key Performance Indicators Monitoring Service Model
      **O-RAN.WG3.E2SM-RC / E2SM-MAC:** RAN Control & MAC Scheduling Service Model
   
## 💻 Quick Start

### 1. Train Model (Simulation)
The simulation pipeline uses `environment.py` (ISAC Environment) and `agent.py` (D3QN Architecture) to train the resource allocation policy.

```bash
cd “Train Model"
python train.py
# Output: d3qn_isac_final.pth
```

### 2. OAI / FlexRIC (Testbed Validation)

To validate the trained `d3qn_isac_final.pth` model on a live OpenAirInterface (OAI) & FlexRIC Near-RT RIC testbed, execute the following steps in sequence:

#### Step 1: Start 5G Core Network (OAI CN)
```bash
cd ~/openairinterface5g/oai-workshops/cn
docker compose up -d
docker ps
```
#### Step 2: Launch Near-RT RIC (FlexRIC)
```bash
sudo ~/openairinterface5g/openair2/E2AP/flexric/build/examples/ric/nearRT-RIC
```

#### Step 3: Launch Network Topology (3 gNBs + 10 UEs)
```bash
cd ~/openairinterface5g
sudo python3 launch_topology.py
```

#### Step 4: Generate Background Traffic (eMBB + URLLC)
```bash
sudo bash generate_traffic.sh
```

#### Step 5: Run ISAC xApp Controller
Run the xApp controller to perform real-time E2 control using the trained D3QN policy:
```bash
cd ~/openairinterface5g/openair2/E2AP/flexric/examples/xApp/python3/isac_xapp.py
```

#### Step 6: Plot & Verify Testbed Results
Generate throughput, latency, and sensing accuracy plots from metrics.csv:
```bash
cd ~/openairinterface5g
python3 plot_metrics.py
```
### 3. Results and Conclusion

#### 🔑 Results
1. **eMBB Throughput (+31.0% Gain)**:
   - DJCSR dynamically adapts subcarrier allocation ratios ($\rho$) and transmit power budgets ($p$), yielding a **31% throughput increase** over conventional DQN while consistently exceeding the **60 Mbps QoS target**.
2. **URLLC Latency (-16.7% Reduction)**:
   - Dynamic subcarrier allocation ratios ($\rho$) and joint power splitting ($p$) optimize resource assignment to maintain median URLLC latency at **3.93 ms** (a **16.7% reduction** over conventional DQN), strictly enforcing the **< 5 ms ultra-low latency constraint** under dynamic traffic bursts.
3. **Sensing Accuracy**:
   - By joint optimization of sensing subcarrier interleaved spacing ($\Delta_s$), DJCSR achieves an average range estimation RMSE of **0.184 m**.

#### 🏁 Conclusion
- **Adaptive Joint Resource Allocation**: DJCSR successfully resolves the fundamental trade-off between communication Quality-of-Service (QoS) and radar sensing precision in multi-cell 6G ISAC networks.
- **Hardware Feasibility**: Real-time E2 control loop validation on the FlexRIC Near-RT RIC and OpenAirInterface (OAI) gNB testbed confirms that D3QN-driven xApps operate seamlessly within sub-10ms Near-RT RIC decision epochs.
- **Scalability for 6G O-RAN**: The framework demonstrates that AI-native xApps can dynamically adjust subcarrier spacing, power splitting, and slice priorities in real time under dynamic fading and mobile traffic.
