# Resource Optimization in O-RAN via D3QN

![O-RAN xApp](https://img.shields.io/badge/O--RAN-Near--RT%20RIC%20xApp-blue.svg)
![FlexRIC](https://img.shields.io/badge/FlexRIC-E2AP%20v2.0-orange.svg)
![OpenAirInterface](https://img.shields.io/badge/OAI-5G%20NR-red.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

An Open RAN (O-RAN) Near-Real-Time RIC xApp for **Dynamic Subcarrier Allocation** in 5G/6G Integrated Sensing and Communication (ISAC) networks using **DJCSR  (D3QN-Based Joint Communication–Sensing Resource Optimization) Framework** by **Deep Dueling Double Q-Networks (D3QN)**, integrated with **OpenAirInterface (OAI)** and **FlexRIC**.

---

## 📌 Overview

As 5G-Advanced and 6G networks evolve toward **Integrated Sensing and Communication (ISAC)**, base stations (gNBs) must dynamically share spectrum and subcarriers between target radar sensing and multi-tenant communication services (**eMBB** and **URLLC**).

---

## 📂 Repository Structure

```
├── Train_Model
│   ├── agent.py              # D3QN Agent
│   ├── environment.py        # 5G ISAC Network Simulation Environment
│   └── train.py              # Training loop & evaluation script
├── OAI_FlexRIC
│   ├── generate_traffic.sh   # iperf3 & ping traffic generator script
│   └── launch_topology.py    # OAI 5G NR RFSimulator Multi-gNB & UE launcher
│   └── d3qn_isac_final.pth   # Trained Inference
├── examples                  # xApp for RAN Slicing and information monitor
│   ├── emulator             
│   ├── ric                   # RIC including E2 interface
│   └── xApp                  # Source code for xApps
│       ├── c                 # xApps based on C code
│       └── python3           # xApps based on Python code
│           └── xapp_isac.py  # Source code for OAI FlexRIC E2 xApp D3QN Controller
├── oai-workshops
│   ├── cn
│   └── ran
│       └── conf
│    		└── isac_setup	  # gNBs and UEs Configuration
├── openair2
│   ├── E2AP
│   │   ├── RAN_FUNCTION
│   │   │   └── CUSTOMIZED	  # Monitor Functions
│   │   │       ├── ran_func_mac.c   <-- Thread-safe MAC stats harvester
│   │   │       └── ran_func_kpm.c   <-- RLC SDU delay & UE metrics
│   │   └── flexric           # FlexRIC Near-RT RIC & Python SWIG (xapp_sdk)
│   │
│   └── LAYER2
│   	└── NR_MAC_gNB
│        	├── slicing
│        	│   └── slicing.c     	 <-- NVS Subcarrier Reservation
│           └── gNB_scheduler_dlsch.c <-- Downlink MAC Scheduler & DLSCH PDU Assembly
├── Metrics					  # OAI/FlexRIC Metrics
│   ├── plot_metrics.py       # Plot the metrics
│   ├── metrics.csv           # Metrics of each epoch
├── fig						  # OAI/FlexRIC Testbed results
├── requirement.txt           # Necessary Libraries and Packages
├── .gitignore                # Git ignore rules
├── CMakeLists.txt            # CMakeLists
├── LICENSE                	  # Licence
├── flexric.conf              # FlexRIC Configurations
└── README.md                 # Project documentation
```

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

## ⚙️ System Requirements
The requirements.txt file includes all required libraries for running DJCSR (Train_Model) and the FlexRIC Python xApp (OAI_FlexRIC):  

### 1. Python 3 Requirements
   - Deep Reinforcement Learning & ML Frameworks:  
		torch>=2.0.0, torchvision, torchaudio, gymnasium>=0.28.1, gym>=0.26.0
   - Numerical & Signal Processing:  
		numpy>=1.22.0,<2.0.0, scipy>=1.8.0
   - Data Logging & Analysis:  
		pandas>=1.4.0, h5py>=3.6.0
   - Visualization:  
		matplotlib>=3.5.0, seaborn>=0.11.2, tensorboard>=2.10.0
   - FlexRIC SWIG/Python Bindings & Utilities:  
		cffi>=1.15.0, pyyaml>=6.0, pydantic>=1.10.0, requests>=2.28.0

```bash
pip install -r requirements.txt
```
### 2. How to Install Docker on Ubuntu

- **Add Docker's official GNU Privacy Guard (GPG) key:**
```bash
	sudo mkdir -p /etc/apt/keyrings
	curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```
- **Set up the Docker stable repository:**  
```bash
	echo \
	  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
	$(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
	sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```
- **Install Docker Engine, CLI, Containerd, and Docker Compose Plugin:**  
```bash
	sudo apt-get update
	sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```
- **Verify Docker installation:**  
```bash
	docker --version
	docker compose version
```
### 3. How to Install [OAI](https://gitlab.eurecom.fr/oai/openairinterface5g.git) & [FlexRIC](https://gitlab.eurecom.fr/mosaic5g/flexric.git)
- **System Build Prerequisites (Ubuntu 22.04 LTS):**  
```bash
	sudo apt update && sudo apt install -y \
	build-essential cmake ninja-build gcc-12 g++-12 git bison flex swig \
	python3-dev python3-pip python3-numpy libsctp-dev libpcre2-dev ccache
```
- **Clone & Build FlexRIC (v1.0, 2024.w12):**
```bash
  	git clone https://gitlab.eurecom.fr/mosaic5g/flexric.git
	cd flexric
	git fetch --all --tags
	mkdir -p build && cd build
	cmake -DCMAKE_C_COMPILER=gcc-12 \
	      -DCMAKE_CXX_COMPILER=g++-12 \
	      -DXAPP_MULTICELL=ON \
	      -DXAPP_PYTHON3=ON ..
	make -j$(nproc)
	sudo make install
	sudo ldconfig
```
- **Clone & Build OpenAirInterface 5G (v2024.w24):**  
```bash
	cd ~
	git clone https://gitlab.eurecom.fr/oai/openairinterface5g.git
	cd openairinterface5g
	git fetch --all --tags
```
```bash
	cd cmake_targets
	# -I installs missing OAI packages, --build-e2 links against installed FlexRIC
	./build_oai -I -w SIMU --gNB --nrUE --build-e2 --ninja
```
---
## Quick Start

### 1. Train Model (Simulation)
The simulation pipeline uses `environment.py` (ISAC Environment) and `agent.py` (D3QN Architecture) to train the resource allocation policy.
```bash
cd “1. Train Model"
python train.py
# Output: d3qn_isac_final.pth
```
### 2. OAI / FlexRIC (Testbed Validation)
To validate the trained `d3qn_isac_final.pth` model on a live OpenAirInterface (OAI) & FlexRIC Near-RT RIC testbed, execute the following steps in sequence:

#### Step 1: Start 5G Core Network (OAI CN) containers (MySQL, AMF, SMF, UPF):
```bash
cd ~/oai-workshops/cn
docker compose up -d
docker ps
```
#### Step 2: Launch Near-RT RIC (FlexRIC)
```bash
sudo ~/openair2/E2AP/flexric/build/examples/ric/nearRT-RIC
```
#### Step 3: Start a gNB
```bash
	cd ~/openairinterface5g/cmake_targets/ran_build/build
	sudo ./nr-softmodem -O ~/openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf --rfsim -E --gNBs.[0].min_rxtxtime 6
```
#### Step 4: Start a UE
  ```bash
	sudo ./nr-uesoftmodem -r 106 --numerology 1 --band 78 -C 3619200000 --rfsim -O
	~/openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/ue.conf -E --num-ues 10
  ```
#### Step 5: Launch Network Topology (3 gNBs + 10 UEs)
```bash
sudo python3 launch_topology.py
```
#### Step 6: Generate Background Traffic (eMBB + URLLC)
```bash
sudo bash generate_traffic.sh
```
##### You can use the following commands to check gNB and UEs Information:
```bash
docker logs oai-amf -f	
```
##### gNBs and UEs Information:
```bash
[2026-09-01 13:55:35.592] [amf_app] [info] 
   |------------------------------------------------------------------------------------------------------------------------------------------------------------|
   |----------------------------------------------------------------------gNBs' Information---------------------------------------------------------------------|
   |  Index |               Status               |              Global Id             |              gNB Name              |                PLMN                |
   |    1   |              Connected             |               0x0E00               |              gNB-OAI-1             |               001,01               |
   |    2   |              Connected             |               0x0E01               |              gNB-OAI-2             |               001,01               |
   |    3   |              Connected             |               0x0E02               |              gNB-OAI-3             |               001,01               |
   |------------------------------------------------------------------------------------------------------------------------------------------------------------|

   |-----------------------------------------------------------------------------------------------------------------------------------------------------------|
   |---------------------------------------------------------------------UEs' Information----------------------------------------------------------------------|
   |  Index |     5GMM State     |        IMSI        |        GUTI        |   RAN UE NGAP ID   |   AMF UE NGAP ID   |        PLMN        |       Cell Id      |
   |    1   |   5GMM-REGISTERED  |   001010000000100  |00101010041259605347|        0x01        |        0x01        |       001,01       |      0000e014f     |
   |    2   |   5GMM-REGISTERED  |   001010000000101  |00101010041026984396|        0x01        |        0x02        |       001,01       |      0000e0ec6     |
   |    3   |   5GMM-REGISTERED  |   001010000000102  |00101010041366923341|        0x02        |        0x03        |       001,01       |      0000e014f     |
   |    4   |   5GMM-REGISTERED  |   001010000000103  |00101010041329122198|        0x03        |        0x04        |       001,01       |      0000e014f     |
   |    5   |   5GMM-REGISTERED  |   001010000000104  |00101010041217329206|        0x04        |        0x05        |       001,01       |      0000e014f     |
   |    6   |   5GMM-REGISTERED  |   001010000000105  |00101010041373143116|        0x02        |        0x06        |       001,01       |      0000e0ec6     |
   |    7   |   5GMM-REGISTERED  |   001010000000106  |00101010041302980548|        0x05        |        0x07        |       001,01       |      0000e014f     |
   |    8   |   5GMM-REGISTERED  |   001010000000107  |00101010041553893943|        0x06        |        0x08        |       001,01       |      0000e014f     |
   |    9   |   5GMM-REGISTERED  |   001010000000108  |00101010041148878210|        0x07        |        0x09        |       001,01       |      0000e014f     |
   |   10   |   5GMM-REGISTERED  |   001010000000109  |00101010041379311732|        0x01        |        0x0A        |       001,01       |      0000e014e     |
   |-----------------------------------------------------------------------------------------------------------------------------------------------------------|

```
#### Step 7: Run ISAC xApp Controller
Run the xApp controller to perform real-time E2 control using the trained policy:
```bash
python3 ~/openair2/E2AP/flexric/examples/xApp/python3/xapp_isac.py
```
##### Closed-Loop D3QN Agent Convergence & E2SM-RC Control Telemetry
The below demonstrates real-time E2 control exchange (`E2SM-RC` NVS capacity slice allocation) and the D3QN agent's score convergence from Episode 10 (`score: 1613.3`) to Episode 200 (`score: 5513.5`):
```bash
[xApp]: CONTROL-REQUEST tx 
[xApp]: CONTROL ACK rx
[xApp]: Successfully received CONTROL response
# of episode :10, avg score : 1613.3, optimization step: 100
ADD NVS DL SLICE: id 1, label eMBB, conf 1(capacity), pct_reserved 0.596154
ADD NVS DL SLICE: id 2, label URLLC, conf 1(capacity), pct_reserved 0.4

# of episode :20, avg score : 1562.2, optimization step: 200
# of episode :30, avg score : 1920.9, optimization step: 300
# of episode :40, avg score : 2151.6, optimization step: 400
# of episode :50, avg score : 2228.6, optimization step: 500
# of episode :60, avg score : 2596.4, optimization step: 600
# of episode :70, avg score : 2673.2, optimization step: 700

# of episode :200, avg score : 5513.5, optimization step: 2000
```
#### Step 8: Plot & Verify Testbed Results
Generate throughput, latency, and sensing accuracy plots from metrics.csv:
```bash
python3 plot_metrics.py
```
### 3. Results and Conclusion

#### Results
1. **eMBB Throughput (+31.0% Gain)**:
   - DJCSR dynamically adapts subcarrier allocation ratios ($\rho$) and transmit power budgets ($p$), yielding a **31% throughput increase** over conventional DQN while consistently exceeding the **60 Mbps QoS target**.
2. **URLLC Latency (-16.7% Reduction)**:
   - Dynamic subcarrier allocation ratios ($\rho$) and joint power splitting ($p$) optimize resource assignment to maintain median URLLC latency at **3.93 ms** (a **16.7% reduction** over conventional DQN), strictly enforcing the **< 5 ms ultra-low latency constraint** under dynamic traffic bursts.
3. **Sensing Accuracy**:
   - By joint optimization of sensing subcarrier interleaved spacing ($\Delta_s$), DJCSR achieves an average range estimation RMSE of **0.184 m**.

#### Conclusion
- **Adaptive Joint Resource Allocation**: DJCSR successfully resolves the fundamental trade-off between communication Quality-of-Service (QoS) and radar sensing precision in multi-cell 6G ISAC networks.
- **Hardware Feasibility**: Real-time E2 control loop validation on the FlexRIC Near-RT RIC and OpenAirInterface (OAI) gNB testbed confirms that D3QN-driven xApps operate seamlessly within sub-10ms Near-RT RIC decision epochs.
- **Scalability for 6G O-RAN**: The framework demonstrates that AI-native xApps can dynamically adjust subcarrier spacing, power splitting, and slice priorities in real time under dynamic fading and mobile traffic.
