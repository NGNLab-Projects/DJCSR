import sys
import os
import subprocess
import time
import numpy as np

# Import ISACEnvironment from OAI directory
sys.path.append("/home/ngnlab/openairinterface5g")
try:
    from environment import ISACEnvironment
except ImportError:
    print("Error: Could not import environment.py")
    sys.exit(1)

def main():
    print("Initializing ISAC Environment to compute UE-gNB associations...")
    env = ISACEnvironment(num_bs=3, num_urllc=5, num_embb=5)
    env.reset()
    
    associations = env.association
    print(f"Computed Associations (UE index -> gNB index): {associations}")
    
    # 1. Launch 3 gNBs
    gnb_processes = []
    print("Starting 3 gNBs...")
    for i in range(1, 4):
        conf_file = f"/home/ngnlab/openairinterface5g/oai-workshops/ran/conf/isac_setup/gnb{i}.conf"
        cmd = f"cd /home/ngnlab/openairinterface5g/cmake_targets/ran_build/build && sudo ./nr-softmodem -O {conf_file} --rfsim -E --rfsimulator.[0].serverport {4040+i}"
        log_file = open(f"/home/ngnlab/openairinterface5g/gnb{i}.out", "w")
        p = subprocess.Popen(cmd, shell=True, stdout=log_file, stderr=log_file)
        gnb_processes.append(p)
        print(f"Launched gNB {i} (PID: {p.pid})")
    
    time.sleep(15) # Wait for gNBs to connect to AMF
    
    # 2. Launch 10 UEs in namespaces
    ue_processes = []
    print("Starting 10 UEs in namespaces...")
    for ue_idx in range(10):
        ue_id = ue_idx + 1
        gnb_idx = associations[ue_idx]
        gnb_port = 4041 + gnb_idx
        gateway_ip = f"10.{200+ue_id}.1.100"
        
        # Clean up namespace if it already exists
        os.system(f"sudo /home/ngnlab/openairinterface5g/tools/scripts/multi-ue.sh -d {ue_id} > /dev/null 2>&1")
        
        # Create namespace
        os.system(f"sudo /home/ngnlab/openairinterface5g/tools/scripts/multi-ue.sh -c {ue_id}")
        
        conf_file = f"/home/ngnlab/openairinterface5g/oai-workshops/ran/conf/isac_setup/ue{ue_id}.conf"
        
        # Launch UE inside namespace
        cmd = f"sudo ip netns exec ue{ue_id} /home/ngnlab/openairinterface5g/cmake_targets/ran_build/build/nr-uesoftmodem " \
              f"-O {conf_file} --rfsim -E " \
              f"--rfsimulator.serveraddr {gateway_ip} " \
              f"--rfsimulator.serverport {gnb_port} " \
              f"--numerology 1 --band 78 -C 3619200000 -r 106"
        
        log_file = open(f"/home/ngnlab/openairinterface5g/ue{ue_id}.out", "w")
        p = subprocess.Popen(cmd, shell=True, stdout=log_file, stderr=log_file)
        ue_processes.append(p)
        print(f"Launched UE {ue_id} connected to gNB {gnb_idx+1} on port {gnb_port} (PID: {p.pid})")
        
        time.sleep(2) # Stagger UE launches
        
    print("All nodes launched. Check /home/ngnlab/openairinterface5g/gnb*.out and ue*.out for details.")
    print("Press Ctrl+C to stop all nodes and clean up namespaces.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping all processes...")
        os.system("sudo killall -9 nr-softmodem nr-uesoftmodem")
        for ue_idx in range(10):
            os.system(f"sudo /home/ngnlab/openairinterface5g/tools/scripts/multi-ue.sh -d {ue_idx+1}")
        print("Cleanup complete.")

if __name__ == "__main__":
    main()
