import sys
import time
import csv
import os
import numpy as np
import torch
import torch.nn as nn

# Load the ISAC Environment
try:
    from environment import ISACEnvironment
except ImportError:
    print("Error: Could not import environment.py")
    sys.exit(1)

# Reconstruct the D3QN Architecture from the state_dict
class DuelingDQN(nn.Module):
    def __init__(self, state_dim=16, action_dim=1025):
        super(DuelingDQN, self).__init__()
        
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, state):
        features = self.feature(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

def main():
    print("Initializing ISAC RL xApp...")
    env = ISACEnvironment(num_bs=3, num_urllc=5, num_embb=5)
    state = env.reset()
    
    # Load the Trained Model
    model_paths = [
        "/home/ngnlab/d3qn_isac_final.pth",
        "/home/ngnlab/openairinterface5g/d3qn_isac_final.pth"
    ]
    model_path = next((p for p in model_paths if os.path.exists(p)), None)
    
    model = None
    if model_path:
        print(f"Loading trained model from: {model_path}")
        model = DuelingDQN(state_dim=env.state_dim, action_dim=env.action_dim)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
    else:
        print("WARNING: Trained model not found. Using random actions.")
    
    csv_file = "/home/ngnlab/openairinterface5g/metrics.csv"
    
    # Initialize CSV with headers
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "embb_throughput", "urllc_latency", "prb_embb", "prb_urllc", "rmse"])
    
    print(f"Metrics will be logged to {csv_file}")
    print("Starting closed-loop control (Averaging for Academic Plotting)...")
    
    # Define how many real-time steps make up 1 Episode
    # You can change this to match your Python offline simulation!
    EPISODE_LENGTH = 100 
    
    try:
        episode = 1
        while True:
            
            # Accumulators for the episode
            ep_embb_throughput = []
            ep_urllc_latency = []
            ep_rmse = []
            ep_prb_embb = []
            ep_prb_urllc = []
            
            # Run the steps for one full episode
            for step in range(EPISODE_LENGTH):
                # Agent chooses action
                if model:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        q_values = model(state_tensor)
                        action = q_values.argmax().item()
                else:
                    action = np.random.randint(0, env.action_dim) 
                
                # Step the environment
                next_state, reward, done, info = env.step(action)
                state = next_state
                
                # Extract metrics
                ep_embb_throughput.append(info.get("R_embb", 0) / 1e6)
                ep_urllc_latency.append(info.get("latency_ms", 0))
                ep_rmse.append(info.get("range_rmse", 0))
                
                total_sc = env.N_sc
                ep_prb_embb.append(env.embb_sc / total_sc)
                ep_prb_urllc.append(env.urllc_sc / total_sc)
                
                # Sleep to emulate live polling
                time.sleep(0.5) 
                
            # Calculate Episode Averages
            avg_embb = np.mean(ep_embb_throughput)
            avg_lat = np.mean(ep_urllc_latency)
            avg_rmse = np.mean(ep_rmse)
            avg_prb_e = np.mean(ep_prb_embb)
            avg_prb_u = np.mean(ep_prb_urllc)
            
            # Write exactly ONE row per Episode to the CSV
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode, avg_embb, avg_lat, avg_prb_e, avg_prb_u, avg_rmse])
            
            print(f"Episode {episode} (Avg over {EPISODE_LENGTH} steps) | eMBB: {avg_embb:.2f} Mbps | Latency: {avg_lat:.2f} ms | RMSE: {avg_rmse:.4f} m")
            
            episode += 1
            
    except KeyboardInterrupt:
        print("\nStopping xApp.")

if __name__ == "__main__":
    main()
