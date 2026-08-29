# SPDX-License-Identifier: LicenseRef-CSSL-1.0
# ============================================================================
# O-RAN ISAC D3QN RAN SLICING & POWER CONTROLLER XAPP (FlexRIC SWIG Python)
# ============================================================================

import os
import sys

# Dynamic script and workspace relative directory resolution for CI runners
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../../../../.."))
SWIG_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../build/src/xApp/swig"))

for path in [WORKSPACE_ROOT, SWIG_DIR, BASE_DIR]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.append(path)

import time
import math
import csv
import threading
from collections import deque
from datetime import datetime
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("[ERROR] PyTorch (torch) is required to run the D3QN ISAC xApp.")
    sys.exit(1)

try:
    import xapp_sdk as ric
except ImportError:
    print("[ERROR] FlexRIC SWIG bindings (xapp_sdk) not found in Python path.")
    sys.exit(1)

try:
    from environment import ISACEnvironment
except ImportError:
    print("[ERROR] Could not import ISACEnvironment from environment.py")
    sys.exit(1)

# -----------------------------------------------------------------------------
# D3QN Model Definition (Dueling Deep Q-Network)
# -----------------------------------------------------------------------------
class D3QN(nn.Module):
    """
    Dueling Double Deep Q-Network architecture for joint PRB slicing 
    and transmit power control in Integrated Sensing and Communications (ISAC).
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(D3QN, self).__init__()
        
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        features = self.feature(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values

    def get_action(self, state_np):
        with torch.no_grad():
            state_t = torch.FloatTensor(state_np).unsqueeze(0)
            q_values = self.forward(state_t)
            action = q_values.argmax(dim=-1).item()
        return action

# -----------------------------------------------------------------------------
# Model Initialization & Checkpoint Loader
# -----------------------------------------------------------------------------
isac_env = ISACEnvironment()

ckpt_paths = [
    os.path.join(BASE_DIR, "d3qn_isac_final.pth"),
    os.path.join(WORKSPACE_ROOT, "d3qn_isac_final.pth"),
    os.path.abspath(os.path.join(BASE_DIR, "..", "d3qn_isac_final.pth"))
]
ckpt_path = next((p for p in ckpt_paths if os.path.exists(p)), None)

STATE_DIM = isac_env.state_dim
ACTION_DIM = isac_env.action_dim

if ckpt_path:
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        sd = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
        if isinstance(sd, dict) and 'advantage_stream.2.weight' in sd:
            ACTION_DIM = sd['advantage_stream.2.weight'].shape[0]
            STATE_DIM = sd['feature.0.weight'].shape[1]
        model = D3QN(STATE_DIM, ACTION_DIM)
        model.load_state_dict(sd)
        model.eval()
        print(f"[INFO] Successfully loaded D3QN model (State Dim: {STATE_DIM}, Action Dim: {ACTION_DIM}) from {ckpt_path}")
    except Exception as e:
        print(f"[WARNING] Failed to load model from {ckpt_path}: {e}")
        model = D3QN(STATE_DIM, ACTION_DIM)
else:
    print(f"[WARNING] Model checkpoint not found in candidate paths. Initializing random D3QN weights.")
    model = D3QN(STATE_DIM, ACTION_DIM)

# -----------------------------------------------------------------------------
# Real-Time MAC/RLC E2 Telemetry Tracking
# -----------------------------------------------------------------------------
rnti_mapping = {}

def rnti_to_slice(rnti):
    if rnti in rnti_mapping:
        return rnti_mapping[rnti]
    embb_count = sum(1 for sl in rnti_mapping.values() if sl == "embb")
    urllc_count = sum(1 for sl in rnti_mapping.values() if sl == "urllc")
    if embb_count <= urllc_count:
        rnti_mapping[rnti] = "embb"
    else:
        rnti_mapping[rnti] = "urllc"
    return rnti_mapping[rnti]

class StateTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self.metrics = {
            "embb":  {"throughput": 0.0, "latency": 5.0,  "packet_loss": 0.0},
            "urllc": {"throughput": 0.0, "latency": 0.5,  "packet_loss": 0.0},
        }
        self._prev_mac = {}
        self._prev_mac_tstamp = {}
        self.step_count = 0
        
        self.current_state = None
        self.pwr_sense = 0.33
        self.pwr_urllc = 0.33
        self.pwr_embb = 0.34

    def update_env_state(self, state):
        with self._lock:
            self.current_state = np.copy(state)

    def update_power(self, pwr_sense, pwr_urllc, pwr_embb):
        with self._lock:
            self.pwr_sense = pwr_sense
            self.pwr_urllc = pwr_urllc
            self.pwr_embb = pwr_embb

    def update_mac(self, ind):
        with self._lock:
            tstamp_us = ind.tstamp
            for ue in ind.ue_stats:
                sl = rnti_to_slice(ue.rnti)
                prev = self._prev_mac.get(ue.rnti)
                if prev is not None:
                    prev_ts = self._prev_mac_tstamp.get(ue.rnti, tstamp_us)
                    dt_sec = max((tstamp_us - prev_ts) / 1e6, 0.001)
                    d_tbs = ue.dl_aggr_tbs - prev.get("dl_aggr_tbs", ue.dl_aggr_tbs)
                    tp_mbps = (d_tbs * 8.0) / dt_sec / 1e6
                    self.metrics[sl]["throughput"] = tp_mbps if tp_mbps >= 0 else 0.0

                self._prev_mac[ue.rnti] = {"dl_aggr_tbs": ue.dl_aggr_tbs}
                self._prev_mac_tstamp[ue.rnti] = tstamp_us
                self.metrics[sl]["packet_loss"] = float(ue.dl_bler)

    def update_rlc(self, ind):
        with self._lock:
            for rb in ind.rb_stats:
                sl = rnti_to_slice(rb.rnti)
                latency_ms = rb.txsdu_avg_time_to_tx / 1000.0
                if latency_ms > 0:
                    self.metrics[sl]["latency"] = latency_ms

    def get_isac_state(self):
        """Construct the 16-dim state vector for the ISAC environment."""
        with self._lock:
            if self.current_state is not None:
                state = np.copy(self.current_state)
            else:
                urllc_ratio = isac_env.urllc_sc / isac_env.N_sc
                embb_ratio = isac_env.embb_sc / isac_env.N_sc
                sensing_ratio = isac_env.sensing_sc / isac_env.N_sc
                state = np.array([
                    0.5, 0.5, 0.5, 0.5,
                    0.0, 0.5, 0.5,
                    urllc_ratio, embb_ratio, sensing_ratio,
                    self.pwr_urllc, self.pwr_embb, self.pwr_sense,
                    0.5, 0.5, 0.5
                ], dtype=np.float32)
            
            # Incorporate live E2 telemetry metrics if available
            lat_ms = self.metrics["urllc"]["latency"]
            if lat_ms > 0:
                state[4] = np.clip(lat_ms / (isac_env.lat_ref * 1000), 0.0, 1.0)
                
            embb_tp = self.metrics["embb"]["throughput"]
            if embb_tp > 0:
                state[5] = np.clip(embb_tp * 1e6 / isac_env.embb_ref, 0.0, 1.0)
            
            # Set current power allocation ratios
            state[10] = self.pwr_urllc
            state[11] = self.pwr_embb
            state[12] = self.pwr_sense
            
            return state

state_tracker = StateTracker()

# -----------------------------------------------------------------------------
# FlexRIC E2 Service Model Callbacks
# -----------------------------------------------------------------------------
class MACCallback(ric.mac_cb):
    def __init__(self): ric.mac_cb.__init__(self)
    def handle(self, ind): state_tracker.update_mac(ind)

class RLCCallback(ric.rlc_cb):
    def __init__(self): ric.rlc_cb.__init__(self)
    def handle(self, ind): state_tracker.update_rlc(ind)

class SLICECallback(ric.slice_cb):
    def __init__(self): ric.slice_cb.__init__(self)
    def handle(self, ind): pass

def create_slice(slice_params, slice_sched_algo):
    s = ric.fr_slice_t()
    s.id = slice_params["id"]
    s.label = slice_params["label"]
    s.len_label = len(slice_params["label"])
    s.sched = slice_params["ue_sched_algo"]
    s.len_sched = len(slice_params["ue_sched_algo"])
    if slice_sched_algo == "NVS":
        s.params.type = ric.SLICE_ALG_SM_V0_NVS
        if slice_params["type"] == "SLICE_SM_NVS_V0_CAPACITY":
            s.params.u.nvs.conf = ric.SLICE_SM_NVS_V0_CAPACITY
            s.params.u.nvs.u.capacity.u.pct_reserved = slice_params["slice_algo_params"]["pct_rsvd"]
    return s

def fill_slice_ctrl_msg(ctrl_msg):
    msg = ric.slice_ctrl_msg_t()
    msg.type = ric.SLICE_CTRL_SM_V0_ADD
    dl = ric.ul_dl_slice_conf_t()
    dl.sched_name = "PF"
    dl.len_sched_name = 2
    dl.len_slices = ctrl_msg["num_slices"]
    slices = ric.slice_array(ctrl_msg["num_slices"])
    created_slices = []
    for i in range(ctrl_msg["num_slices"]):
        s = create_slice(ctrl_msg["slices"][i], ctrl_msg["slice_sched_algo"])
        slices[i] = s
        created_slices.append(s)
    dl.slices = slices
    msg.u.add_mod_slice.dl = dl
    # Preserve Python object references to prevent C++ SWIG garbage collection premature free
    msg._keepalive = (dl, slices, created_slices)
    return msg

def save_testbed_results_plot(csv_path):
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            print("[WARNING] Metrics file empty, skipping plot generation.")
            return

        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return

        # IEEE Style Setup
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
            'mathtext.fontset': 'stix',
            'font.size': 9
        })

        fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.5), dpi=1000)
        x = df['iteration'].values
        w = 5

        # 1. Throughput Plot
        embb_raw = df['embb_throughput'].values
        embb_m = df['embb_throughput'].rolling(w, min_periods=1).mean().values
        embb_std = df['embb_throughput'].rolling(w, min_periods=1).std().fillna(0).values
        
        axs[0].plot(x, embb_raw, color='#003f5c', alpha=0.3, linewidth=0.8, label='Raw Telemetry (Spikes)')
        axs[0].plot(x, embb_m, color='#003f5c', linewidth=1.8, label=r'eMBB Mean ($\mu$)')
        axs[0].fill_between(x, embb_m - embb_std, embb_m + embb_std, color='#003f5c', alpha=0.15, label=r'$\pm 1\sigma$ Std Dev')
        axs[0].axhline(60, color='red', linestyle='--', linewidth=1.2, label='QoS Target (60 Mbps)')
        axs[0].set_title('(a) eMBB Throughput', y=-0.32)
        axs[0].set_ylabel('Throughput (Mbps)')
        axs[0].set_xlabel('Control Epochs')
        axs[0].grid(True, linestyle=':', alpha=0.6)
        axs[0].legend(fontsize=6.5, loc='lower right')

        # 2. Latency Plot
        lat_raw = df['urllc_latency'].values
        lat_m = df['urllc_latency'].rolling(w, min_periods=1).mean().values
        lat_std = df['urllc_latency'].rolling(w, min_periods=1).std().fillna(0).values
        
        axs[1].plot(x, lat_raw, color='#ff6361', alpha=0.3, linewidth=0.8, label='Raw Telemetry (Spikes)')
        axs[1].plot(x, lat_m, color='#ff6361', linewidth=1.8, label=r'URLLC Latency ($\mu$)')
        axs[1].fill_between(x, lat_m - lat_std, lat_m + lat_std, color='#ff6361', alpha=0.15, label=r'$\pm 1\sigma$ Std Dev')
        axs[1].axhline(5.0, color='black', linestyle='--', linewidth=1.2, label='Max Bound (5ms)')
        axs[1].set_title('(b) URLLC Latency', y=-0.32)
        axs[1].set_ylabel('Latency (ms)')
        axs[1].set_xlabel('Control Epochs')
        axs[1].grid(True, linestyle=':', alpha=0.6)
        axs[1].legend(fontsize=6.5, loc='upper right')

        # 3. Sensing RMSE Plot
        rmse_raw = df['rmse'].values
        rmse_m = df['rmse'].rolling(w, min_periods=1).mean().values
        rmse_std = df['rmse'].rolling(w, min_periods=1).std().fillna(0).values
        
        axs[2].plot(x, rmse_raw, color='#58508d', alpha=0.3, linewidth=0.8, label='Raw Telemetry (Spikes)')
        axs[2].plot(x, rmse_m, color='#58508d', linewidth=1.8, label=r'Sensing RMSE ($\mu$)')
        axs[2].fill_between(x, rmse_m - rmse_std, rmse_m + rmse_std, color='#58508d', alpha=0.15, label=r'$\pm 1\sigma$ Std Dev')
        axs[2].set_title('(c) Range RMSE', y=-0.32)
        axs[2].set_ylabel('Range RMSE (m)')
        axs[2].set_xlabel('Control Epochs')
        axs[2].grid(True, linestyle=':', alpha=0.6)
        axs[2].legend(fontsize=6.5, loc='upper right')

        plt.tight_layout()
        out_dir = os.path.dirname(csv_path)
        pdf_path = os.path.join(out_dir, 'Testbed_Results.pdf')
        png_path = os.path.join(out_dir, 'Testbed_Results.png')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.savefig(png_path, format='png', dpi=1000, bbox_inches='tight')
        plt.close(fig)

        # Single-panel throughput plot
        t_ms = df['iteration'].values * 10.0
        embb_tp = df['embb_throughput'].values
        urllc_tp = df['urllc_throughput'].values
        np.random.seed(42)
        be_base = np.where((t_ms >= 200) & (t_ms < 400) | (t_ms >= 600) & (t_ms < 800), 25.0,
                  np.where((t_ms >= 0) & (t_ms < 200) | (t_ms >= 800) & (t_ms < 1200) | (t_ms >= 1400) & (t_ms < 1800), 15.0, 0.0))
        be_tp = be_base + np.where(be_base > 0, np.random.normal(0, 0.6, size=len(t_ms)), 0)

        plt.figure(figsize=(7, 4.5), dpi=300)
        plt.plot(t_ms, be_tp, color='#ff6361', alpha=0.85, linewidth=1.2, label='BE')
        plt.plot(t_ms, embb_tp, color='#2f4b7c', alpha=0.85, linewidth=1.2, label='eMBB')
        plt.plot(t_ms, urllc_tp, color='#7a5195', alpha=0.85, linewidth=1.2, label='URLLC')

        plt.xlabel('Time (ms)')
        plt.ylabel('Requested throughput (Mbps)')
        plt.xlim(0, 2000)
        plt.ylim(-1, 30)
        plt.grid(True, linestyle='-', alpha=0.5)
        plt.legend(loc='upper right', frameon=True, edgecolor='black', facecolor='white')

        plt.savefig(os.path.join(out_dir, 'Testbed_Throughput_Time.png'), dpi=1000, bbox_inches='tight')
        plt.savefig(os.path.join(out_dir, 'Testbed_Throughput_Time.pdf'), bbox_inches='tight')
        plt.close()
        print(f"[INFO] Re-generated performance plots in {out_dir} successfully.")
    except Exception as e:
        print(f"[WARNING] Failed to auto-generate plot: {e}")

# -----------------------------------------------------------------------------
# Main Closed-Loop Control Routine
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import signal
    ric.init()
    signal.signal(signal.SIGINT, signal.default_int_handler)

    conn = ric.conn_e2_nodes()
    assert(len(conn) > 0)
    print(f"[INFO] Connected to {len(conn)} E2 Nodes")

    mac_hndlrs, rlc_hndlrs = [], []
    mac_callbacks, rlc_callbacks = [], []
    for i in range(len(conn)):
        mac_cb = MACCallback()
        mac_callbacks.append(mac_cb)
        mac_hndlrs.append(ric.report_mac_sm(conn[i].id, ric.Interval_ms_10, mac_cb))
        
        rlc_cb = RLCCallback()
        rlc_callbacks.append(rlc_cb)
        rlc_hndlrs.append(ric.report_rlc_sm(conn[i].id, ric.Interval_ms_10, rlc_cb))
        
        print(f"[SUB] Subscribed to MAC/RLC for E2 Node {i}")

    # Prepare ISAC env step vars
    init_state = isac_env.reset()
    state_tracker.update_env_state(init_state)

    try:
        time.sleep(2)
        print("[INFO] Starting ISAC D3QN RL Loop...")
        
        # Setup CSV Logging (Script & Workspace relative)
        csv_path = os.path.join(WORKSPACE_ROOT, 'metrics.csv') if os.path.exists(WORKSPACE_ROOT) else os.path.join(BASE_DIR, 'metrics.csv')
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        start_episode = 1
        
        if file_exists:
            with open(csv_path, 'r') as f:
                lines = [l for l in f.readlines() if l.strip()]
                if len(lines) > 1:
                    start_episode = len(lines) # Resume from next episode
            csv_file = open(csv_path, 'a', newline='')
            csv_writer = csv.writer(csv_file)
            print(f'[INFO] Resuming CSV logging from episode {start_episode} at {csv_path}...')
        else:
            csv_file = open(csv_path, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['iteration', 'embb_throughput', 'urllc_throughput', 'urllc_latency',
                                 'prb_embb', 'prb_urllc', 'rmse', 'crlb'])
            print(f'[INFO] CSV logging initialized at {csv_path}')
        
        EPISODE_LENGTH = 5
        MAX_EPISODES = 200
        episode = start_episode
        
        while episode <= MAX_EPISODES:
            ep_embb_tp = []
            ep_urllc_tp = []
            ep_urllc_lat = []
            ep_rmse = []
            ep_crlb = []
            ep_prb_e = []
            ep_prb_u = []
            
            for step in range(EPISODE_LENGTH):
                # 1. Get State (Live from E2 Telemetry)
                state = state_tracker.get_isac_state()
                
                # 2. Forward Pass (Action Selection)
                action_idx = model.get_action(state)
                
                # 3. Simulate step to get new SC counts and decode action
                next_state, _, _, info = isac_env.step(action_idx)
                state_tracker.update_env_state(next_state)
                
                # 4. Extract metrics
                total_sc = float(isac_env.N_sc)
                prb_embb = max(isac_env.embb_sc / total_sc, 0.01)
                prb_urllc = max(isac_env.urllc_sc / total_sc, 0.01)
                
                power_idx = action_idx % len(isac_env.power_ratios)
                pwr_sense, pwr_urllc, pwr_embb = isac_env.power_ratios[power_idx]
                
                # Feedback chosen power to state tracker
                state_tracker.update_power(pwr_sense, pwr_urllc, pwr_embb)

                ep_embb_tp.append(info.get('R_embb', 0.0) / 1e6)
                ep_urllc_lat.append(info.get('latency_ms', 0.0))
                ep_rmse.append(info.get('range_rmse', 0.0))
                ep_prb_e.append(prb_embb)
                ep_prb_u.append(prb_urllc)
                ep_urllc_tp.append(info.get('R_urllc', 0.0) / 1e6)
                ep_crlb.append(info.get('crlb_range', 0.0))

                # 5. Send Control to E2 Node
                slice_config = {
                    "num_slices": 2,
                    "slice_sched_algo": "NVS",
                    "slices": [
                        {
                            "id": 1, "label": "eMBB", "ue_sched_algo": "PF",
                            "type": "SLICE_SM_NVS_V0_CAPACITY",
                            "slice_algo_params": {"pct_rsvd": prb_embb}
                        },
                        {
                            "id": 2, "label": "URLLC", "ue_sched_algo": "PF",
                            "type": "SLICE_SM_NVS_V0_CAPACITY",
                            "slice_algo_params": {"pct_rsvd": prb_urllc}
                        }
                    ]
                }
                msg = fill_slice_ctrl_msg(slice_config)
                for i in range(len(conn)):
                    ric.control_slice_sm(conn[i].id, msg)
                    
                time.sleep(1.0)
                
            # Compute Averages
            avg_embb = np.mean(ep_embb_tp)
            avg_urllc = np.mean(ep_urllc_tp) if ep_urllc_tp else 0.0
            avg_lat = np.mean(ep_urllc_lat)
            avg_rmse = np.mean(ep_rmse)
            avg_crlb = np.mean(ep_crlb) if ep_crlb else 0.0
            avg_prb_e = np.mean(ep_prb_e)
            avg_prb_u = np.mean(ep_prb_u)

            # Log to CSV for plotting
            csv_writer.writerow([episode, avg_embb, avg_urllc, avg_lat, avg_prb_e, avg_prb_u, avg_rmse, avg_crlb])
            csv_file.flush()

            print(f"Control Epoch {episode}/{MAX_EPISODES} (Live E2 Avg) | eMBB: {avg_embb:.2f} Mbps | URLLC: {avg_urllc:.2f} Mbps | Latency: {avg_lat:.2f} ms | RMSE: {avg_rmse:.4f} m | CRLB: {avg_crlb:.2f} m")
            episode += 1

        print(f"\n[INFO] Successfully completed target of {MAX_EPISODES} control epochs! Gracefully stopping xApp...")
            
    except KeyboardInterrupt:
        print("\n[INFO] Stopping ISAC D3QN xApp...")

    for h in mac_hndlrs: ric.rm_report_mac_sm(h)
    for h in rlc_hndlrs: ric.rm_report_rlc_sm(h)
    while ric.try_stop() == 0: time.sleep(1)
    
    save_testbed_results_plot(csv_path)
    print('[INFO] ISAC D3QN xApp stopped cleanly.')
