# SPDX-License-Identifier: LicenseRef-CSSL-1.0
# ISAC RAN Slicing xApp — D3QN Controller
import torch
import torch.nn as nn
import numpy as np
import os

import sys
from pathlib import Path

# Resolve everything relative to the repository root so the xApp is machine independent.
# This file lives at <root>/openair2/E2AP/flexric/examples/xApp/python3/xapp_isac.py,
# so parents[6] is the DJCSR repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[6]
TRAIN_MODEL_DIR = PROJECT_ROOT / "Train_Model"
SWIG_DIR = Path(os.environ["FLEXRIC_SWIG_DIR"]) if "FLEXRIC_SWIG_DIR" in os.environ else None

sys.path.insert(0, str(TRAIN_MODEL_DIR))
if SWIG_DIR is not None:
    sys.path.insert(0, str(SWIG_DIR))

import xapp_sdk as ric
import time
import math
from collections import deque
from datetime import datetime
import threading
import csv

from environment import ISACEnvironment

# -----------------------------------------------------------------------------
# D3QN Model Definition
# -----------------------------------------------------------------------------
class D3QN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(D3QN, self).__init__()
        
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
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values

    def get_action(self, state_np):
        with torch.no_grad():
            state_t = torch.FloatTensor(state_np).unsqueeze(0)
            q_values = self.forward(state_t)
            action = q_values.argmax(dim=-1).item()
        return action

# -----------------------------------------------------------------------------
# Load Model and Env
# -----------------------------------------------------------------------------
isac_env = ISACEnvironment()
# NOTE (DJCSR): ISACEnvironment.__init__ does not create the subcarrier-allocation state;
# sensing_sc / embb_sc / urllc_sc are only assigned inside reset(). get_isac_state() reads
# those attributes for its fallback state vector, so without this call the xApp aborts with
# AttributeError: 'ISACEnvironment' object has no attribute 'urllc_sc' on the first step.
# reset() only seeds the allocation split here; the live E2 telemetry still drives the state.
isac_env.reset()

# -----------------------------------------------------------------------------
# Simulated-grid -> real-grid subcarrier/PRB conversion
# -----------------------------------------------------------------------------
# isac_env.N_sc (780) is the simulator's own subcarrier grid, used only so the D3QN
# policy has a per-subcarrier reward signal (sensing FIM/CRLB, per-SC SINR, etc).
# It does NOT match the real gNB's resource grid: DJCSR's own ISAC configuration
# (oai-workshops/ran/conf/isac_setup/gnb1.conf) sets dl_carrierBandwidth = 106 PRBs.
# The E2 SLICE service model (flexric/src/sm/slice_sm/ie/slice_data_ie.h,
# nvs_capacity_u) can only carry one float, `pct_reserved` — a fraction of the RAN's
# real PRB pool, not a subcarrier list or count. So the D3QN's subcarrier-count
# decision is first turned into a fraction (grid-independent), and that fraction is
# then re-expanded here against the REAL grid to get the actual subcarrier/PRB counts
# that a correctly-implemented ran_func_slice.c would need to reserve. This is for
# visibility/logging (see Vijay_docs/05_Subcarrier_vs_PRB_Granularity.md) — the E2
# message itself still only carries the fraction, because that field is all the wire
# format has room for.
SC_PER_PRB = 12
REAL_N_RB = 106                      # from gnb1.conf: dl_carrierBandwidth = 106
REAL_N_SC = REAL_N_RB * SC_PER_PRB   # 1272 real subcarriers


def sim_ratio_to_real_grid(ratio):
    """Convert a [0,1] slice fraction (as sent in pct_reserved) into the equivalent
    subcarrier count and PRB count on the REAL 106-PRB grid, not the 780-subcarrier
    simulator grid the fraction was derived from."""
    real_sc = int(round(ratio * REAL_N_SC))
    real_prb = int(round(ratio * REAL_N_RB))
    return real_sc, real_prb


# Load the Trained Model
ckpt_path = PROJECT_ROOT / "OAI_FlexRIC" / "d3qn_isac_final.pth"

STATE_DIM = isac_env.state_dim
ACTION_DIM = isac_env.action_dim

if not ckpt_path.is_file():
    raise FileNotFoundError(f"Required D3QN checkpoint not found: {ckpt_path}")

checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
sd = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
if not isinstance(sd, dict) or 'advantage_stream.2.weight' not in sd:
    raise ValueError(f"Invalid D3QN checkpoint format: {ckpt_path}")
ACTION_DIM = sd['advantage_stream.2.weight'].shape[0]
STATE_DIM = sd['feature.0.weight'].shape[1]
model = D3QN(STATE_DIM, ACTION_DIM)
model.load_state_dict(sd, strict=True)
model.eval()
print(f"Successfully loaded D3QN model (State Dim: {STATE_DIM}, Action Dim: {ACTION_DIM}) from {ckpt_path}")

# -----------------------------------------------------------------------------
# MAC/RLC Metrics Tracking (using exact layout from your previous code)
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
        """Construct the 16-dim state vector for the ISAC environment matching d3qn_isac_final.pth."""
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
# E2 SM Callbacks
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
    # Retain Python references so SWIG C++ memory isn't garbage collected prematurely
    msg._keepalive = (dl, slices, created_slices)
    return msg

def save_testbed_results_plot():
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        csv_path = PROJECT_ROOT / "Metrics" / "live_metrics.csv"
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            print("[WARNING] metrics.csv is empty, skipping plot generation.")
            return

        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return

        # Journal Style Setup
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

        # 1. Throughput Plot (Raw Spikes + Mean + Std Dev)
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

        # 2. Latency Plot (Raw Spikes + Mean + Std Dev)
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

        # 3. Sensing RMSE Plot (Raw Spikes + Mean + Std Dev)
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
        fig_dir = PROJECT_ROOT / "fig"
        fig_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = str(fig_dir / 'Testbed_Results.pdf')
        png_path = str(fig_dir / 'Testbed_Results.png')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.savefig(png_path, format='png', dpi=1000, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"[WARNING] Failed to auto-generate plot: {e}")

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import signal
    ric.init()
    signal.signal(signal.SIGINT, signal.default_int_handler)

    conn = ric.conn_e2_nodes()
    assert(len(conn) > 0)
    print(f"Connected to {len(conn)} E2 Nodes")

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

    try:
        time.sleep(2)
        print("Starting ISAC RL Loop...")
        
        # Setup CSV Logging (Append/Resume Mode)
        csv_path = PROJECT_ROOT / "Metrics" / "live_metrics.csv"
        CSV_HEADER = ['iteration', 'embb_throughput', 'urllc_throughput', 'urllc_latency',
                      'prb_embb', 'prb_urllc',
                      'sim_embb_sc', 'sim_urllc_sc', 'sim_total_sc',
                      'real_embb_sc', 'real_urllc_sc', 'real_embb_prb', 'real_urllc_prb', 'real_total_prb',
                      'rmse', 'crlb']
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        start_episode = 1

        if file_exists:
            with open(csv_path, 'r') as f:
                lines = [l for l in f.readlines() if l.strip()]
            # NOTE (DJCSR): the sc/PRB columns above are a schema change. A CSV written
            # by the pre-conversion xApp has a different (shorter) header, so appending
            # to it would misalign every column. Detect that and archive it instead of
            # silently corrupting the file.
            header_matches = bool(lines) and lines[0].strip() == ",".join(CSV_HEADER)
            if not header_matches:
                archive_dir = PROJECT_ROOT / "Metrics" / "archive"
                archive_dir.mkdir(parents=True, exist_ok=True)
                archived_to = archive_dir / f"live_metrics_pre_sc_conversion_{int(time.time())}.csv"
                os.replace(csv_path, archived_to)
                print(f"[INFO] Existing CSV used an older schema; archived to {archived_to} and starting fresh.")
                lines = []
            if len(lines) > 1:
                start_episode = len(lines) # Resume from next episode

        if start_episode > 1:
            csv_file = open(csv_path, 'a', newline='')
            csv_writer = csv.writer(csv_file)
            print(f'[INFO] Resuming CSV logging from episode {start_episode}...')
        else:
            csv_file = open(csv_path, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(CSV_HEADER)
            print('[INFO] CSV logging initialized – fresh file created.')
        
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
            ep_sim_sc_e = []
            ep_sim_sc_u = []
            ep_real_sc_e = []
            ep_real_sc_u = []
            ep_real_prb_e = []
            ep_real_prb_u = []

            for step in range(EPISODE_LENGTH):
                # 1. Get State (Live from E2 Telemetry)
                state = state_tracker.get_isac_state()
                
                # 2. Forward Pass (Action Selection)
                action_idx = model.get_action(state)
                
                # 3. Decode the action without advancing the offline simulator.
                resource_idx = action_idx // len(isac_env.power_ratios)
                power_idx = action_idx % len(isac_env.power_ratios)
                _, _, step_size, _ = isac_env.action_table[resource_idx]
                _, embb_direction, _, _ = isac_env.action_table[resource_idx]
                current_embb = state_tracker.get_isac_state()[8] * isac_env.N_sc
                current_urllc = state_tracker.get_isac_state()[7] * isac_env.N_sc
                embb_sc = np.clip(current_embb + embb_direction * step_size, isac_env.num_embb, isac_env.N_sc)
                urllc_sc = np.clip(current_urllc - embb_direction * step_size, isac_env.num_urllc, isac_env.N_sc)
                total_sc = max(embb_sc + urllc_sc, 1.0)
                prb_embb = float(embb_sc / total_sc)
                prb_urllc = float(urllc_sc / total_sc)
                pwr_sense, pwr_urllc, pwr_embb = isac_env.power_ratios[power_idx]

                # 3b. Requested subcarrier count, on the SIMULATOR's own grid (780 SC) --
                # this is the count the D3QN policy actually reasoned about.
                sim_embb_sc = int(round(embb_sc))
                sim_urllc_sc = int(round(urllc_sc))

                # 3c. The same decision re-expanded on the REAL RAN grid (106 PRB /
                # 1272 SC, see sim_ratio_to_real_grid above) -- this is the count that
                # would actually need to be reserved for the fraction sent over E2 to
                # mean what the policy intended.
                real_embb_sc, real_embb_prb = sim_ratio_to_real_grid(prb_embb)
                real_urllc_sc, real_urllc_prb = sim_ratio_to_real_grid(prb_urllc)

                # 4. Extract live E2 metrics for logging.
                total_sc = float(isac_env.N_sc)
                metrics = state_tracker.metrics
                ep_embb_tp.append(metrics["embb"]["throughput"])
                ep_urllc_tp.append(metrics["urllc"]["throughput"])
                ep_urllc_lat.append(metrics["urllc"]["latency"])
                ep_prb_e.append(prb_embb)
                ep_prb_u.append(prb_urllc)
                ep_sim_sc_e.append(sim_embb_sc)
                ep_sim_sc_u.append(sim_urllc_sc)
                ep_real_sc_e.append(real_embb_sc)
                ep_real_sc_u.append(real_urllc_sc)
                ep_real_prb_e.append(real_embb_prb)
                ep_real_prb_u.append(real_urllc_prb)
                ep_rmse.append(float("nan"))
                ep_crlb.append(float("nan"))

                # 5. Send Control to E2 Node
                slice_config = {
                    "num_slices": 2,
                    "slice_sched_algo": "NVS",
                    "slices": [
                        {
                            "id": 1, "label": "eMBB", "ue_sched_algo": "PF",
                            "type": "SLICE_SM_NVS_V0_CAPACITY",
                            "slice_algo_params": {"pct_rsvd": prb_embb},
                            "power_fraction": pwr_embb
                        },
                        {
                            "id": 2, "label": "URLLC", "ue_sched_algo": "PF",
                            "type": "SLICE_SM_NVS_V0_CAPACITY",
                            "slice_algo_params": {"pct_rsvd": prb_urllc},
                            "power_fraction": pwr_urllc
                        }
                    ]
                }
                msg = fill_slice_ctrl_msg(slice_config)
                for i in range(len(conn)):
                    ric.control_slice_sm(conn[i].id, msg)
                    
                time.sleep(1.0) # Poll/control interval (optimized for 100+ episode stability)
                
            # Compute Averages
            avg_embb = np.mean(ep_embb_tp)
            avg_urllc = np.mean(ep_urllc_tp) if ep_urllc_tp else 0.0
            avg_lat = np.mean(ep_urllc_lat)
            avg_rmse = np.mean(ep_rmse)
            avg_crlb = np.mean(ep_crlb) if ep_crlb else 0.0
            avg_prb_e = np.mean(ep_prb_e)
            avg_prb_u = np.mean(ep_prb_u)
            avg_sim_sc_e = np.mean(ep_sim_sc_e)
            avg_sim_sc_u = np.mean(ep_sim_sc_u)
            avg_real_sc_e = np.mean(ep_real_sc_e)
            avg_real_sc_u = np.mean(ep_real_sc_u)
            avg_real_prb_e = np.mean(ep_real_prb_e)
            avg_real_prb_u = np.mean(ep_real_prb_u)

            # Log to CSV for plotting
            csv_writer.writerow([
                episode, avg_embb, avg_urllc, avg_lat, avg_prb_e, avg_prb_u,
                avg_sim_sc_e, avg_sim_sc_u, isac_env.N_sc,
                avg_real_sc_e, avg_real_sc_u, avg_real_prb_e, avg_real_prb_u, REAL_N_RB,
                avg_rmse, avg_crlb,
            ])
            csv_file.flush()

            print(f"Control Epoch {episode}/{MAX_EPISODES} (Live E2 Avg) | eMBB: {avg_embb:.2f} Mbps | URLLC: {avg_urllc:.2f} Mbps | "
                  f"Latency: {avg_lat:.2f} ms | RMSE: {avg_rmse:.4f} m | CRLB: {avg_crlb:.2f} m | "
                  f"SC(sim {isac_env.N_sc}): eMBB {avg_sim_sc_e:.0f}/URLLC {avg_sim_sc_u:.0f} | "
                  f"SC(real {REAL_N_SC}): eMBB {avg_real_sc_e:.0f}/URLLC {avg_real_sc_u:.0f} | "
                  f"PRB(real {REAL_N_RB}): eMBB {avg_real_prb_e:.0f}/URLLC {avg_real_prb_u:.0f}")
            episode += 1

        print(f"\n[INFO] Successfully completed target of {MAX_EPISODES} control epochs! Gracefully stopping xApp...")
            
    except KeyboardInterrupt:
        print("\nStopping...")

    for h in mac_hndlrs: ric.rm_report_mac_sm(h)
    for h in rlc_hndlrs: ric.rm_report_rlc_sm(h)
    while ric.try_stop() == 0: time.sleep(1)
    
    # Auto-generate JSAC Testbed_Results.pdf
    save_testbed_results_plot()
    print('ISAC RL xApp stopped')
