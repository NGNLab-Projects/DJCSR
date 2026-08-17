# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# train.py

import sys

# Force reload of environment and agent modules on re-run in Jupyter
for mod in list(sys.modules.keys()):
    if mod in ['environment', 'agent']:
        del sys.modules[mod]

import numpy as np
import matplotlib.pyplot as plt
from environment import ISACEnvironment
from agent import D3QNAgent

# ======================================================================
# Environment & Agent instantiation
# ======================================================================

env = ISACEnvironment(
    num_bs=3,       # 3 gNBs placed in a straight line (500 m apart)
    num_urllc=5,    # 5 URLLC UEs (latency-critical)
    num_embb=5,     # 5 eMBB UEs (throughput-hungry)
    Nt=8,           # 8 antennas per gNB (ULA, MRT beamforming)
    N_sc=780        # 780 sub-carriers ≈ 23.4 MHz (30 kHz SCS)
)

agent = D3QNAgent(
    state_dim     = env.state_dim,    # 8-element state vector
    action_dim    = env.action_dim,   # number of valid (sensing, URLLC, eMBB) splits
    gamma         = 0.99,             # discount factor: balance current vs future reward
    epsilon       = 1.0,              # start fully random (pure exploration)
    epsilon_decay = 0.9988            # decay per episode
)

# ======================================================================
# Training hyper-parameters
# ======================================================================

num_episodes = 10000  # total training episodes
max_steps    = 100    # time-slots per episode (one step = one resource allocation)

# ======================================================================
# Metric accumulators (one value appended per episode)
# ======================================================================

reward_history    = []   # average reward per episode
fim_history       = []   # average FIM (sensing quality)
embb_history      = []   # average eMBB throughput [Mbps]
urllc_history     = []   # average URLLC throughput [Mbps]
latency_history   = []   # average URLLC median latency [ms]
r_fim_history     = []   # average sensing sub-reward
r_embb_history    = []   # average eMBB sub-reward
r_latency_history = []   # average latency sub-reward
range_error_history = []   # CRLB-based range estimation error [metres]
range_rmse_history = []
range_rmse_fixed_history  = []   # NEW: Fixed-gap (Delta_s=10) baseline RMSE per episode
range_rmse_random_history = []   # NEW: Random-allocation baseline RMSE per episode

best_reward  = -np.inf   # track best episode reward for model checkpoint
best_episode = 0

# ======================================================================
# Training Loop
# ======================================================================

for ep in range(num_episodes):

    # reset() re-draws UE positions and channel realisations for this episode
    state = env.reset()

    # Per-episode accumulators (averaged at the end of the episode)
    total_reward  = 0.0
    total_fim     = 0.0
    total_embb    = 0.0
    total_urllc   = 0.0
    total_latency = 0.0
    total_r_fim   = 0.0
    total_r_embb  = 0.0
    total_r_lat   = 0.0
    total_rmse    = 0.0
    total_rmse_fixed  = 0.0   # NEW
    total_rmse_random = 0.0   # NEW

    for step in range(max_steps):

        # Agent selects an action (ε-greedy)
        action = agent.select_action(state)

        # Environment executes the action and returns feedback
        next_state, reward, done, info = env.step(action)

        # Mark the last step of the episode as a terminal transition
        # so the Bellman target doesn't bootstrap beyond the episode boundary
        is_terminal = (step == max_steps - 1)

        # Store the transition in the replay buffer
        agent.store_transition(state, action, reward, next_state, is_terminal)

        # Perform one mini-batch gradient update (no-op if buffer too small)
        agent.learn()

        # Advance state and accumulate metrics
        state         = next_state
        total_reward  += reward
        total_fim     += info["FIM"]
        total_embb    += info["R_embb"]
        total_urllc   += info["R_urllc"]
        total_latency += info["latency_ms"]
        total_r_fim   += info["r_fim"]
        total_r_embb  += info["r_embb"]
        total_r_lat   += info["r_latency"]
        total_rmse    += info["range_rmse"]
        total_rmse_fixed  += info["range_rmse_fixed"]    # NEW (requires patched environment.py)
        total_rmse_random += info["range_rmse_random"]   # NEW (requires patched environment.py)

    # ── Per-episode averages ──────────────────────────────────────────
    ep_reward  = total_reward  / max_steps
    ep_fim     = total_fim     / max_steps
    ep_embb    = total_embb    / max_steps / 1e6   # convert bps → Mbps
    ep_urllc   = total_urllc   / max_steps / 1e6
    ep_latency = total_latency / max_steps

    c_light        = 3e8
    ep_range_error = np.sqrt((c_light / 2)**2 / (ep_fim + 1e-30))  # Range RMSE [m]
    range_error_history.append(ep_range_error)


    reward_history.append(ep_reward)
    fim_history.append(ep_fim)
    embb_history.append(ep_embb)
    urllc_history.append(ep_urllc)
    latency_history.append(ep_latency)
    r_fim_history.append(total_r_fim    / max_steps)
    r_embb_history.append(total_r_embb  / max_steps)
    r_latency_history.append(total_r_lat / max_steps)
    range_rmse_history.append(total_rmse / max_steps)
    range_rmse_fixed_history.append(total_rmse_fixed / max_steps)     # NEW
    range_rmse_random_history.append(total_rmse_random / max_steps)   # NEW

    # ── Epsilon decay (applied after every episode) ───────────────────
    # Decreases exploration probability; floored at epsilon_min
    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

    # ── Save the best model checkpoint ───────────────────────────────
    if ep_reward > best_reward:
        best_reward  = ep_reward
        best_episode = ep + 1
        agent.save_model("d3qn_isac_best.pth")

    # ── Periodic console log every 100 episodes ───────────────────────

    if (ep + 1) % 100 == 0:
        print(
            f"Episode {ep+1:4d} | "
            f"Reward={ep_reward:.3f} | "
            f"FIM={ep_fim:.2e} | "
            f"RMSE={ep_range_error:.3f} m | "
            f"eMBB={ep_embb:.1f} Mbps | "
            f"Latency={ep_latency:.2f} ms | "
            f"ε={agent.epsilon:.3f}"
        )
# Save the final model after all episodes
agent.save_model("d3qn_isac_final.pth")
print(f"\nTraining complete. Best reward={best_reward:.3f} at episode {best_episode}.")

# ======================================================================
# NEW: Save all training curves to .npz so any plot can be redrawn later
# WITHOUT re-running the 10,000-episode training loop.
# ======================================================================
np.savez(
    "training_curves.npz",
    reward_history      = np.array(reward_history),
    fim_history          = np.array(fim_history),
    embb_history          = np.array(embb_history),
    urllc_history         = np.array(urllc_history),
    latency_history       = np.array(latency_history),
    r_fim_history         = np.array(r_fim_history),
    r_embb_history        = np.array(r_embb_history),
    r_latency_history     = np.array(r_latency_history),
    range_error_history   = np.array(range_error_history),
    range_rmse_history        = np.array(range_rmse_history),
    range_rmse_fixed_history  = np.array(range_rmse_fixed_history),
    range_rmse_random_history = np.array(range_rmse_random_history),
    num_episodes = num_episodes,
)
print("Saved training_curves.npz")


# ======================================================================
# Utility: moving-average smoother for noisy training curves
# ======================================================================

def smooth(x, w):
    """Apply a simple box-car (uniform) moving average of window w."""
    return np.convolve(x, np.ones(w) / w, mode='valid')

window = 100                             # smoothing window (episodes)
x_ax   = np.arange(window - 1, num_episodes)   # x-axis after warm-up

# ======================================================================
# Plot 1 — Total Reward Convergence
# ======================================================================
plt.figure(figsize=(6,5))
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'    
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
s = smooth(reward_history, window)
#plt.plot(x_ax, s, linewidth=2, color='steelblue', label='Smoothed reward')
plt.fill_between(x_ax, s, 0, where=s > 0, alpha=0.15, color='steelblue')
plt.axhline(0, color='gray', linestyle=':', linewidth=1)
plt.xlabel("Episodes")
plt.ylabel("Avg Episode Reward")
plt.legend(loc='upper right', fontsize=12, frameon=True)
plt.grid(True); plt.tight_layout()
plt.savefig("plot_reward.png", dpi=300)
plt.savefig("plot_reward.pdf", dpi=300)
plt.show()

# ======================================================================
# Plot 2 — eMBB Throughput
# ======================================================================
plt.figure(figsize=(6,5))
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
#plt.plot(x_ax, smooth(embb_history, window), linewidth=2, color='steelblue', label='eMBB')
plt.axhline(60, linestyle=':', color='red', linewidth=1.2, label='QoS Target (60 Mbps)')
plt.xlabel("Episodes")
plt.ylabel("Throughput (Mbps)")
plt.legend(loc='center right', bbox_to_anchor=(1,0.5), fontsize=12, frameon=True)
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_embb_throughput.png", dpi=300)
plt.savefig("plot_embb_throughput.pdf", dpi=300)
plt.show()

# ======================================================================
# Plot 3 — URLLC Latency
# ======================================================================
plt.figure(figsize=(6,5))
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
#plt.plot(x_ax, smooth(latency_history, window), linewidth=2, color='steelblue')
plt.axhline(5, linestyle='--', color='red', label='QoS target (5 ms)', linewidth=1.2)
plt.ylim(0, 10)
plt.xlabel("Episodes")
plt.ylabel("Latency (ms)")
plt.legend(loc='upper right', fontsize=12, frameon=True)
plt.grid(True); plt.tight_layout()
plt.savefig("plot_latency.png", dpi=300)
plt.savefig("plot_latency.pdf", dpi=300)
plt.show()


# ============================================================
# Plot 4: Sensing Range RMSE
# ============================================================
plt.figure(figsize=(6,5))
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
#plt.plot(x_ax, smooth(np.array(range_rmse_history), window), linewidth=2,  color='steelblue', label='Range RMSE')
plt.fill_between(x_ax, s, 0, where=s > 0, alpha=0.15, color='steelblue')
plt.xlabel("Episodes")
plt.ylabel("Range RMSE (m)")
plt.legend(loc='upper right', fontsize=12, frameon=True)
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_range_rmse.png", dpi=300)
plt.savefig("plot_range_rmse.pdf", dpi=300)
plt.show()

# ============================================================
# Plot 5 : Sensing Quality (FIM + Range RMSE)
# ============================================================
c_light = 3e8
fim_arr = np.array(fim_history)
range_rmse = np.sqrt((c_light / 2)**2 / (fim_arr + 1e-30))
fim_smooth  = smooth(fim_arr / 1e18, window)
rmse_smooth = smooth(range_rmse, window)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
# ── ADD THIS LINE BELOW TO CREATE ax1 ─────────────────────────
fig, ax1 = plt.subplots(figsize=(6, 5))
# ------------------------------------------------
# Left axis : FIM
# ------------------------------------------------
ax1.plot(x_ax, fim_smooth, linewidth=2, color='steelblue', label='FIM')
ax1.xlabel("Episodes", fontsize=12)
ax1.ylabel(r"FIM ($\times 10^{18}$)", color='steelblue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='steelblue', labelsize=10)
ax1.tick_params(axis='x', labelsize=12)
ax1.grid(True, linestyle='--', alpha=0.5)
# ------------------------------------------------
# Right axis : Range RMSE
# ------------------------------------------------
ax2 = ax1.twinx()
ax2.plot(x_ax, rmse_smooth, '--', linewidth=2, color='darkorange', label='Range RMSE (m)')
ax2.set_ylabel("Range RMSE (m)", color='darkorange', fontsize=12)
ax2.tick_params(axis='y', labelcolor='darkorange', labelsize=10)
# ------------------------------------------------
# Combined Legend
# ------------------------------------------------
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', bbox_to_anchor=(1, 0.4), fontsize=11, frameon=True)
fig.tight_layout()
plt.savefig("plot_sensing_quality.png", dpi=300, bbox_inches='tight')
plt.savefig("plot_sensing_quality.pdf", dpi=300, bbox_inches='tight')
plt.show()


# %%
# ============================================================
# Plot 6 : Proposed vs Fixed vs Random — Range RMSE comparison
# ============================================================
# This cell is SELF-CONTAINED: it reloads the saved training curves from
# "training_curves.npz" so you can re-run / re-style this plot as many
# times as you like WITHOUT re-running the training loop above.
# Just re-run this cell on its own after training has completed once.
import numpy as np
import matplotlib.pyplot as plt

_data = np.load("training_curves.npz")

_num_episodes = int(_data["num_episodes"])
_window = 100
_x_ax = np.arange(_window - 1, _num_episodes)

def _smooth(x, w):
    return np.convolve(x, np.ones(w) / w, mode='valid')

_rmse_proposed = _smooth(_data["range_rmse_history"],        _window)
_rmse_fixed    = _smooth(_data["range_rmse_fixed_history"],  _window)
_rmse_random   = _smooth(_data["range_rmse_random_history"], _window)

plt.figure(figsize=(6,5))
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.plot(_x_ax, _rmse_random, '--', linewidth=2, color='seagreen',
        label='Random subcarrier allocation')
#plt.plot(_x_ax, _rmse_fixed, '-.', linewidth=2, color='darkorange',
        #label=r'Fixed interleaved gap ($\Delta_s=10$)')
plt.plot(_x_ax, _rmse_proposed, '-', linewidth=2, color='steelblue',
        label='DJCSR (Adaptive ($\Delta_s$))')

plt.xlabel("Episodes")
plt.ylabel("Range RMSE (m)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right', fontsize=12, frameon=True)
plt.tight_layout()
plt.savefig("plot_rmse_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig("plot_rmse_comparison.pdf", dpi=300, bbox_inches='tight')
plt.show()

print(f"Final (last-100-episode-avg) RMSE — "
      f"Proposed: {_rmse_proposed[-1]:.3f} m | "
      f"Fixed: {_rmse_fixed[-1]:.3f} m | "
      f"Random: {_rmse_random[-1]:.3f} m")
# %%
