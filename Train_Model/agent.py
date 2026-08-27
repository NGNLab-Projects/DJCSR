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
# agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ======================================================================
# Replay Buffer — circular buffer storing (s, a, r, s', done) transitions
# ======================================================================

class ReplayBuffer:

    def __init__(self, capacity, state_dim):
        self.capacity    = capacity           # maximum number of transitions stored
        self.ptr         = 0                  # pointer to the next write position
        self.size        = 0                  # current number of valid entries

        # Pre-allocate numpy arrays for speed (avoids Python list overhead)
        self.state      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action     = np.zeros(capacity,              dtype=np.int64)
        self.reward     = np.zeros(capacity,              dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done       = np.zeros(capacity,              dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        """Write one transition; overwrite oldest entry when buffer is full."""
        self.state[self.ptr]      = state
        self.action[self.ptr]     = action
        self.reward[self.ptr]     = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr]       = float(done)   # store as float for Bellman equation

        # Circular increment
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Uniformly sample a mini-batch; returns PyTorch tensors."""
        idx = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.FloatTensor(self.state[idx]),       # (B, state_dim)
            torch.LongTensor(self.action[idx]),        # (B,)
            torch.FloatTensor(self.reward[idx]),       # (B,)
            torch.FloatTensor(self.next_state[idx]),   # (B, state_dim)
            torch.FloatTensor(self.done[idx])          # (B,)
        )

# ======================================================================
# Dueling DQN Network
# ======================================================================

class DuelingDQN(nn.Module):
    """
    Shared feature extractor → split into:
      Value stream  : V(s)       — how good is this state?
      Advantage stream: A(s,a)   — how much better is action a than average?
    Combined: Q(s,a) = V(s) + [A(s,a) − mean_a A(s,a)]
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Shared trunk: two hidden layers with ReLU activations
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),   # input → 128 units
            nn.Linear(128, 128),       nn.ReLU()    # 128 → 128 units
        )

        # Value stream: scalar V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)                  # outputs a single scalar
        )

        # Advantage stream: vector A(s,a) for all actions
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)          # one value per action
        )

    def forward(self, x):
        """Forward pass: returns Q-values for all actions."""
        features = self.feature(x)                    # shared representation
        v = self.value_stream(features)               # (B, 1)
        a = self.advantage_stream(features)           # (B, action_dim)
        # Combine using the mean-centred advantage trick (Wang et al., 2016)
        q = v + (a - a.mean(dim=1, keepdim=True))     # (B, action_dim)
        return q


# ======================================================================
# D3QN Agent
# ======================================================================

class D3QNAgent:

    def __init__(self,
                 state_dim,
                 action_dim,
                 lr            = 1e-4,     # Adam learning rate
                 gamma         = 0.99,     # discount factor
                 epsilon       = 1.0,      # initial exploration probability
                 epsilon_min   = 0.05,     # floor on epsilon
                 epsilon_decay = 0.9988,    # multiplicative decay per episode
                 batch_size    = 64,       # mini-batch size for gradient update
                 memory_size   = 50000,    # replay buffer capacity
                 target_update = 200):     # steps between target-net hard updates

        self.action_dim    = action_dim
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.learn_step    = 0   # counts gradient updates (used for target sync)

        # Use GPU if available, otherwise fall back to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Networks ──────────────────────────────────────────────────
        # Online net: updated every learning step via gradient descent
        self.online_net = DuelingDQN(state_dim, action_dim).to(self.device)
        # Target net: periodically synced from online net; provides stable Q targets
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())  # initialise identical
        self.target_net.eval()   # target net is never trained directly

        # ── Optimiser & loss ──────────────────────────────────────────
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn   = nn.SmoothL1Loss()   # mean squared Bellman error

        # ── Experience replay buffer ───────────────────────────────────
        self.memory = ReplayBuffer(memory_size, state_dim)

    # ------------------------------------------------------------------

    def select_action(self, state):
        """ε-greedy action selection.
        With probability ε: random action (exploration).
        Otherwise: argmax Q(s,a) from the online network (exploitation).
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)   # random action
        # Convert state to a (1, state_dim) tensor and get Q-values
        t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.online_net(t).argmax(dim=1).item()

    # ------------------------------------------------------------------

    def store_transition(self, state, action, reward, next_state, done):
        """Push a (s, a, r, s', done) tuple into the replay buffer."""
        self.memory.store(state, action, reward, next_state, done)

    # ------------------------------------------------------------------

    def learn(self):
        """Sample a mini-batch and perform one gradient update (Double DQN)."""

        # Wait until the buffer has enough samples
        if self.memory.size < self.batch_size:
            return

        s, a, r, ns, d = self.memory.sample(self.batch_size)

        # Move tensors to the compute device (GPU or CPU)
        s  = s.to(self.device)
        a  = a.to(self.device)
        r  = r.to(self.device)
        ns = ns.to(self.device)
        d  = d.to(self.device)

        # ── Current Q-values for the taken actions ─────────────────────
        # online_net(s) → (B, action_dim); gather picks Q(s,a) for each sample
        current_q = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)  # (B,)

        with torch.no_grad():
            # ── Double DQN target ──────────────────────────────────────
            # Step 1: use ONLINE net to choose the best next action
            next_a = self.online_net(ns).argmax(dim=1)           # (B,)
            # Step 2: use TARGET net to evaluate that action (reduces overestimation)
            next_q = self.target_net(ns).gather(1, next_a.unsqueeze(1)).squeeze(1)  # (B,)
            # Bellman target: r + γ * Q_target(s', a*) * (1 - done)
            target_q = r + self.gamma * next_q * (1 - d)

        # ── Gradient update ────────────────────────────────────────────
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients with sparse rewards
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ── Periodic hard update of the target network ─────────────────
        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------

    def save_model(self, path):
        """Persist the online network's weights to disk."""
        torch.save(self.online_net.state_dict(), path)

    def load_model(self, path):
        """Restore weights and sync both nets (for inference or fine-tuning)."""
        self.online_net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.target_net.load_state_dict(self.online_net.state_dict())

# %%

# %%

