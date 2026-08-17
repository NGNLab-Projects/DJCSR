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

# environment.py

import numpy as np

class ISACEnvironment:

    def __init__(self,
                 num_bs=3,           # Number of base stations (gNBs)
                 num_urllc=5,        # Number of URLLC UEs (ultra-reliable, low-latency)
                 num_embb=5,         # Number of eMBB UEs (enhanced mobile broadband)
                 Nt=8,               # Transmit antennas per gNB (ULA)
                 N_sc=780,           # Total subcarriers (NR numerology 1, ~25 MHz)
                 cell_radius=500):   # Cell radius in metres

        # ── System dimensions ──────────────────────────────────────────
        self.num_bs      = num_bs
        self.num_urllc   = num_urllc
        self.num_embb    = num_embb
        self.total_users = num_urllc + num_embb
        self.Nt          = Nt          # 4 antennas per gNB as required
        self.N_sc        = N_sc
        self.cell_radius = cell_radius

        # ── Radio parameters (3GPP NR Band n78) ───────────────────────
        self.fc       = 3.5e9                  # Carrier frequency: 3.5 GHz
        self.c        = 3e8                    # Speed of light (m/s)
        self.lambda_c = self.c / self.fc       # Wavelength ≈ 0.0857 m
        self.SCS      = 30e3                   # Subcarrier spacing: 30 kHz (numerology µ=1)
        self.BW       = self.N_sc * self.SCS   # Total bandwidth ≈ 23.4 MHz

        # Transmit power: 43 dBm ≈ 20 W (macro BS)
        self.Pt_dBm = 26
        self.Pt     = 10 ** ((self.Pt_dBm - 30) / 10)  # Convert dBm → Watts

        # Thermal noise power: N0 = -174 dBm/Hz + 10log10(BW) + NF(5dB)
        noisePower_dBm = -174 + 10 * np.log10(self.BW) + 5
        self.sigma2    = 10 ** ((noisePower_dBm - 30) / 10)   # Watts

        # ── Calibrated reference values (used to normalise the reward) ─
        self.FIM_ref  = 1e18   # Typical median FIM for sensing reward baseline
        self.embb_ref = 60e6   # Target eMBB aggregate rate [bps]
        self.lat_ref  = 0.010  # Target URLLC latency: 10 ms [s]

        # ── gNB positions: 3 sites in a STRAIGHT LINE along the x-axis ─
        # Inter-site distance (ISD) = 500 m, centred at origin.
        # Previously these were arranged as a triangle; changed to linear.
        isd = 500.0   # inter-site distance (metres)
        self.bs_pos = np.array([
            [-isd,  0.0],   # gNB-0: left
            [  0.0, 0.0],   # gNB-1: centre
            [ isd,  0.0],   # gNB-2: right
        ], dtype=float)

        # ── Subcarrier frequency offsets (symmetric around DC) ─────────
        # Used in FIM (CRLB) calculation to weight sensing sub-carriers
        sc_idx     = np.arange(-self.N_sc // 2, self.N_sc // 2)  # e.g., -306 … +305
        self.fk    = sc_idx * self.SCS      # Absolute frequency offset of each SC [Hz]
        self.fk_sq = self.fk ** 2           # Squared frequencies, used directly in FIM formula

        # ── State / Action spaces ──────────────────────────────────────
        self.state_dim = 16

        # ==================================================
        # Power Allocation Actions
        # ==================================================
        self.power_ratios = [
            [0.50,0.25,0.25],   # sensing
            [0.40,0.30,0.30],   # sensing-balanced
            [0.33,0.33,0.34],   # balanced
            [0.25,0.50,0.25],   # URLLC
            [0.20,0.30,0.50]    # eMBB
        ]
        # ── Incremental (delta) resource-allocation actions ────────────
        # Each action nudges sensing/eMBB SC counts up, down, or holds.
        # URLLC always gets whatever SCs remain.
        self.step_sizes = [5, 10, 15, 20, 25]
        self.spacing_actions = [1, 2, 5, 10, 20]

        self.action_table = []
        for sa in [-1, 0, 1]:           # sensing direction
            for ea in [-1, 0, 1]:       # eMBB direction
                for spacing in self.spacing_actions:
                    if sa == 0 and ea == 0:
                        self.action_table.append([sa, ea, 0, spacing])
                    else:
                        for step_size in self.step_sizes:
                            self.action_table.append([sa, ea, step_size, spacing])

        self.action_table = np.array(self.action_table)  # shape: (n_resource_actions, 4)
        self.action_dim = len(self.action_table) * len(self.power_ratios)
    # ==================================================
    # LOS Probability
    # ==================================================
    def get_los_probability(self, d2d):

        p_los = np.ones_like(d2d)

        idx = d2d > 18.0
        p_los[idx] = (18.0 / d2d[idx] + np.exp(-d2d[idx] / 63.0) * (1.0 - 18.0 / d2d[idx]))
        return np.clip(p_los, 0.0, 1.0)

    # ==================================================================
    # RESET  — called at the start of every episode
    # UEs are placed RANDOMLY in each episode as required.
    # ==================================================================

    def reset(self):

        self.user_x = self.cell_radius * (2*np.random.rand(self.total_users)-1)
        self.user_y = self.cell_radius * (2*np.random.rand(self.total_users)-1)

        self.user_pos = np.stack([self.user_x, self.user_y], axis=1)

        # ── Distance matrix: shape (num_bs, total_users) ──────────────
        dx  = self.bs_pos[:, 0][:, None] - self.user_pos[:, 0][None, :]
        dy  = self.bs_pos[:, 1][:, None] - self.user_pos[:, 1][None, :]

        d2d = np.sqrt(dx**2 + dy**2)   # 2-D ground distance [m]
        # --------------------------------------------------
        # Initial target range
        # --------------------------------------------------
        self.target_x = 100.0
        self.target_y = 50.0

        target_range = np.sqrt(self.target_x**2 + self.target_y**2)
        target_range_norm = np.clip(target_range / self.cell_radius, 0.0, 1.0)

        self.target_range = np.random.uniform(100, 200, self.num_bs)
        self.rcs = 10.0     # vehicle RCS 5-20
        # Radar echo gain (monostatic radar equation)
        self.target_gain = (self.rcs * self.lambda_c**2 / ((4*np.pi)**3 * (self.target_range**4 + 1e-12)))
        # Optional small-scale fading
        self.target_channel = np.sqrt(self.target_gain[:,None]/2) * (np.random.randn(self.num_bs,self.N_sc) + 1j*np.random.randn(self.num_bs,self.N_sc))
        self.bk = np.abs(self.target_channel)

        # UE pedestrian mobility (m/s)
        self.user_speed = np.random.uniform(0.5, 2.0, self.total_users)
        self.user_theta = np.random.uniform(0, 2*np.pi, self.total_users)

        # Target vehicle mobility (m/s)
        self.target_speed = np.random.uniform(10.0, 20.0)
        self.target_theta = np.random.uniform(0, 2*np.pi)

        # --------------------------------------------------
        # Doppler parameters
        # --------------------------------------------------

        self.fd_ue = (np.mean(self.user_speed) * self.fc / self.c)
        self.fd_target = (self.target_speed * self.fc / self.c)

        # ── LOS probability (3GPP TR 38.901 UMa model) ────────────────
        d2d = np.sqrt(dx**2 + dy**2)   # 2-D ground distance [m]

        # 3-D distance including height difference (BS at 25 m, UE at 1.5 m)
        d3d = np.sqrt(d2d**2 + (25.0 - 1.5)**2)

        # ==================================================
        # Stage-4 LOS / NLOS Model (3GPP TR 38.901)
        # ==================================================
        fc_GHz = 3.5
        h_BS = 25.0
        h_UT = 1.5

        # LOS Probability
        p_los = self.get_los_probability(d2d)

        los_mask = (np.random.rand(*p_los.shape) < p_los)

        # Breakpoint Distance
        d_bp = (4 * (h_BS - 1.0) * (h_UT - 1.0) * fc_GHz * 1e9 / self.c)

        # LOS Pathloss
        PL1 = (28.0 + 22.0*np.log10(d3d + 1e-9) + 20.0*np.log10(fc_GHz))
        PL2 = (28.0 + 40.0*np.log10(d3d + 1e-9) + 20.0*np.log10(fc_GHz) - 9.0*np.log10(d_bp**2 + (h_BS - h_UT)**2))

        pl_los = np.where(d2d <= d_bp, PL1, PL2)

        # NLOS Pathloss
        pl_nlos_prime = (13.54 + 39.08*np.log10(d3d + 1e-9) + 20*np.log10(fc_GHz) - 0.6*(h_UT - 1.5))
        pl_nlos = np.maximum(pl_los, pl_nlos_prime)

        # LOS/NLOS Selection
        pl_db = np.where(los_mask, pl_los, pl_nlos)

        # Shadowing
        shadow_sigma = np.where(los_mask, 4.0, 7.8)
        self.shadow_sigma = shadow_sigma
        shadow_db = np.random.normal(0.0, shadow_sigma)
        self.shadow_db = shadow_db
        avg_shadow = np.mean(np.abs(shadow_db))
        pl_db += shadow_db
        self.los_ratio = np.mean(los_mask)
        avg_shadow_norm = np.clip(avg_shadow / 10.0, 0.0, 1.0)
        PL = 10**(-pl_db/10)
        # --------------------------------------------------
        # Strongest received power association
        # --------------------------------------------------
        received_power = self.Pt * PL
        self.association = np.argmax(received_power, axis=0)

        # ── Channel matrix H: shape (total_users, num_bs, Nt, N_sc) ──
        # Rayleigh fading per (BS, UE, antenna, subcarrier) combination.
        # Path loss is applied so |H|² reflects actual received power.
        # --------------------------------------------------
        # Initial Rayleigh fading (without pathloss)
        # --------------------------------------------------
        self.H_fading = (np.random.randn(self.total_users, self.num_bs, self.Nt, self.N_sc)
            + 1j*np.random.randn(self.total_users, self.num_bs, self.Nt, self.N_sc)) / np.sqrt(2)

        # --------------------------------------------------
        # Apply pathloss
        # --------------------------------------------------
        self.H = (np.sqrt(PL.T[:,:,None,None]) * self.H_fading)

        # Save path-loss for cell-edge compensation (used in step())
        self.PL_matrix = PL  # shape (num_bs, total_users)

        # ── Traffic demands (randomly drawn each episode) ──────────────

        self.D_urllc = np.full(self.num_urllc, 480)      # Bytes per slot
        self.D_embb  = np.full(self.num_embb, 1024)      # Bytes per slot

        # ------------------------------------------------------------------
        # Persistent SC allocation state (incremental/delta action design)
        # Initialised to an equal-ish 3-way split; step() nudges these.
        # ------------------------------------------------------------------
         # ------------------------------------------------------------------
        # Persistent SC allocation state
        # ------------------------------------------------------------------
        
        # During Monte-Carlo evaluation, use a fixed sensing allocation
        if hasattr(self, "fixed_sensing_sc"):
            self.sensing_sc = int(self.fixed_sensing_sc)
            self.embb_sc = (self.N_sc - self.sensing_sc) // 2
            self.urllc_sc = self.N_sc - self.sensing_sc - self.embb_sc
        else:
            # Default initialization used during RL training
            self.sensing_sc = self.N_sc // 10
            self.embb_sc = self.N_sc // 3
            self.urllc_sc = self.N_sc - self.sensing_sc - self.embb_sc
        
        self.first_step = True

        urllc_ratio   = self.urllc_sc   / self.N_sc
        embb_ratio    = self.embb_sc    / self.N_sc
        sensing_ratio = self.sensing_sc / self.N_sc

        state = np.array([
          0.50,                          # avg_urllc_sinr_norm
          0.50,                          # min_urllc_sinr_norm
          0.50,                          # avg_embb_sinr_norm
          0.50,                          # min_embb_sinr_norm (new)
          0.00,                          # latency_norm
          0.50,                          # embb_rate_norm
          0.50,                          # fim_norm_state
          urllc_ratio,                   # urllc_sc_ratio (new)
          embb_ratio,                    # embb_sc_ratio  (new)
          sensing_ratio,                 # sensing_ratio
          0.33,                          # p_urllc_ratio (new)
          0.33,                          # p_embb_ratio  (new)
          0.33,                          # p_sense_ratio
          target_range_norm,
          self.los_ratio,
          avg_shadow_norm
      ], dtype=np.float32)
        return state

    # ==================================================================
    # STEP  — called once per time-slot with a chosen action index
    # ==================================================================

    def step(self, action):

        # ==================================================
        # Joint Resource (delta) + Power Action Decode
        # ==================================================
        num_resource_actions = len(self.action_table)
        num_power_actions    = len(self.power_ratios)

        resource_idx = action // num_power_actions
        power_idx    = action % num_power_actions

        sa, ea, step_size, spacing = self.action_table[resource_idx]
        spacing = int(spacing)

        p_sense_ratio = self.power_ratios[power_idx][0]
        p_urllc_ratio = self.power_ratios[power_idx][1]
        p_embb_ratio  = self.power_ratios[power_idx][2]

        spacing_norm = spacing / max(self.spacing_actions)

        # ── Apply the incremental SC update ────────────────────────────
        infeasible_action = False
        if self.first_step:
            new_sensing = self.sensing_sc
            new_embb    = self.embb_sc
            self.first_step = False
        else:
            new_sensing = self.sensing_sc + sa * step_size
            new_embb    = self.embb_sc    + ea * step_size

        new_urllc = self.N_sc - new_sensing - new_embb

        feasible = (
            new_sensing >= self.num_bs and        # at least 1 SC/BS for sensing
            new_embb    >= self.num_embb and       # at least 1 SC/UE for eMBB
            new_urllc   >= self.num_urllc and      # at least 1 SC/UE for URLLC
            (new_sensing // 2) * spacing + 1 <= self.N_sc   # spacing fits in band
        )

        if feasible:
            self.sensing_sc = new_sensing
            self.embb_sc    = new_embb
        else:
            infeasible_action = True
        self.urllc_sc = self.N_sc - self.sensing_sc - self.embb_sc

        sensing_ratio = self.sensing_sc / self.N_sc
        urllc_ratio   = self.urllc_sc   / self.N_sc
        embb_ratio    = self.embb_sc    / self.N_sc

        # ==================================================
        # Stage-1 Mobility Update
        # ==================================================
        dt = 0.01   # slot duration
        # --------------------------------------------------
        # Move UEs
        # --------------------------------------------------
        self.user_x += (self.user_speed *  np.cos(self.user_theta) * dt)
        self.user_y += (self.user_speed *  np.sin(self.user_theta) * dt)

        # --------------------------------------------------
        # Move Target
        # --------------------------------------------------
        self.target_x += (self.target_speed * np.cos(self.target_theta) * dt)
        self.target_y += (self.target_speed * np.sin(self.target_theta) * dt)

        # --------------------------------------------------
        # Rebuild UE positions
        # --------------------------------------------------
        self.user_pos = np.stack( [self.user_x, self.user_y], axis=1)

        # --------------------------------------------------
        # Keep User Inside the Cell radius
        # --------------------------------------------------
        r = np.sqrt(self.user_x**2 + self.user_y**2)
        outside = r > self.cell_radius

        self.user_x[outside] *= (self.cell_radius / r[outside])
        self.user_y[outside] *= (self.cell_radius / r[outside])

        self.user_pos = np.stack([self.user_x, self.user_y], axis=1)
        # --------------------------------------------------
        # Keep Target Inside the Cell radius
        # --------------------------------------------------
        target_r = np.sqrt( self.target_x**2 + self.target_y**2)

        if target_r > self.cell_radius:

            self.target_x *= (self.cell_radius /target_r)
            self.target_y *= (self.cell_radius /target_r)

        # ── Distance matrix: shape (num_bs, total_users) ──────────────
        dx  = self.bs_pos[:, 0][:, None] - self.user_pos[:, 0][None, :]
        dy  = self.bs_pos[:, 1][:, None] - self.user_pos[:, 1][None, :]

        d2d = np.sqrt(dx**2 + dy**2)   # 2-D ground distance [m]

        # 3-D distance including height difference (BS at 25 m, UE at 1.5 m)
        d3d = np.sqrt(d2d**2 + (25.0 - 1.5)**2)

        # ==================================================
        # LOS / NLOS Update
        # ==================================================
        fc_GHz = 3.5
        h_BS = 25.0
        h_UT = 1.5

        p_los = self.get_los_probability(d2d)

        los_mask = (np.random.rand(*p_los.shape) < p_los)

        d_bp = (4 * (h_BS - 1.0) * (h_UT - 1.0) * fc_GHz * 1e9 /self.c)

        PL1 = (28.0 + 22.0*np.log10(d3d + 1e-9) + 20.0*np.log10(fc_GHz))
        PL2 = (28.0 + 40.0*np.log10(d3d + 1e-9) + 20.0*np.log10(fc_GHz) - 9.0*np.log10(d_bp**2 + (h_BS - h_UT)**2))

        pl_los = np.where(d2d <= d_bp, PL1, PL2)

        pl_nlos_prime = (13.54 + 39.08*np.log10(d3d + 1e-9) + 20*np.log10(fc_GHz) - 0.6*(h_UT - 1.5))
        pl_nlos = np.maximum(pl_los, pl_nlos_prime)

        pl_db = np.where(los_mask, pl_los, pl_nlos)

        shadow_sigma = np.where(los_mask, 4.0, 7.8)
        shadow_db = np.random.normal(0.0, shadow_sigma)
        avg_shadow = np.mean(np.abs(shadow_db))
        avg_shadow_norm = np.clip(avg_shadow / 10.0, 0.0, 1.0)

        pl_db += shadow_db
        self.los_ratio = np.mean(los_mask)
        PL = 10**(-pl_db/10)
        self.PL_matrix = PL
        # ==================================================
        # Correlated Fading
        # ==================================================
        rho = 0.98  # Correlation Coefficient
        W = (np.random.randn(*self.H_fading.shape) + 1j*np.random.randn(*self.H_fading.shape) ) / np.sqrt(2)
        self.H_fading = (rho * self.H_fading + np.sqrt(1-rho**2) * W)
        self.H = (np.sqrt(self.PL_matrix.T[:,:,None,None]) * self.H_fading)

        # --------------------------------------------------
        # Dynamic gNB association
        # --------------------------------------------------
        received_power = self.Pt * PL
        self.association = np.argmax(received_power, axis=0)

       # SC counts already finalised above by the incremental action decode.
        sensing_sc = self.sensing_sc
        urllc_sc   = self.urllc_sc
        embb_sc    = self.embb_sc

        # URLLC sub-carrier allocation ──────────────────────────
        urllc_counts = np.full(
            self.num_urllc,
            urllc_sc // self.num_urllc
        )
        urllc_counts[:urllc_sc % self.num_urllc] += 1

        # Build per-UE SC index arrays (contiguous blocks, starting after sensing SCs)
        urllc_user_sc = []
        ptr = int(sensing_sc)
        for uu in range(self.num_urllc):
            urllc_user_sc.append(np.arange(ptr, ptr + urllc_counts[uu]))
            ptr += urllc_counts[uu]
        # ptr == sensing_sc + urllc_sc exactly; use it as the eMBB start
        embb_start = ptr

        # ── 2b. eMBB sub-carrier allocation (same approach) ───────────

        embb_counts = np.full(
            self.num_embb,
            embb_sc // self.num_embb
        )
        embb_counts[:embb_sc % self.num_embb] += 1

        # Build per-UE SC index arrays (contiguous blocks, starting after sensing SCs)
        embb_user_sc = []
        ptr = embb_start   # start right after the last URLLC SC
        for ii in range(self.num_embb):
            embb_user_sc.append(np.arange(ptr, ptr + embb_counts[ii]))
            ptr += embb_counts[ii]
        # ptr == N_sc exactly; no SC index can go out of bounds

        H = self.H  # local alias for brevity
        # --------------------------------------------------
        # Power per subcarrier (Total BS power distributed
        # across all OFDM subcarriers)
        # --------------------------------------------------
        P_sense_total = self.Pt * p_sense_ratio
        P_urllc_total = self.Pt * p_urllc_ratio
        P_embb_total  = self.Pt * p_embb_ratio

        P_sc_sense = P_sense_total / max(sensing_sc,1)
        P_sc_urllc = P_urllc_total / max(urllc_sc,1)
        P_sc_embb  = P_embb_total  / max(embb_sc,1)

        # ── 3a. URLLC throughput with MRT beamforming ─────────────────
        # MRT (Maximum Ratio Transmission) aligns the beamforming vector
        # with the serving channel: w = h*/||h||. Signal power ∝ Nt*||h||².
        # Each gNB transmits independently (no coordination → inter-cell
        # interference remains).

        R_urllc_users   = np.zeros(self.num_urllc)   # bps per URLLC UE
        urllc_sinr_list = []                         # collect SINRs for the state

        for uu in range(self.num_urllc):
            bs = self.association[uu]   # serving gNB index
            Pt_eff = P_sc_urllc * len(urllc_user_sc[uu])
            sc_sinrs = []
            for sc in urllc_user_sc[uu]:
                h_s = H[uu, bs, :, sc]   # serving channel vector [Nt complex]

                # MRT signal power: ||h||² * Pt_eff * Nt (coherent combining gain)
                sig = Pt_eff * np.linalg.norm(h_s)**2 * self.Nt

                # Inter-cell interference: sum over non-serving BSs (worst case)
                intf = sum(Pt_eff * np.linalg.norm(H[uu, b, :, sc])**2
                    for b in range(self.num_bs) if b != bs)

                sinr = sig / (intf + self.sigma2)   # post-beamforming SINR
                sc_sinrs.append(sinr)
                #urllc_sinr_list.append(sinr)
                # Shannon capacity per sub-carrier [bps] = SCS * log2(1+SINR)
                R_urllc_users[uu] += self.SCS * np.log2(1 + sinr)
            urllc_sinr_list.append(float(np.mean(sc_sinrs)))  # one value per user

        # ── 3b. eMBB throughput (same MRT structure, no extra power boost) ─
        R_embb_users   = np.zeros(self.num_embb)
        embb_sinr_list = []

        for ii, uu in enumerate(range(self.num_urllc, self.total_users)):
            bs      = self.association[uu]
            Pt_eff = P_sc_embb * len(embb_user_sc[ii])
            sc_sinrs = []
            for sc in embb_user_sc[ii]:
                h_s  = H[uu, bs, :, sc]
                sig  = Pt_eff * np.linalg.norm(h_s)**2 * self.Nt
                intf = sum(
                     Pt_eff * np.linalg.norm(H[uu, b, :, sc])**2
                    for b in range(self.num_bs) if b != bs
                )
                sinr = sig / (intf + self.sigma2)
                sc_sinrs.append(sinr)
                #embb_sinr_list.append(sinr)
                R_embb_users[ii] += self.SCS * np.log2(1 + sinr)   # bps per SC
            embb_sinr_list.append(float(np.mean(sc_sinrs)))   # one value per user

        R_urllc = float(np.sum(R_urllc_users))   # aggregate URLLC rate [bps]
        R_embb  = float(np.sum(R_embb_users))    # aggregate eMBB rate [bps]

        # --------------------------------------------------
        # eMBB QoS Satisfaction Ratio
        # --------------------------------------------------
        embb_qos_ok = R_embb_users >= 10e6
        embb_qos_ratio = float(np.mean(embb_qos_ok))

        # ── 4. URLLC per-UE latency ────────────────────────────────────
        # Latency ≈ packet_size / achievable_rate (simple M/D/1 approximation)
        # D_urllc [bytes] → bits / rate [bps] = seconds
        min_rate = 1e4   # 10 kbps floor
        per_user_lat   = self.D_urllc * 8 / (R_urllc_users + min_rate)# seconds
        urllc_latency  = float(np.max(per_user_lat))                  # maximum latency [s]
        per_user_lat_ms = per_user_lat * 1000.0                       # convert to ms

        R_embb  = float(np.sum(R_embb_users))
        # ==================================================
        # eMBB Delay
        # ==================================================
        min_rate = 1e4
        per_user_embb_delay = (self.D_embb * 8 /(R_embb_users + min_rate))
        embb_delay_ms = per_user_embb_delay * 1000.0
        avg_embb_delay_ms = float(np.mean(embb_delay_ms))

        urllc_qos_ok = per_user_lat_ms < 5.0
        urllc_qos_ratio = float(np.mean(urllc_qos_ok))

        # ── 5. Sensing FIM (Cramér–Rao Lower Bound for range estimation) ──
        def _sensing_idx(n_sc, sp):
            """Band-edge symmetric subcarrier selection (same rule as before)."""
            if n_sc <= 0:
                return np.array([], dtype=int)
            half_low  = n_sc // 2
            half_high = n_sc - half_low
            low_stop   = min(half_low * sp, self.N_sc // 2)
            high_start = max(self.N_sc - min(half_high * sp, self.N_sc // 2), self.N_sc // 2)
            low_idx  = np.arange(0, low_stop, sp)
            high_idx = np.arange(high_start, self.N_sc, sp)
            idx = np.concatenate([low_idx, high_idx])
            return np.clip(idx, 0, self.N_sc - 1).astype(int)

        def _fim_crlb_rmse(sensing_idx, P_sense_total):
            """Exact FIM -> CRLB -> RMSE formula, unchanged from the original."""
            n_idx = len(sensing_idx)
            if n_idx == 0:
                return 1e-9, np.inf, np.inf
            BW_sensing = n_idx * self.SCS
            noisePower_sensing_dBm = (-174 + 10*np.log10(BW_sensing) + 5)
            sigma2_sensing = 10**((noisePower_sensing_dBm - 30)/10)
            P_sc_sense = P_sense_total / n_idx
            FIM_ = 0.0
            for n in range(self.num_bs):
                bk_sq = self.bk[n, sensing_idx]**2
                J_n = ((8*np.pi**2) / sigma2_sensing) * np.sum(
                    self.fk_sq[sensing_idx] * bk_sq * P_sc_sense * self.Nt)
                FIM_ += J_n
            CRLB_ = self.c**2 / (4*FIM_ + 1e-12)
            return FIM_, CRLB_, np.sqrt(CRLB_)

        FIXED_SPACING = 5   # baseline interleaved gap for comparison

        if sensing_sc > 0:
            # ---- Proposed: the agent's OWN chosen spacing -> drives reward/state ----
            sensing_idx = _sensing_idx(sensing_sc, spacing)
            FIM, CRLB_range, range_rmse = _fim_crlb_rmse(sensing_idx, P_sense_total)

            # ---- Fixed interleaved gap baseline (Delta_s = 5), same sc count & power ----
            fixed_idx = _sensing_idx(sensing_sc, FIXED_SPACING)
            FIM_fixed, CRLB_fixed, range_rmse_fixed = _fim_crlb_rmse(fixed_idx, P_sense_total)

            # ---- Random subcarrier allocation baseline, same sc count & power ----
            random_idx = np.random.choice(self.N_sc, size=sensing_sc, replace=False)
            FIM_random, CRLB_random, range_rmse_random = _fim_crlb_rmse(random_idx, P_sense_total)

        else:
            FIM = 1e-9
            CRLB_range = np.inf
            range_rmse = np.inf
            FIM_fixed = FIM_random = 1e-9
            CRLB_fixed = CRLB_random = np.inf
            range_rmse_fixed = range_rmse_random = np.inf

        # ── 6. Reward shaping ─────────────────────────────────────────
        # ==========================================
        # URLLC
        # ==========================================
        Ru = R_urllc_users      # per-user URLLC throughput
        Rmin_u = 5e6            # 5 Mbps target

        Du = per_user_lat_ms
        Dmax_u = 5.0            # 5 ms target

        term_u_rate = np.mean(np.clip((Ru - Rmin_u)/Rmin_u, -1.0, 1.0))
        term_u_lat = np.mean(np.clip((Dmax_u - Du)/Dmax_u, -1.0, 1.0))

        alpha_u = 0.4
        beta_u  = 0.6
        term_u = (alpha_u * term_u_rate + beta_u  * term_u_lat)

        # ==========================================
        # eMBB
        # ==========================================
        Re = R_embb_users
        Rmin_e = 60e6     # 60 Mbps target

        De = embb_delay_ms
        Dmax_e = 10.0

        term_e_rate = np.mean(np.clip((Re - Rmin_e)/Rmin_e, -1.0, 1.0))
        term_e_lat = np.mean(np.clip((Dmax_e - De)/Dmax_e, -1.0, 1.0))

        alpha_e = 0.8
        beta_e  = 0.2
        term_e = (alpha_e * term_e_rate + beta_e  * term_e_lat)
        # ==========================================
        # Sensing
        # ==========================================
        #term_s = np.clip((np.log10(FIM + 1e-30) - 20)/3, 0, 1)
        term_s = np.clip((np.log10(FIM + 1e-30) - 18) / 3, 0, 1)

        # ==========================================
        # QoS penalties
        # ==========================================
        penalty_delay_u = np.mean(np.maximum(0, (Du - Dmax_u) / Dmax_u))
        penalty_rate_e = np.mean(np.maximum(0, (Rmin_e - Re) / Rmin_e))

        total_penalty = (0.6 * penalty_delay_u + 0.4 * penalty_rate_e)

        # ==========================================
        # Final Reward
        # ==========================================
        #reward = (0.35 * term_u +  0.35 * term_e + 0.30 * term_s)
        #reward -= 0.2 * total_penalty

        w1 = 1.0
        w2 = 5.0
        w3 = 5.0
        reward = (w1 * (-np.log10(CRLB_range))- w2 * penalty_rate_e- w3 * penalty_delay_u)
        if infeasible_action:
            reward -= 20   # discourage choosing infeasible deltas

        #reward = (w1 * term_s - w2 * penalty_rate_e - w3 * penalty_delay_u)

        # ── 7. Build next state vector (8 elements) ───────────────────
        # log10 compresses the SINR dynamic range to a learner-friendly scale
        #u_sinr = float(np.mean(urllc_sinr_list)) if urllc_sinr_list else 1.0
        #e_sinr = float(np.mean(embb_sinr_list))  if embb_sinr_list  else 1.0

        # ==================================================
        # SINR State Features
        # ==================================================
        # ------------------------------------------------------------------
        # Normalised latency
        # ------------------------------------------------------------------
        lat_ms = float(np.mean(per_user_lat_ms))
        #latency_norm = np.clip(lat_ms / 100.0, 0.0, 1.0)
        latency_norm = np.clip(lat_ms / (self.lat_ref * 1000), 0.0, 1.0)

        # ------------------------------------------------------------------
        # Normalised eMBB throughput
        # ------------------------------------------------------------------
        embb_rate_norm = np.clip(np.mean(R_embb_users) / self.embb_ref, 0.0, 1.0)
        # ------------------------------------------------------------------
        # Normalised FIM
        # ------------------------------------------------------------------
        fim_norm_state = np.clip((np.log10(FIM + 1e-30) - 18) / 4,  0.0, 1.0)

        # Average SINR (dB), normalised to [0,1] over [-10, 40] dB range
        # Per-user mean SINRs (lists now have 5 entries each)
        avg_urllc_sinr = float(np.mean(urllc_sinr_list))
        min_urllc_sinr = float(np.min(urllc_sinr_list))   # worst URLLC user
        avg_embb_sinr  = float(np.mean(embb_sinr_list))
        min_embb_sinr  = float(np.min(embb_sinr_list))    # worst eMBB user

        avg_urllc_sinr_db  = 10*np.log10(max(avg_urllc_sinr, 1e-10))
        min_urllc_sinr_db  = 10*np.log10(max(min_urllc_sinr, 1e-10))
        avg_embb_sinr_db   = 10*np.log10(max(avg_embb_sinr,  1e-10))
        min_embb_sinr_db   = 10*np.log10(max(min_embb_sinr,  1e-10))

        avg_urllc_sinr_norm = np.clip((avg_urllc_sinr_db + 10) / 50.0, 0.0, 1.0)
        min_urllc_sinr_norm = np.clip((min_urllc_sinr_db + 10) / 50.0, 0.0, 1.0)
        avg_embb_sinr_norm  = np.clip((avg_embb_sinr_db  + 10) / 50.0, 0.0, 1.0)
        min_embb_sinr_norm  = np.clip((min_embb_sinr_db  + 10) / 50.0, 0.0, 1.0)

        urllc_sc_ratio = urllc_sc / self.N_sc
        embb_sc_ratio  = embb_sc  / self.N_sc

        # target range Norm
        target_range = np.sqrt(self.target_x**2 + self.target_y**2)
        target_range_norm = np.clip(target_range / self.cell_radius, 0.0,  1.0)
        # ------------------------------------------------------------------
        # QoS-aware ISAC State Vector
        # ------------------------------------------------------------------

        state = np.array([
          avg_urllc_sinr_norm,   # 0
          min_urllc_sinr_norm,   # 1  — worst URLLC user
          avg_embb_sinr_norm,    # 2
          min_embb_sinr_norm,    # 3  — worst eMBB user (new)
          latency_norm,          # 4
          embb_rate_norm,        # 5
          fim_norm_state,        # 6
          urllc_sc_ratio,        # 7  — action feedback (new)
          embb_sc_ratio,         # 8  — action feedback (new)
          sensing_ratio,         # 9
          p_urllc_ratio,         # 10 — action feedback (new)
          p_embb_ratio,          # 11 — action feedback (new)
          p_sense_ratio,         # 12
          target_range_norm,     # 13
          self.los_ratio,        # 14
          avg_shadow_norm        # 15
      ], dtype=np.float32)

        # ── 8. Info dictionary (detailed diagnostics) ─────────────────
        info = {
            "FIM"        : float(FIM),
            "range_rmse" : float(range_rmse),
            "crlb_range" : float(CRLB_range),
            "range_rmse_fixed"  : float(range_rmse_fixed),
            "range_rmse_random" : float(range_rmse_random),
            "R_embb"     : float(R_embb),
            "R_urllc"    : float(R_urllc),
            "r_fim"      : float(term_s),
            "r_embb"     : float(term_e),
            "r_latency"  : float(term_u),
            "latency_ms" : float(urllc_latency * 1e3),
            "term_u"     : float(term_u),
            "term_e"     : float(term_e),
            "term_s"     : float(term_s),
            "total_penalty" : float(total_penalty),
            "urllc_qos_ratio" : float(urllc_qos_ratio),
            "embb_qos_ratio"  : float(embb_qos_ratio),
        }
        # done=False: episodes run for max_steps; termination handled in train.py
        return state, reward, False, info

# %%


# %%
