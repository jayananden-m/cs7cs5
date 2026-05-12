import os


SUMO_CFG = os.path.join(os.path.dirname(__file__), "scenarios", "single-lane", "single-lane.sumocfg")
STEP_LENGTH = 0.1
MAX_STEPS = 1000

# Vehicles
N_AGENTS = 5
LEADER_ID = "leader0"
AGENT_IDS = [f"agent{i}" for i in range(N_AGENTS)]
ALL_IDS = [LEADER_ID] + AGENT_IDS

# Observation
K_NEIGHBORS = 3
LOCAL_OBS_DIM = K_NEIGHBORS * 3
GLOBAL_OBS_DIM = N_AGENTS * LOCAL_OBS_DIM

# Action
DELTA_A_MAX = 1.0 # max acceleration per step (m/s^2)
A_MIN = -4.5
A_MAX = 3.0

# Normalization constants
GAP_NORM = 100.0 # m
RELV_NORM = 15.0 # m/s
V_NORM = 30.0 # m/s
V_MAX = 30.0 # speed cap for agents (m/s)

# IDM parameters (desired gap)
D0 = 7.0 # standstill gap (m)
TAU = 1.5 # time headway (s)

# Safety thresholds
HARD_MIN_GAP = 1.0 # collision threshold (m)
RUNAWAY_GAP = 200.0 # lost platoon threshold (m)
LEADER_N_EVENTS = 3 # braking disturbance events per episode


# Reward weights (Chen et al. 2023)
# r = -W_SPACING*(e_gap/GAP_NORM)^2 - W_VELOCITY*(Δv/RELV_NORM)^2
#     -W_ACCEL*(a/A_MAX)^2 - W_SAFETY*ReLU(2*SAFETY_GAP - gap)^2

# NOTE: raising W_SPACING beyond ~10 without reducing RUNAWAY_GAP causes
#       critic divergence — squared errors grow as (gap/GAP_NORM)^2 and at
#       200 m the penalty becomes ~3.24 × W_SPACING per step.

# Reward Weights
W_SPACING = 10.0
W_VELOCITY = 5.0
W_ACCEL = 0.5
W_SAFETY = 50.0
SAFETY_GAP = 5.0 # penalty triggers when gap < 2 * SAFETY_GAP (10m)

# Terminal penalties
W_COLLISION = 1000.0
W_RUNAWAY = 50.0

# Training hyperparameters
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.001
VALUE_COEF = 0.5
PPO_EPOCHS = 10
BATCH_SIZE = 128
MAX_GRAD_NORM = 0.5
