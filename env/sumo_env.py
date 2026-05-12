import numpy as np
import traci

import config as cfg
from env.leader_profile import SmoothBrakeLeader
from env.reward import compute_reward


class SumoEnv:
    def __init__(self, gui=False):
        self.gui = gui
        self.leader_profile = SmoothBrakeLeader(n_events=cfg.LEADER_N_EVENTS)
        self._started = False
        self.t = 0


    # Lifecycle
    def _start_sumo(self):
        if self._started:
            return
        traci.start([
            "sumo-gui" if self.gui else "sumo",
            "-c", cfg.SUMO_CFG,
            "--step-length", str(cfg.STEP_LENGTH),
            "--no-warnings",
            "--no-step-log",
            "--start",
            "--quit-on-end",            
        ])
        self._started = True


    def close(self):
        if self._started:
            traci.close()
        self._started = False
        self.t = 0


    def reset(self):
        if self._started:
            self.close()
        self._start_sumo()
        self.t = 0

        # wait for all vehicles till present
        for _ in range(2000):
            traci.simulationStep()
            if all(v in traci.vehicle.getIDList() for v in cfg.ALL_IDS):
                break

        for aid in cfg.AGENT_IDS:
            traci.vehicle.setSpeedMode(aid, 31)
            traci.vehicle.setMaxSpeed(aid, cfg.V_MAX)
            traci.vehicle.setSpeed(aid, -1)
        
        episode_duration = cfg.MAX_STEPS * cfg.STEP_LENGTH
        self.leader_profile.reset(episode_horizon_duration=episode_duration)

        local_obs, global_obs = self._get_obs()
        return local_obs, global_obs, {}


    def step(self, actions):
        ids = traci.vehicle.getIDList()
        if any(v not in ids for v in cfg.ALL_IDS):
            loc, glo = self._get_obs()
            return loc, glo, np.full(cfg.N_AGENTS, -100.0, dtype=np.float32), True, {"reason": "vehicle_missing"}

        # Leader disturbance
        t_s = self.t * cfg.STEP_LENGTH
        traci.vehicle.setSpeed(cfg.LEADER_ID, self.leader_profile.desired_speed(t_s))

        # Agent actions (delta acceleration)
        for i, aid in enumerate(cfg.AGENT_IDS):
            da = float(np.clip(actions[i] * cfg.DELTA_A_MAX, -cfg.DELTA_A_MAX, cfg.DELTA_A_MAX))
            current_a = traci.vehicle.getAcceleration(aid)
            new_a = float(np.clip(current_a + da, cfg.A_MIN, cfg.A_MAX))
            traci.vehicle.setAcceleration(aid, new_a, cfg.STEP_LENGTH)
        
        traci.simulationStep()
        self.t += 1

        # Collect per agent state for reward computation
        ids = traci.vehicle.getIDList()
        gaps, rel_vs, accels, speeds = self._collect_agent_state(ids)

        # Termination check
        done, reason = self._check_done(gaps, ids)
        rewards = compute_reward(
                        gaps=gaps,
                        rel_velocities=rel_vs,
                        accelerations=accels,
                        speeds=speeds,
                        done=done,
                        termination_reason=reason or ""
                    )
        local_obs, global_obs = self._get_obs()
        return local_obs, global_obs, rewards, done, {
            "reason": reason,
            "t": self.t,
            "gaps": gaps,
            "rel_vs": rel_vs,
            "accels": accels,
            "speeds": speeds,
        }


    # Observations
    def _get_obs(self):
        ids = traci.vehicle.getIDList()
        local_obs = np.zeros((cfg.N_AGENTS, cfg.LOCAL_OBS_DIM), dtype=np.float32)

        for i, aid in enumerate(cfg.AGENT_IDS):
            v_ego = traci.vehicle.getSpeed(aid)
            pos_ego = traci.vehicle.getPosition(aid)[0]

            neighbors_ahead = cfg.ALL_IDS[:i+1][::-1] # closest first
            for j in range(cfg.K_NEIGHBORS):
                if j < len(neighbors_ahead) and neighbors_ahead[j] in ids:
                    nid = neighbors_ahead[j]
                    v_n = traci.vehicle.getSpeed(nid)
                    pos_n = traci.vehicle.getPosition(nid)[0]
                    len_n = traci.vehicle.getLength(nid)
                    gap = pos_n - pos_ego - len_n
                    rel_v = v_n - v_ego
                else:
                    gap, rel_v, v_n = 500.0, 0.0, 0.0
                
                b = j*3
                local_obs[i, b] = np.clip(gap/cfg.GAP_NORM, 0.0, 5.0)
                local_obs[i, b+1] = np.clip(rel_v/cfg.RELV_NORM, -5.0, 5.0)
                local_obs[i, b+2] = np.clip(v_n/cfg.V_NORM, 0.0, 2.0)
        
        return local_obs, local_obs.flatten()

    # Helpers
    def _collect_agent_state(self, ids):
        gaps = np.zeros(cfg.N_AGENTS, dtype=np.float32)
        rel_vs = np.zeros(cfg.N_AGENTS, dtype=np.float32)
        accels = np.zeros(cfg.N_AGENTS, dtype=np.float32)
        speeds = np.zeros(cfg.N_AGENTS, dtype=np.float32)

        for i, aid in enumerate(cfg.AGENT_IDS):
            pred_id = cfg.ALL_IDS[i]
            if aid not in ids or pred_id not in ids:
                gaps[i] = 500.0
                continue
            v = traci.vehicle.getSpeed(aid)
            pos = traci.vehicle.getPosition(aid)[0]
            v_p = traci.vehicle.getSpeed(pred_id)
            pos_p = traci.vehicle.getPosition(pred_id)[0]
            len_p = traci.vehicle.getLength(pred_id)

            gaps[i] = pos_p - pos - len_p
            rel_vs[i] = v_p - v
            accels[i] = traci.vehicle.getAcceleration(aid)
            speeds[i] = v

        return gaps, rel_vs, accels, speeds


    def _check_done(self, gaps, ids):
        if any(v not in ids for v in cfg.ALL_IDS):
            return True, "vehicle_missing"
        if np.any(gaps <= cfg.HARD_MIN_GAP):
            return True, "collision"
        if np.any(gaps >= cfg.RUNAWAY_GAP):
            return True, "runaway"
        if self.t >= cfg.MAX_STEPS:
            return True, "time_up"
        return False, None

