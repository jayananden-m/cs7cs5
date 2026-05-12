import random

def _clamp(x, low, high): # clips a value to stay within [low, high]
    return low if x < low else high if x > high else x


def _lerp(a, b, u): # linear interpolation btw a and b
    u = _clamp(u, 0.0, 1.0)
    return a + (b - a) * u


class SmoothBrakeLeader:
    """
        Generates a piecewise smooth speed profile for the leader vehicle.

        Each braking event has three phases:
            ramp_down: cruise speed → brake speed  (smooth deceleration)
            hold:      constant brake speed
            ramp_up:   brake speed → cruise speed  (smooth re-acceleration)

        Events are placed after a warm-up period so agents have time to reach
        steady-state formation before the first disturbance.
    """
    def __init__(
            self,
            cruise_speed = 28.0,
            brake_speed = 15.0,
            ramp_down_duration = 4.0,
            hold_duration = 5.0,
            ramp_up_duration = 4.0,
            n_events = 1,
            earliest_start_duration = 8.0,
            min_gap_duration = 12.0,
            warmup_duration = 20.0
        ):
        self.cruise_speed = cruise_speed
        self.brake_speed = brake_speed
        self.ramp_down_duration = ramp_down_duration
        self.hold_duration = hold_duration
        self.ramp_up_duration = ramp_up_duration
        self.n_events = n_events
        self.earliest_start_duration = earliest_start_duration
        self.min_gap_duration = min_gap_duration
        self.warmup_duration = warmup_duration
        self._rng = random.Random()
        self._events = []


    def reset(self, episode_horizon_duration, seed=None):
        # Schedule braking events for new episode
        if seed is not None:
            self._rng = random.Random(seed)
        self._events = []
        event_duration = self.ramp_down_duration + self.hold_duration + self.ramp_up_duration

        t = self.warmup_duration + self.earliest_start_duration
        for _ in range(self.n_events):
            latest = episode_horizon_duration - event_duration - 1.0
            if t > latest:
                break
            self._events.append((t, t + event_duration))
            t += event_duration + self.min_gap_duration


    def desired_speed(self, t_s):
        # Return desired leader speed at time t_s
        
        if t_s < self.warmup_duration: # Warm-up: ramp from 0 to cruise
            return _lerp(0.0, self.cruise_speed, t_s / max(self.warmup_duration, 1e-6))
        
        for (start, end) in self._events:
            if t_s < start or t_s > end:
                continue
            tau = t_s - start
            if tau <= self.ramp_down_duration:
                return _lerp(self.cruise_speed, self.brake_speed, tau/self.ramp_down_duration)
            
            tau -= self.ramp_down_duration
            if tau <= self.hold_duration:
                return self.brake_speed
            
            tau -= self.hold_duration
            return _lerp(self.brake_speed, self.cruise_speed, tau/self.ramp_up_duration)
        
        return self.cruise_speed
