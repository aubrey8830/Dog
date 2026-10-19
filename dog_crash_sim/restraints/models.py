"""Constitutive models of the restraint components.

All models are explicit: they read the current geometric state from MuJoCo
(tendon length/velocity, joint angle/velocity), update their internal state
and return a force or moment that the simulation applies through an
actuator or ``qfrc_applied``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Webbing:
    """Tension-only webbing with loading/unloading hysteresis and breakage.

    ``axial_stiffness`` is the force per unit strain (N), i.e. rated load
    divided by the elongation fraction at that load.  Unloading follows a
    slope ``unload_factor`` times steeper than loading from the largest
    strain reached, leaving a permanent set like real polyamide/polyester
    webbing.
    """
    name: str
    free_length: float                 # m, unstretched length of the strap between its end points
    axial_stiffness: float             # N per unit strain
    strength: float = math.inf         # N, breaking load
    damping_ratio: float = 0.08
    unload_factor: float = 3.0
    mass_for_damping: float = 5.0      # kg, effective mass used to size the damping coefficient
    broken: bool = False
    broken_at: Optional[float] = None
    eps_max: float = 0.0
    tension: float = 0.0
    peak_tension: float = 0.0

    def stiffness(self) -> float:
        return self.axial_stiffness / max(self.free_length, 1e-3)

    def force(self, length: float, velocity: float, t: float) -> float:
        if self.broken:
            self.tension = 0.0
            return 0.0
        L0 = max(self.free_length, 1e-3)
        eps = (length - L0) / L0
        if eps <= 0.0 and self.eps_max <= 0.0:
            self.tension = 0.0
            return 0.0
        k = self.axial_stiffness
        if eps >= self.eps_max:
            T = k * eps
            self.eps_max = eps
        else:
            T = k * self.eps_max - self.unload_factor * k * (self.eps_max - eps)
        if T <= 0.0:
            self.tension = 0.0
            return 0.0
        c = 2.0 * self.damping_ratio * math.sqrt(self.stiffness() * self.mass_for_damping)
        T = max(0.0, T + c * velocity)
        if T > self.strength:
            self.broken = True
            self.broken_at = t
            self.tension = 0.0
            return 0.0
        self.tension = T
        self.peak_tension = max(self.peak_tension, T)
        return T

    def reset_set(self) -> None:
        """Forget the stretch history (used after the settling phase)."""
        self.eps_max = 0.0
        self.peak_tension = 0.0


@dataclass
class EnergyAbsorber:
    """Tear webbing / load limiter: pays out at constant force until the stroke is used."""
    force: float
    stroke: float
    payout: float = 0.0

    def limit(self, web: Webbing, length: float, velocity: float, t: float) -> float:
        """Adjust the webbing free length so that the tension does not exceed the limit force."""
        if self.force is None or self.force <= 0 or self.payout >= self.stroke:
            return web.force(length, velocity, t)
        trial = Webbing(web.name, web.free_length, web.axial_stiffness, math.inf, web.damping_ratio, web.unload_factor,
                        web.mass_for_damping)
        trial.eps_max = web.eps_max
        T = trial.force(length, velocity, t)
        if T > self.force:
            # elastic strain that carries the limit force
            eps_lim = self.force / web.axial_stiffness
            new_L0 = length / (1.0 + eps_lim)
            dL = min(max(new_L0 - web.free_length, 0.0), self.stroke - self.payout)
            if dL > 0:
                web.free_length += dL
                self.payout += dL
                web.eps_max = min(web.eps_max, eps_lim)
        return web.force(length, velocity, t)


@dataclass
class Retractor:
    """Vehicle belt retractor: locking, film spool, pretensioner and load limiter."""
    lock_g: float
    film_spool: float                        # m
    pretensioner_stroke: float = 0.0         # m
    pretensioner_force: float = 0.0          # N
    pretensioner_speed: float = 12.0         # m/s
    pretensioner_fire_time: Optional[float] = None
    load_limit: Optional[float] = None       # N
    load_limit_stroke: float = 0.0
    rewind_tension: float = 6.0              # N, comfort spring
    locked: bool = False
    lock_time: Optional[float] = None
    retracted: float = 0.0
    paid_out: float = 0.0
    film_used: float = 0.0
    fired: bool = False

    def update(self, web: Webbing, length: float, velocity: float, t: float, vehicle_accel_g: float, dt: float) -> float:
        """Advance the retractor and return the belt tension."""
        if not self.locked and vehicle_accel_g >= self.lock_g:
            self.locked = True
            self.lock_time = t
        if not self.locked:
            # the spool follows the webbing; only the rewind spring acts
            web.free_length = max(length - self.rewind_tension / web.axial_stiffness * length, 1e-3)
            web.eps_max = 0.0
            web.force(length, velocity, t)
            web.tension = self.rewind_tension
            return self.rewind_tension
        # pretensioner retracts webbing while its stroke and stall force allow
        if self.pretensioner_fire_time is not None and t >= self.pretensioner_fire_time and \
                self.retracted < self.pretensioner_stroke:
            self.fired = True
            if web.tension < self.pretensioner_force:
                d = min(self.pretensioner_speed * dt, self.pretensioner_stroke - self.retracted,
                        max(web.free_length - 0.05, 0.0))
                web.free_length -= d
                self.retracted += d
        # film spool: extra webbing pulled out of the locked spool as the load builds
        if self.film_used < self.film_spool and web.tension > 50.0:
            d = min(self.film_spool - self.film_used, self.film_spool * min(web.tension / 2000.0, 1.0) - self.film_used)
            if d > 0:
                web.free_length += d
                self.film_used += d
        if self.load_limit is not None and self.load_limit > 0:
            ea = EnergyAbsorber(self.load_limit, self.load_limit_stroke, self.paid_out)
            T = ea.limit(web, length, velocity, t)
            self.paid_out = ea.payout
            return T
        return web.force(length, velocity, t)


@dataclass
class SeatBack:
    """Elasto-plastic folding seat back with a breakable latch.

    Positive angle = top moving forward.  The returned moment acts on the
    hinge joint; ``theta_p`` is the plastic rotation.
    """
    stiffness: float          # N m / rad
    forward_yield: float      # N m
    rearward_yield: float     # N m
    latch_strength: float     # N m
    damping: float            # N m s / rad
    theta_p: float = 0.0
    latch_broken: bool = False
    latch_broken_at: Optional[float] = None
    moment: float = 0.0
    peak_moment: float = 0.0

    def update(self, theta: float, omega: float, t: float) -> float:
        if self.latch_broken:
            self.moment = -0.2 * self.damping * omega
            return self.moment
        M = -self.stiffness * (theta - self.theta_p)
        if -M > self.latch_strength:          # forward demand beyond the latch
            self.latch_broken = True
            self.latch_broken_at = t
            self.moment = 0.0
            return 0.0
        if -M > self.forward_yield:
            self.theta_p = theta - self.forward_yield / self.stiffness
            M = -self.forward_yield
        elif M > self.rearward_yield:
            self.theta_p = theta + self.rearward_yield / self.stiffness
            M = self.rearward_yield
        self.moment = M - self.damping * omega
        self.peak_moment = max(self.peak_moment, abs(M))
        return self.moment


@dataclass
class AirbagState:
    """Deployment kinematics and pressure of one airbag."""
    name: str
    fire_time: Optional[float]
    inflation_time: float
    vent_tau: float
    peak_pressure: float
    stroke: float               # m of travel from stowed to deployed
    deployed_fraction: float = 0.0
    pressure: float = 0.0

    def kinematics(self, t: float):
        """(position, velocity, acceleration) along the deployment axis, plus pressure."""
        if self.fire_time is None or t < self.fire_time:
            self.deployed_fraction = 0.0
            self.pressure = 0.0
            return 0.0, 0.0, 0.0
        tau = (t - self.fire_time) / self.inflation_time
        if tau < 1.0:
            # smooth-step deployment
            s = 3 * tau ** 2 - 2 * tau ** 3
            sd = (6 * tau - 6 * tau ** 2) / self.inflation_time
            sdd = (6 - 12 * tau) / self.inflation_time ** 2
            self.pressure = self.peak_pressure * s
        else:
            s, sd, sdd = 1.0, 0.0, 0.0
            self.pressure = self.peak_pressure * math.exp(-(t - self.fire_time - self.inflation_time) / self.vent_tau)
        self.deployed_fraction = s
        return self.stroke * s, self.stroke * sd, self.stroke * sdd


def first_crossing(t: np.ndarray, signal: np.ndarray, threshold: float, t_min: float = -math.inf) -> Optional[float]:
    idx = np.nonzero((signal >= threshold) & (t >= t_min))[0]
    return float(t[idx[0]]) if len(idx) else None
