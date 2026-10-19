"""Neuromuscular and passive-tissue model of the dog.

The dog model is driven by 38 joint-level torque generators (one per
functional muscle group; the lumbar, cervical and caudal groups act through
fixed tendons that couple several vertebral joints).  Each generator is
treated as a lumped Hill-type muscle group:

* **activation dynamics** - the excitation computed here is low-pass
  filtered by the actuator's first-order activation filter (the model's
  ``dynprm``), re-tuned to ``activation_time_ms``;
* **postural tone** - a baseline activation that produces short-range
  stiffness/damping about the posture held at the start of the event;
* **startle / bracing reflex** - after ``reflex_delay_ms`` from the onset
  of the vehicle deceleration the activation rises to
  ``reflex_activation`` (co-contraction), stiffening the joints;
* **stretch reflex** - a spinal reflex with ``stretch_reflex_delay_ms``
  latency that acts on the delayed joint state;
* **force-velocity** - the available torque drops when a muscle group is
  lengthened or shortened quickly (Hill hyperbola, ``v_max`` = 10 rad/s),
  with eccentric enhancement capped at 1.4;
* **postural balance** - the dog's active balance (which in reality comes
  from coordinated limb and trunk muscles) is lumped into a torque on the
  thorax that keeps it near its starting orientation.  It is limited to
  ``0.3 M g h`` - enough to stop a sitting dog from slowly toppling on a
  soft, sloped cushion, but two orders of magnitude below crash loads;
* **passive tissue** - ligaments and joint capsules generate an
  exponentially stiffening torque in the last 12 % of each joint range,
  in addition to MuJoCo's hard joint limits.  The tissue torque relative to
  the joint's muscle capacity is reported as a ligament-strain measure.
"""

from __future__ import annotations

from collections import deque

import mujoco
import numpy as np

from ..config import NeuromuscularConfig


class NeuromuscularController:
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, cfg: NeuromuscularConfig, dog_actuators, dog_joints,
                 torque_scale: float, dt: float):
        self.m = m
        self.cfg = cfg
        self.act = np.asarray(dog_actuators, int)
        self.gain = m.actuator_gainprm[self.act, 0].copy()
        self.dt = dt
        # posture targets (actuator lengths) are set by capture_targets()
        self.L_target = d.actuator_length[self.act].copy()
        theta_sat = 0.35  # rad of error that saturates a fully activated group
        self.kp = self.gain / theta_sat
        self.kd = self.kp * 0.04
        # activation filter time constant
        tau = max(cfg.activation_time_ms * 1e-3, 1e-3)
        for a in self.act:
            if m.actuator_dyntype[a] == mujoco.mjtDyn.mjDYN_FILTER:
                m.actuator_dynprm[a, 0] = tau
        n_delay = max(1, int(round(cfg.stretch_reflex_delay_ms * 1e-3 / dt)))
        self.hist = deque(maxlen=n_delay)
        self.tone = float(cfg.muscle_tone)
        self.onset = None
        self.activation = self.tone
        # passive tissue
        self.joints = np.asarray(dog_joints, int)
        self.qadr = m.jnt_qposadr[self.joints]
        self.dadr = m.jnt_dofadr[self.joints]
        lo = m.jnt_range[self.joints, 0]
        hi = m.jnt_range[self.joints, 1]
        span = hi - lo
        self.lig_lo = lo + 0.12 * span
        self.lig_hi = hi - 0.12 * span
        self.lig_beta = 1.0 / np.maximum(0.04 * span, 0.01)
        self.lig_k = 0.4 * torque_scale  # N m at one e-fold
        self.lig_cap = 400.0 * torque_scale
        self.ligament_torque = np.zeros(len(self.joints))

    def setup_balance(self, torso: int, body_mass: float, com_height: float, inertia: float) -> None:
        g = 9.81
        self.torso = torso
        self.bal_k = 2.0 * body_mass * g * com_height
        self.bal_c = 2.0 * np.sqrt(self.bal_k * inertia)
        self.bal_max = 0.3 * body_mass * g * com_height
        self.R_ref = None

    def capture_targets(self, d: mujoco.MjData) -> None:
        self.L_target = d.actuator_length[self.act].copy()
        if hasattr(self, "torso"):
            self.R_ref = d.xmat[self.torso].reshape(3, 3).copy()

    def balance(self, d: mujoco.MjData, gain: float) -> None:
        if getattr(self, "R_ref", None) is None or gain <= 0:
            return
        R = d.xmat[self.torso].reshape(3, 3)
        Rerr = R @ self.R_ref.T
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, Rerr.flatten())
        if q[0] < 0:
            q = -q
        ang = 2.0 * np.arctan2(np.linalg.norm(q[1:]), q[0])
        axis = q[1:] / max(np.linalg.norm(q[1:]), 1e-12)
        tau = -gain * (self.bal_k * ang * axis + self.bal_c * d.cvel[self.torso, :3])
        n = np.linalg.norm(tau)
        lim = gain * self.bal_max
        if n > lim:
            tau *= lim / n
        d.xfrc_applied[self.torso, 3:] += tau

    def set_tone(self, tone: float) -> None:
        self.tone = float(tone)

    def trigger(self, t: float) -> None:
        if self.onset is None:
            self.onset = t

    def activation_level(self, t: float) -> float:
        a = self.tone
        c = self.cfg
        if c.startle_reflex and self.onset is not None:
            t_r = self.onset + c.reflex_delay_ms * 1e-3
            if t > t_r:
                tau = max(c.activation_time_ms * 1e-3, 1e-3)
                a = self.tone + (max(c.reflex_activation, self.tone) - self.tone) * (1.0 - np.exp(-(t - t_r) / tau))
        self.activation = a
        return a

    def step(self, t: float, d: mujoco.MjData) -> None:
        L = d.actuator_length[self.act]
        Ld = d.actuator_velocity[self.act]
        self.hist.append((L.copy(), Ld.copy()))
        Ldel, Lddel = self.hist[0]
        a = self.activation_level(t)
        tau = a * (self.kp * (self.L_target - L) - self.kd * Ld)
        tau += self.cfg.stretch_reflex_gain * a * (self.kp * (self.L_target - Ldel) - self.kd * Lddel)
        # Hill force-velocity: shortening in the direction of the torque weakens it, lengthening enhances it
        v = -np.sign(tau) * Ld  # >0 when the muscle group shortens
        vmax = 10.0
        fv = np.where(v >= 0, (1 - v / vmax) / (1 + 4 * v / vmax), 1.4 - 0.4 * np.exp(np.minimum(v, 0) * 8 / vmax))
        tau *= np.clip(fv, 0.0, 1.4)
        d.ctrl[self.act] = np.clip(tau / self.gain, -1.0, 1.0)

    def passive_tissue(self, d: mujoco.MjData) -> np.ndarray:
        """Ligament/capsule torques near the joint limits; adds them to qfrc_applied."""
        q = d.qpos[self.qadr]
        over_hi = np.maximum(q - self.lig_hi, 0.0)
        over_lo = np.maximum(self.lig_lo - q, 0.0)
        t_hi = self.lig_k * (np.exp(np.minimum(self.lig_beta * over_hi, 8.0)) - 1.0)
        t_lo = self.lig_k * (np.exp(np.minimum(self.lig_beta * over_lo, 8.0)) - 1.0)
        tau = np.clip(t_lo - t_hi, -self.lig_cap, self.lig_cap)
        d.qfrc_applied[self.dadr] += tau
        self.ligament_torque = tau
        return tau
