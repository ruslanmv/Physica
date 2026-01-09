from __future__ import annotations

"""Physics engine.

This module provides a small but production-friendly ballistic simulator.

Key improvements vs the prototype:
- Structured results (time vector + state + metadata)
- Accurate ground impact using solver events (SciPy) and interpolation fallback
- Input validation and clear error messages
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

Backend = Literal["numpy", "jax"]


def _try_import_jax() -> bool:
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
        from diffrax import (  # noqa: F401
            Dopri5,
            ODETerm,
            PIDController,
            SaveAt,
            diffeqsolve,
        )

        return True
    except Exception:
        return False


_JAX_AVAILABLE = _try_import_jax()


@dataclass(frozen=True)
class TrajectoryResult:
    """Result of a ballistic simulation.

    Attributes
    ----------
    t:
        Time samples (shape: [N]).
    y:
        State samples (shape: [N, 4]) with columns [x, y, vx, vy].
    backend:
        Backend actually used.
    success:
        Whether the solver succeeded.
    message:
        Solver status message.
    meta:
        Backend-specific metadata.
    impact:
        Ground-impact information (None if not reached within max_time).
    """

    t: np.ndarray
    y: np.ndarray
    backend: Backend
    success: bool
    message: str
    meta: Dict[str, Any]
    impact: Optional[Dict[str, Any]]

    @property
    def x(self) -> np.ndarray:
        return self.y[:, 0]

    @property
    def y_pos(self) -> np.ndarray:
        return self.y[:, 1]

    @property
    def vx(self) -> np.ndarray:
        return self.y[:, 2]

    @property
    def vy(self) -> np.ndarray:
        return self.y[:, 3]


def _validate_inputs(
    v0: float, angle_deg: float, max_time: float, steps: int
) -> Tuple[float, float, float, int]:
    try:
        v0f = float(v0)
        ang = float(angle_deg)
        mt = float(max_time)
        st = int(steps)
    except Exception as e:
        raise TypeError("v0, angle_deg, max_time must be numbers and steps must be int-like") from e

    if not np.isfinite(v0f) or v0f < 0:
        raise ValueError("v0 must be a finite non-negative number")
    if not np.isfinite(ang):
        raise ValueError("angle_deg must be finite")
    if not np.isfinite(mt) or mt <= 0:
        raise ValueError("max_time must be a finite positive number")
    if st < 2:
        raise ValueError("steps must be >= 2")
    return v0f, ang, mt, st


@dataclass
class BallisticSimulator:
    """Projectile motion with gravity and quadratic drag.

    State vector: [x, y, vx, vy]

    Drag model (quadratic):
        a_drag = -c * |v| * v
    """

    gravity: float = 9.81
    # Default tuned for a reasonable demo range; set to 0.0 for ideal ballistic motion.
    drag_coeff: float = 0.005
    backend: Backend = "numpy"

    def __post_init__(self) -> None:
        if self.gravity <= 0:
            raise ValueError("gravity must be > 0")
        if self.drag_coeff < 0:
            raise ValueError("drag_coeff must be >= 0")
        if self.backend == "jax" and not _JAX_AVAILABLE:
            raise RuntimeError(
                "backend='jax' requested but jax/diffrax are not installed. "
                "Install extras: pip install -e '.[jax]'"
            )

    # -------------------------
    # NumPy/SciPy backend
    # -------------------------
    def _vf_numpy(self, _t: float, y: np.ndarray) -> np.ndarray:
        _x, _y, vx, vy = y
        vmag = float(np.sqrt(vx * vx + vy * vy))
        ax = -self.drag_coeff * vx * vmag
        ay = -self.gravity - self.drag_coeff * vy * vmag
        return np.array([vx, vy, ax, ay], dtype=float)

    def _simulate_numpy(
        self, v0: float, angle_deg: float, max_time: float, steps: int
    ) -> TrajectoryResult:
        from scipy.integrate import solve_ivp

        ang = np.deg2rad(angle_deg)
        vx0 = float(v0 * np.cos(ang))
        vy0 = float(v0 * np.sin(ang))
        y0 = np.array([0.0, 0.0, vx0, vy0], dtype=float)

        # Time grid (we'll be truncated if an event happens)
        ts = np.linspace(0.0, float(max_time), int(steps))

        # Stop when we hit the ground on the way down.
        def hit_ground(_t: float, yy: np.ndarray) -> float:
            return float(yy[1])

        hit_ground.terminal = True  # type: ignore[attr-defined]
        hit_ground.direction = -1.0  # type: ignore[attr-defined]

        sol = solve_ivp(
            fun=lambda t, y: self._vf_numpy(t, y),
            t_span=(0.0, float(max_time)),
            y0=y0,
            t_eval=ts,
            events=hit_ground,
            dense_output=True,
            rtol=1e-7,
            atol=1e-9,
        )

        if not sol.success:
            raise RuntimeError(f"ODE solve failed: {sol.message}")

        traj = sol.y.T  # (N, 4)
        t_out = sol.t

        impact: Optional[Dict[str, Any]] = None
        if sol.t_events and len(sol.t_events[0]) > 0:
            t_imp = float(sol.t_events[0][0])
            y_imp = sol.sol(t_imp) if sol.sol is not None else sol.y_events[0][0]
            y_imp = np.asarray(y_imp, dtype=float)
            impact = {
                "t": t_imp,
                "state": y_imp,
                "x": float(y_imp[0]),
                "y": float(y_imp[1]),
            }

        return TrajectoryResult(
            t=np.asarray(t_out, dtype=float),
            y=np.asarray(traj, dtype=float),
            backend="numpy",
            success=True,
            message=str(sol.message),
            meta={"nfev": sol.nfev, "njev": sol.njev, "status": sol.status},
            impact=impact,
        )

    # -------------------------
    # JAX/Diffrax backend (optional)
    # -------------------------
    def _simulate_jax(
        self, v0: float, angle_deg: float, max_time: float, steps: int
    ) -> TrajectoryResult:
        import jax.numpy as jnp
        from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve

        def vf(_t, y, _args):
            _x, _y, vx, vy = y
            vmag = jnp.sqrt(vx**2 + vy**2)
            ax = -self.drag_coeff * vx * vmag
            ay = -self.gravity - self.drag_coeff * vy * vmag
            return jnp.stack([vx, vy, ax, ay])

        ang = jnp.deg2rad(angle_deg)
        vx0 = v0 * jnp.cos(ang)
        vy0 = v0 * jnp.sin(ang)
        y0 = jnp.array([0.0, 0.0, vx0, vy0])

        term = ODETerm(vf)
        solver = Dopri5()
        t_eval = jnp.linspace(0.0, max_time, steps)
        saveat = SaveAt(ts=t_eval)
        step_controller = PIDController(rtol=1e-6, atol=1e-8)

        sol = diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=max_time,
            dt0=0.01,
            y0=y0,
            saveat=saveat,
            stepsize_controller=step_controller,
        )

        ys = np.asarray(sol.ys)
        ts = np.asarray(t_eval)

        # Diffrax event handling is more involved; for production we provide
        # a robust interpolation fallback for ground impact.
        impact = _estimate_ground_impact_from_samples(ts, ys)

        return TrajectoryResult(
            t=ts,
            y=ys,
            backend="jax",
            success=True,
            message="ok",
            meta={"stats": getattr(sol, "stats", None)},
            impact=impact,
        )

    # -------------------------
    # Public API
    # -------------------------
    def simulate(
        self,
        v0: float,
        angle_deg: float,
        max_time: float = 10.0,
        steps: int = 200,
        backend: Optional[Backend] = None,
    ) -> TrajectoryResult:
        """Simulate the trajectory.

        Returns
        -------
        TrajectoryResult
            Structured result containing time samples, state samples, and
            (when available) precise ground impact.
        """

        v0f, ang, mt, st = _validate_inputs(
            v0=v0, angle_deg=angle_deg, max_time=max_time, steps=steps
        )
        be = backend or self.backend
        if be == "jax":
            return self._simulate_jax(v0=v0f, angle_deg=ang, max_time=mt, steps=st)
        return self._simulate_numpy(v0=v0f, angle_deg=ang, max_time=mt, steps=st)


def _estimate_ground_impact_from_samples(
    t: np.ndarray, traj: np.ndarray
) -> Optional[Dict[str, Any]]:
    """Estimate ground impact from discrete samples.

    Used as a backend-agnostic fallback (and for the JAX path).
    """

    y = traj[:, 1]
    x = traj[:, 0]
    for i in range(1, len(y)):
        if y[i] <= 0.0 and y[i - 1] > 0.0:
            y0, y1 = float(y[i - 1]), float(y[i])
            x0, x1 = float(x[i - 1]), float(x[i])
            t0, t1 = float(t[i - 1]), float(t[i])
            alpha = 1.0 if y1 == y0 else (0.0 - y0) / (y1 - y0)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            x_imp = x0 + alpha * (x1 - x0)
            t_imp = t0 + alpha * (t1 - t0)
            state_imp = traj[i - 1] + alpha * (traj[i] - traj[i - 1])
            return {
                "t": t_imp,
                "state": np.asarray(state_imp, dtype=float),
                "x": float(x_imp),
                "y": 0.0,
            }
    return None
