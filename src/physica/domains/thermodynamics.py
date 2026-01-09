"""Thermodynamics domain.

Implements thermodynamic state equations, entropy calculations,
thermodynamic cycles, heat engines, and phase transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.constants import R, k as k_B
from scipy.optimize import fsolve


@dataclass
class ThermodynamicState:
    """Represents a thermodynamic state.

    Attributes
    ----------
    P:
        Pressure (Pa).
    V:
        Volume (m³).
    T:
        Temperature (K).
    n:
        Amount of substance (mol).
    """

    P: float
    V: float
    T: float
    n: float = 1.0

    def internal_energy(self, cv: float) -> float:
        """Calculate internal energy for ideal gas.

        Parameters
        ----------
        cv:
            Molar heat capacity at constant volume (J/(mol·K)).

        Returns
        -------
        U:
            Internal energy (J).
        """
        return self.n * cv * self.T

    def enthalpy(self, cp: float) -> float:
        """Calculate enthalpy.

        Parameters
        ----------
        cp:
            Molar heat capacity at constant pressure (J/(mol·K)).

        Returns
        -------
        H:
            Enthalpy (J).
        """
        return self.n * cp * self.T

    def entropy_ideal_gas(self, s0: float = 0.0) -> float:
        """Calculate entropy for ideal gas (relative to reference state).

        Parameters
        ----------
        s0:
            Reference entropy (J/K).

        Returns
        -------
        S:
            Entropy (J/K).
        """
        # For ideal gas: dS = nR(dV/V) + nCv(dT/T)
        # Simplified: S = S0 + nR*ln(V) + nCv*ln(T)
        return s0 + self.n * R * np.log(self.V) + self.n * (5 / 2) * R * np.log(self.T)


class StateEquation:
    """Base class for equations of state."""

    def pressure(self, V: float, T: float, n: float = 1.0) -> float:
        """Calculate pressure given V, T, n.

        Parameters
        ----------
        V:
            Volume (m³).
        T:
            Temperature (K).
        n:
            Amount of substance (mol).

        Returns
        -------
        P:
            Pressure (Pa).
        """
        raise NotImplementedError


class IdealGasEOS(StateEquation):
    """Ideal gas equation of state: PV = nRT."""

    def pressure(self, V: float, T: float, n: float = 1.0) -> float:
        """Calculate pressure using ideal gas law."""
        return n * R * T / V

    def volume(self, P: float, T: float, n: float = 1.0) -> float:
        """Calculate volume given P, T, n."""
        return n * R * T / P

    def temperature(self, P: float, V: float, n: float = 1.0) -> float:
        """Calculate temperature given P, V, n."""
        return P * V / (n * R)


class VanDerWaalsEOS(StateEquation):
    """Van der Waals equation: (P + a*n²/V²)(V - nb) = nRT.

    Accounts for intermolecular forces (a) and molecular volume (b).
    """

    def __init__(self, a: float, b: float):
        """Initialize Van der Waals equation.

        Parameters
        ----------
        a:
            Attraction parameter (Pa·m⁶/mol²).
        b:
            Excluded volume parameter (m³/mol).
        """
        self.a = a
        self.b = b

    def pressure(self, V: float, T: float, n: float = 1.0) -> float:
        """Calculate pressure using Van der Waals equation."""
        return (n * R * T) / (V - n * self.b) - self.a * (n / V) ** 2

    def critical_point(self) -> Tuple[float, float, float]:
        """Calculate critical point (Tc, Pc, Vc) for n=1 mol.

        Returns
        -------
        Tc, Pc, Vc:
            Critical temperature (K), pressure (Pa), volume (m³/mol).
        """
        Tc = 8 * self.a / (27 * R * self.b)
        Pc = self.a / (27 * self.b**2)
        Vc = 3 * self.b
        return Tc, Pc, Vc


class ThermodynamicProcess:
    """Base class for thermodynamic processes."""

    def __init__(self, eos: StateEquation):
        """Initialize process.

        Parameters
        ----------
        eos:
            Equation of state.
        """
        self.eos = eos

    def work(self, state1: ThermodynamicState, state2: ThermodynamicState) -> float:
        """Calculate work done by the system.

        Parameters
        ----------
        state1:
            Initial state.
        state2:
            Final state.

        Returns
        -------
        W:
            Work (J).
        """
        raise NotImplementedError

    def heat(
        self, state1: ThermodynamicState, state2: ThermodynamicState, cv: float
    ) -> float:
        """Calculate heat transfer (First Law: Q = ΔU + W).

        Parameters
        ----------
        state1:
            Initial state.
        state2:
            Final state.
        cv:
            Molar heat capacity at constant volume (J/(mol·K)).

        Returns
        -------
        Q:
            Heat transfer (J).
        """
        delta_U = state2.internal_energy(cv) - state1.internal_energy(cv)
        W = self.work(state1, state2)
        return delta_U + W


class IsothermalProcess(ThermodynamicProcess):
    """Isothermal process (constant temperature)."""

    def work(self, state1: ThermodynamicState, state2: ThermodynamicState) -> float:
        """Calculate work for isothermal expansion/compression."""
        # W = nRT*ln(V2/V1) for ideal gas
        return state1.n * R * state1.T * np.log(state2.V / state1.V)


class IsobaricProcess(ThermodynamicProcess):
    """Isobaric process (constant pressure)."""

    def work(self, state1: ThermodynamicState, state2: ThermodynamicState) -> float:
        """Calculate work for isobaric process."""
        # W = P(V2 - V1)
        return state1.P * (state2.V - state1.V)


class IsochoricProcess(ThermodynamicProcess):
    """Isochoric process (constant volume)."""

    def work(self, state1: ThermodynamicState, state2: ThermodynamicState) -> float:
        """Calculate work for isochoric process."""
        # W = 0 (no volume change)
        return 0.0


class AdiabaticProcess(ThermodynamicProcess):
    """Adiabatic process (no heat transfer, Q=0)."""

    def __init__(self, eos: StateEquation, gamma: float):
        """Initialize adiabatic process.

        Parameters
        ----------
        eos:
            Equation of state.
        gamma:
            Heat capacity ratio Cp/Cv.
        """
        super().__init__(eos)
        self.gamma = gamma

    def work(self, state1: ThermodynamicState, state2: ThermodynamicState) -> float:
        """Calculate work for adiabatic process."""
        # W = (P1*V1 - P2*V2) / (gamma - 1)
        return (state1.P * state1.V - state2.P * state2.V) / (self.gamma - 1)

    def final_state(
        self, state1: ThermodynamicState, V2: float
    ) -> ThermodynamicState:
        """Calculate final state after adiabatic process.

        Uses relation: P1*V1^gamma = P2*V2^gamma
        """
        P2 = state1.P * (state1.V / V2) ** self.gamma
        T2 = P2 * V2 / (state1.n * R)
        return ThermodynamicState(P=P2, V=V2, T=T2, n=state1.n)


class ThermodynamicCycle:
    """Base class for thermodynamic cycles (heat engines)."""

    def __init__(self):
        """Initialize cycle."""
        self.states = []
        self.processes = []

    def efficiency(self) -> float:
        """Calculate thermal efficiency η = W_net / Q_in.

        Returns
        -------
        eta:
            Thermal efficiency (dimensionless).
        """
        raise NotImplementedError

    def coefficient_of_performance(self) -> float:
        """Calculate COP for refrigeration/heat pump cycles.

        Returns
        -------
        COP:
            Coefficient of performance.
        """
        raise NotImplementedError


class CarnotCycle(ThermodynamicCycle):
    """Carnot cycle (maximum efficiency reversible cycle).

    Process sequence:
    1 → 2: Isothermal expansion at Th
    2 → 3: Adiabatic expansion
    3 → 4: Isothermal compression at Tc
    4 → 1: Adiabatic compression
    """

    def __init__(self, Th: float, Tc: float, V1: float, V2: float, gamma: float = 1.4):
        """Initialize Carnot cycle.

        Parameters
        ----------
        Th:
            Hot reservoir temperature (K).
        Tc:
            Cold reservoir temperature (K).
        V1:
            Initial volume (m³).
        V2:
            Volume after isothermal expansion (m³).
        gamma:
            Heat capacity ratio.
        """
        super().__init__()
        self.Th = Th
        self.Tc = Tc
        self.gamma = gamma

        # Build cycle states (assuming n=1 mol ideal gas)
        eos = IdealGasEOS()
        n = 1.0

        # State 1: (V1, Th)
        P1 = eos.pressure(V1, Th, n)
        state1 = ThermodynamicState(P=P1, V=V1, T=Th, n=n)

        # State 2: (V2, Th) - isothermal expansion
        P2 = eos.pressure(V2, Th, n)
        state2 = ThermodynamicState(P=P2, V=V2, T=Th, n=n)

        # State 3: Adiabatic expansion to Tc
        # Use T2*V2^(gamma-1) = T3*V3^(gamma-1)
        V3 = V2 * (Th / Tc) ** (1 / (gamma - 1))
        P3 = eos.pressure(V3, Tc, n)
        state3 = ThermodynamicState(P=P3, V=V3, T=Tc, n=n)

        # State 4: Isothermal compression at Tc back to cycle closure
        V4 = V1 * (Th / Tc) ** (1 / (gamma - 1))
        P4 = eos.pressure(V4, Tc, n)
        state4 = ThermodynamicState(P=P4, V=V4, T=Tc, n=n)

        self.states = [state1, state2, state3, state4]
        self.processes = [
            IsothermalProcess(eos),
            AdiabaticProcess(eos, gamma),
            IsothermalProcess(eos),
            AdiabaticProcess(eos, gamma),
        ]

    def efficiency(self) -> float:
        """Calculate Carnot efficiency η = 1 - Tc/Th."""
        return 1 - self.Tc / self.Th

    def work_output(self, cv: float = 3 / 2 * R) -> float:
        """Calculate net work output per cycle.

        Parameters
        ----------
        cv:
            Molar heat capacity at constant volume.

        Returns
        -------
        W_net:
            Net work output (J).
        """
        W_net = 0.0
        for i, process in enumerate(self.processes):
            state1 = self.states[i]
            state2 = self.states[(i + 1) % 4]
            W_net += process.work(state1, state2)
        return W_net


class OttoCycle(ThermodynamicCycle):
    """Otto cycle (spark-ignition internal combustion engine).

    Process sequence:
    1 → 2: Adiabatic compression
    2 → 3: Isochoric heat addition (ignition)
    3 → 4: Adiabatic expansion (power stroke)
    4 → 1: Isochoric heat rejection (exhaust)
    """

    def __init__(
        self, compression_ratio: float, gamma: float = 1.4, V1: float = 1e-3
    ):
        """Initialize Otto cycle.

        Parameters
        ----------
        compression_ratio:
            r = V1/V2.
        gamma:
            Heat capacity ratio.
        V1:
            Maximum volume (m³).
        """
        super().__init__()
        self.r = compression_ratio
        self.gamma = gamma
        self.V1 = V1
        self.V2 = V1 / compression_ratio

    def efficiency(self) -> float:
        """Calculate Otto cycle efficiency η = 1 - 1/r^(gamma-1)."""
        return 1 - 1 / (self.r ** (self.gamma - 1))


class EntropyCalculator:
    """Calculate entropy changes and verify Second Law."""

    @staticmethod
    def entropy_change_ideal_gas(
        state1: ThermodynamicState, state2: ThermodynamicState, cv: float
    ) -> float:
        """Calculate entropy change for ideal gas.

        Parameters
        ----------
        state1:
            Initial state.
        state2:
            Final state.
        cv:
            Molar heat capacity at constant volume (J/(mol·K)).

        Returns
        -------
        delta_S:
            Entropy change (J/K).
        """
        cp = cv + R
        delta_S = state1.n * (
            cv * np.log(state2.T / state1.T) + R * np.log(state2.V / state1.V)
        )
        return delta_S

    @staticmethod
    def clausius_inequality(
        heat_transfers: list[float], temperatures: list[float]
    ) -> float:
        """Evaluate Clausius inequality ∮(dQ/T) ≤ 0 for irreversible cycles.

        Parameters
        ----------
        heat_transfers:
            List of heat transfers for each process (J).
        temperatures:
            Corresponding temperatures (K).

        Returns
        -------
        clausius_integral:
            ∮(dQ/T). Should be ≤ 0 for real processes, = 0 for reversible.
        """
        return sum(Q / T for Q, T in zip(heat_transfers, temperatures))

    @staticmethod
    def universe_entropy_change(
        delta_S_system: float, Q_system: float, T_reservoir: float
    ) -> float:
        """Calculate total entropy change (system + surroundings).

        Second Law: ΔS_universe ≥ 0

        Parameters
        ----------
        delta_S_system:
            Entropy change of system (J/K).
        Q_system:
            Heat absorbed by system (J).
        T_reservoir:
            Reservoir temperature (K).

        Returns
        -------
        delta_S_universe:
            Total entropy change (J/K).
        """
        delta_S_surroundings = -Q_system / T_reservoir
        return delta_S_system + delta_S_surroundings


class HeatEngine:
    """Generic heat engine simulator."""

    def __init__(
        self,
        cycle: ThermodynamicCycle,
        operating_frequency: float = 1.0,
    ):
        """Initialize heat engine.

        Parameters
        ----------
        cycle:
            Thermodynamic cycle.
        operating_frequency:
            Cycles per second (Hz).
        """
        self.cycle = cycle
        self.frequency = operating_frequency

    def power_output(self, Q_in: float) -> float:
        """Calculate power output.

        Parameters
        ----------
        Q_in:
            Heat input per cycle (J).

        Returns
        -------
        P:
            Power output (W).
        """
        eta = self.cycle.efficiency()
        W_per_cycle = eta * Q_in
        return W_per_cycle * self.frequency

    def heat_rejected(self, Q_in: float) -> float:
        """Calculate heat rejected to cold reservoir.

        Parameters
        ----------
        Q_in:
            Heat input per cycle (J).

        Returns
        -------
        Q_out:
            Heat rejected (J).
        """
        W = Q_in * self.cycle.efficiency()
        return Q_in - W


def phase_transition_entropy(
    mass: float, latent_heat: float, T_transition: float
) -> float:
    """Calculate entropy change during phase transition.

    Parameters
    ----------
    mass:
        Mass undergoing transition (kg).
    latent_heat:
        Specific latent heat (J/kg).
    T_transition:
        Transition temperature (K).

    Returns
    -------
    delta_S:
        Entropy change (J/K).
    """
    Q = mass * latent_heat
    return Q / T_transition


def maxwell_boltzmann_distribution(
    v: np.ndarray, T: float, m: float
) -> np.ndarray:
    """Calculate Maxwell-Boltzmann speed distribution.

    Parameters
    ----------
    v:
        Speed array (m/s).
    T:
        Temperature (K).
    m:
        Molecular mass (kg).

    Returns
    -------
    f_v:
        Probability density at each speed.
    """
    # f(v) = 4π(m/2πkT)^(3/2) * v² * exp(-mv²/2kT)
    coeff = 4 * np.pi * (m / (2 * np.pi * k_B * T)) ** (3 / 2)
    return coeff * v**2 * np.exp(-m * v**2 / (2 * k_B * T))


def boltzmann_entropy(W: float) -> float:
    """Calculate Boltzmann entropy S = k*ln(W).

    Parameters
    ----------
    W:
        Number of microstates.

    Returns
    -------
    S:
        Entropy (J/K).
    """
    return k_B * np.log(W)
