
# thrml_momentum_swap_search.py
# ---------------------------------------------------
# THRML-based sampler over JJ momentum-computing bit-swap circuit designs.
#
# What it does (out of the box):
#   * Encodes a design-space of discrete parameters: L, gamma, I_minus, DeltaS, DeltaC, tau
#   * Uses THRML Block-Gibbs with a custom 'DesignOracleInteraction' to sample designs
#   * Hard-codes physical/architectural constraints from Ray & Crutchfield (2022)
#       - storage potential has two minima; computation potential one minimum
#       - avoid three-minima region (beta > beta_star) unless far enough from bifurcation
#       - timing half-period match in phi and integer-period in phi_dc
#   * Energy = weighted sum of: feasibility penalties + timing mismatch + asymmetry & dispersion proxies
#   * Reports best designs and provides hooks to plug in a full Langevin 'exact_oracle' later
#
# Usage:
#   python thrml_momentum_swap_search.py
#
# You can also import `build_program_and_spaces()` and `sample_designs()` from another script.
#
# References for formulas used (section/page numbers refer to Ray & Crutchfield, 2022):
#   - Device potential U^0 and parameter map: Sec. III, Eqs. (1,2); Appendix C
#   - Critical control \u03c6_xdc^c(\u03b2,\u03b3): Appendix D, Eq. (D7)
#   - Bifurcation beta_star(\u03b3): Appendix D, Eq. (D14)
#   - Timing match: \u03c9_\u03c6 \u03c4 \u2248 (2n-1)\u03c0 and \u03c9_dc \u03c4 \u2248 2n\u03c0 (Sec. III.B / Fig. 4 discussion)
#   - Work accounting W = W0 + W\u03c4 and the search pipeline: Sec. IV and Appendix E
#
# (c) 2025 — Public domain example for research prototyping.
#

'''
export PYTHONPATH=src
for rank in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$rank \
  .venv/bin/python src/thrml_momentum_swap_search.py \
    --n-chains 96 --n-samples 400 --steps-per-sample 4 --n-warmup 50 \
    --num-workers 8 --worker-rank $rank --seed 123 \
    --output-dir /tmp/thrml_runs --output-prefix h100 \
    --no-summary &
done
wait
'''



from __future__ import annotations

import dataclasses
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import io_callback
import equinox as eqx

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HEATMAP_DIR = REPO_ROOT / "heatmaps"

VAR_NAMES = ("L", "gamma", "I_minus", "DeltaS", "DeltaC", "tau")
VAR_DISPLAY = {
    "L": ("nH", 1e9, "{:.2f}"),
    "gamma": ("", 1.0, "{:.1f}"),
    "I_minus": ("nA", 1e9, "{:.0f}"),
    "DeltaS": ("", 1.0, "{:.2f}"),
    "DeltaC": ("", 1.0, "{:.2f}"),
    "tau": ("\u221a(LC)", 1.0, "{:.2f}"),
}

from thrml.pgm import AbstractNode
from thrml.block_management import Block
from thrml.block_sampling import (
    BlockGibbsSpec,
    BlockSamplingProgram,
    sample_states,
    SamplingSchedule,
)
from thrml.conditional_samplers import (
    _SamplerState,
    _State,
    AbstractConditionalSampler,
)
from thrml.interaction import InteractionGroup

# --------------------------
# Physical constants & utils
# --------------------------

PHI0 = 2.067833848e-15  # magnetic flux quantum [Wb]
PI = np.pi

def clip(v, lo, hi):
    return np.minimum(np.maximum(v, lo), hi)


def _format_axis_label(var_name: str) -> str:
    if var_name not in VAR_DISPLAY:
        return var_name
    unit, _, _ = VAR_DISPLAY[var_name]
    return f"{var_name} [{unit}]" if unit else var_name


def _format_axis_tick(var_name: str, value: float) -> str:
    unit, scale, fmt = VAR_DISPLAY.get(var_name, ("", 1.0, "{:.2f}"))
    scaled = value * scale
    return fmt.format(scaled)


# --------------------------
# Design space
# --------------------------

@dataclass(frozen=True)
class DesignSpaces:
    L_values: np.ndarray            # Henries
    gamma_values: np.ndarray        # dimensionless L/(2ell)
    I_minus_values: np.ndarray      # Amperes
    DeltaS_values: np.ndarray       # storage offset +\u0394S
    DeltaC_values: np.ndarray       # compute offset  -\u0394C
    tau_values: np.ndarray          # computation time in sqrt(LC) units

    I_plus: float = 2.0e-6          # A, fixed as in paper
    kT_over_U0: float = 0.05        # \u03ba0 = kT/U0 (Appendix E)

    def __post_init__(self):
        # sanity
        assert np.all(self.L_values > 0)
        assert np.all(self.gamma_values > 0)
        assert np.all(self.tau_values > 0)

    def size_per_var(self) -> List[int]:
        return [
            len(self.L_values),
            len(self.gamma_values),
            len(self.I_minus_values),
            len(self.DeltaS_values),
            len(self.DeltaC_values),
            len(self.tau_values),
        ]

    def var_name(self, idx: int) -> str:
        return VAR_NAMES[idx]

    def var_index(self, name: str) -> int:
        if name not in VAR_NAMES:
            raise ValueError(f"Unknown variable '{name}'. Choose from {VAR_NAMES}.")
        return VAR_NAMES.index(name)

    def decode(self, indices: Sequence[int]) -> Dict[str, float]:
        L = float(self.L_values[indices[0]])
        gamma = float(self.gamma_values[indices[1]])
        I_minus = float(self.I_minus_values[indices[2]])
        DeltaS = float(self.DeltaS_values[indices[3]])
        DeltaC = float(self.DeltaC_values[indices[4]])
        tau = float(self.tau_values[indices[5]])
        return {"L": L, "gamma": gamma, "I_minus": I_minus,
                "DeltaS": DeltaS, "DeltaC": DeltaC, "tau": tau,
                "I_plus": float(self.I_plus), "kT_over_U0": float(self.kT_over_U0)}

    def values_for(self, name: str) -> np.ndarray:
        mapping = {
            "L": self.L_values,
            "gamma": self.gamma_values,
            "I_minus": self.I_minus_values,
            "DeltaS": self.DeltaS_values,
            "DeltaC": self.DeltaC_values,
            "tau": self.tau_values,
        }
        if name not in mapping:
            raise ValueError(f"Unknown variable '{name}'. Choose from {VAR_NAMES}.")
        return mapping[name]

# --------------------------
# JJ momentum-computing device model (fast surrogate pieces)
# --------------------------

@dataclass
class DeviceParams:
    L: float
    gamma: float
    I_plus: float
    I_minus: float

    def beta(self) -> float:
        # \u03b2 = I_plus * 2\u03c0 L / \u03a60
        return (self.I_plus * 2 * PI * self.L) / PHI0

    def delta_beta(self) -> float:
        # \u03b4\u03b2 = I_minus * 2\u03c0 L / \u03a60
        return (self.I_minus * 2 * PI * self.L) / PHI0

# Dimensionless potential U^0(\u03c6, \u03c6dc; \u03c6x, \u03c6xdc; \u03b2, \u03b4\u03b2, \u03b3)
def U0(
    phi: float,
    phi_dc: float,
    phi_x: float,
    phi_xdc: float,
    beta: float,
    delta_beta: float,
    gamma: float,
) -> float:
    return (
        0.5 * (phi - phi_x) ** 2
        + 0.5 * gamma * (phi_dc - phi_xdc) ** 2
        + beta * np.cos(phi) * np.cos(phi_dc / 2.0)
        - delta_beta * np.sin(phi) * np.sin(phi_dc / 2.0)
    )

# Central fixed point \u03c6=0: solve Eq. (D4) for \u03c6_dc^0 given \u03c6_xdc (Appendix D)
def solve_phi_dc0(phi_xdc: float, beta: float, gamma: float,
                  max_iters: int = 50, tol: float = 1e-10) -> Optional[float]:
    # Equation: phi_dc0 - (\u03b2/(2\u03b3)) * sin(phi_dc0/2) = phi_xdc
    # Use Newton on f(y) = y - (\u03b2/(2\u03b3)) sin(y/2) - phi_xdc
    y = float(phi_xdc)  # initial guess
    for _ in range(max_iters):
        f = y - (beta / (2.0 * gamma)) * np.sin(y / 2.0) - phi_xdc
        df = 1.0 - (beta / (4.0 * gamma)) * np.cos(y / 2.0)
        y_new = y - f / df
        if abs(y_new - y) < tol:
            return float(y_new)
        y = y_new
    return None

# Critical control (bifurcation) value \u03c6_xdc^c(\u03b2,\u03b3) — Appendix D, Eq. (D7)
def phi_xdc_critical(beta: float, gamma: float) -> float:
    if beta <= 1.0:
        return -2.0 * np.arccos(clip(1.0 / max(beta,1.0), -1.0, 1.0))
    term = np.sqrt(max(0.0, 1.0 - 1.0 / (beta * beta)))
    return -2.0 * np.arccos(1.0 / beta) + (beta / (2.0 * gamma)) * term

# Bifurcation threshold \u03b2*(\u03b3) — Appendix D, Eq. (D14)
def beta_star(gamma: float) -> float:
    return np.sqrt((4.0 * gamma + 2.0) / 3.0)

# Hessian entries at \u03c6=0 (used to approximate small-oscillation frequencies)
# \u03bb1 = \u2202^2 U / \u2202\u03c6^2 at \u03c6=0; \u03bb2 = \u2202^2 U / \u2202\u03c6_dc^2 at \u03c6=0 (Appendix D, Eqs. D5,D6)
def hessian_diagonals_at_central(phi_dc0: float, beta: float, gamma: float) -> Tuple[float, float]:
    c = np.cos(phi_dc0 / 2.0)
    lam1 = -beta * c + 1.0
    lam2 = gamma - (beta / 4.0) * c
    return lam1, lam2

# Masses in dimensionless coordinates (Appendix C)
m_phi = 1.0
m_phi_dc = 0.25  # (1/4)

def small_oscillation_freqs(phi_dc0: float, beta: float, gamma: float) -> Tuple[float, float]:
    lam1, lam2 = hessian_diagonals_at_central(phi_dc0, beta, gamma)
    # Frequencies \u03c9 \u2248 sqrt((1/m) * \u03bb) when cross-coupling is small near the central minimum.
    w_phi = np.sqrt(max(1e-12, lam1 / m_phi))
    w_dc = np.sqrt(max(1e-12, lam2 / m_phi_dc))
    return float(w_phi), float(w_dc)

# "min-of-mid" \u03c6_x (Sec. D.2): flatten asymmetry near \u03c6=0 for V_comp
def phi_x_min_of_mid(delta_beta: float, phi_dc: float) -> float:
    # From Appendix D: \u03c6_x = - \u03b4\u03b2 sin(\u03c6_dc/2)
    return - delta_beta * np.sin(phi_dc / 2.0)

# "min-of-max" \u03c6_x (Sec. D.2): numerically minimize the max of U_asym along \u03c6 for V_store
def phi_x_min_of_max(delta_beta: float, phi_dc: float) -> float:
    # Coarse search over \u03c6_x in a modest range around 0 to minimize the worst-case U_asym. This is a simplification.
    xs = np.linspace(-0.6, 0.6, 181)
    best_x = 0.0
    best_val = np.inf
    if abs(np.cos(phi_dc / 2.0)) < 1e-9 or abs(delta_beta) < 1e-9:
        return 0.0
    for phi_x in xs:
        # Maximum U_asym at \u03c6 maximizing U_asym = 0.5 \u03c6_x^2 - \u03c6*\u03c6_x - \u03b4\u03b2 sin \u03c6 cos(\u03c6_dc/2)
        phis = np.linspace(-np.pi, np.pi, 361)
        U_asym = 0.5 * phi_x * phi_x - phis * phi_x - delta_beta * np.sin(phis) * np.cos(phi_dc / 2.0)
        val = float(np.max(U_asym))
        if val < best_val:
            best_val = val
            best_x = float(phi_x)
    return best_x

# --------------------------
# Energy Oracle (fast, constraint-aware)
# --------------------------

@dataclass
class EnergyWeights:
    w_infeasible: float = 1e6
    w_three_minima: float = 5e3
    w_timing: float = 50.0
    w_asym: float = 2.0
    w_dispersion: float = 5.0
    w_distance_to_bifurcation: float = 2.0

@dataclass
class EnergyOracle:
    spaces: DesignSpaces
    weights: EnergyWeights = field(default_factory=EnergyWeights)

    # Minimum offsets recommended to stay away from the bifurcation (Appendix E and Fig. 9 commentary)
    min_DeltaS: float = 0.12
    min_DeltaC: float = 0.25

    N_timing_harmonics: int = 6  # how many 'n' to consider in timing match

    def _compute_common(self, L: float, gamma: float, I_minus: float) -> Tuple[DeviceParams, float, float, float]:
        dev = DeviceParams(L=L, gamma=gamma, I_plus=self.spaces.I_plus, I_minus=I_minus)
        beta = dev.beta()
        delta_b = dev.delta_beta()
        phi_xdc_c = phi_xdc_critical(beta, gamma)
        return dev, beta, delta_b, phi_xdc_c

    def _timing_penalty(self, w_phi: float, w_dc: float, tau: float) -> Tuple[float, Dict[str, float]]:
        # Min over integer n >= 1 of (w_phi*tau - (2n-1)\u03c0)^2 + (w_dc*tau - 2n\u03c0)^2
        best = np.inf
        best_n = 1
        for n in range(1, self.N_timing_harmonics + 1):
            err = (w_phi * tau - (2 * n - 1) * PI) ** 2 + (w_dc * tau - 2 * n * PI) ** 2
            if err < best:
                best = err
                best_n = n
        # Normalize by (2\u03c0)^2 to keep penalty ~O(1) when near match
        norm = (2.0 * PI) ** 2
        return float(best / norm), {"n_match": float(best_n), "phase_err": float(best / norm)}

    def evaluate(self, indices: Sequence[int]) -> Tuple[float, Dict[str, float]]:
        # Return (energy, metrics) for a given design index tuple.
        p = self.spaces.decode(indices)
        dev, beta, delta_b, phi_xdc_c = self._compute_common(p["L"], p["gamma"], p["I_minus"])

        # Quick feasibility: \u03b2 > 1, \u03b3 > \u03b2
        feasible = (beta > 1.0) and (p["gamma"] > beta)
        if not feasible:
            return self.weights.w_infeasible, {
                "feasible": 0.0, "beta": beta, "delta_beta": delta_b, "phi_xdc_c": phi_xdc_c
            }

        # Offsets from critical control: storage above, compute below
        DeltaS = float(p["DeltaS"])
        DeltaC = float(p["DeltaC"])
        tau = float(p["tau"])
        phi_xdc_store = phi_xdc_c + DeltaS
        phi_xdc_comp = phi_xdc_c - DeltaC

        # Avoid too-close-to-bifurcation regions
        if DeltaS < self.min_DeltaS or DeltaC < self.min_DeltaC:
            return self.weights.w_infeasible, {"feasible": 0.0, "reason": "too_close_to_bifurcation"}

        # Three-minima region handling: if \u03b2 > \u03b2*, enforce larger \u0394C
        b_star = beta_star(p["gamma"])
        three_min_pen = 0.0
        if beta > b_star and DeltaC < (self.min_DeltaC + 0.1):
            three_min_pen = self.weights.w_three_minima

        # Compute central \u03c6_dc^0 for V_comp (\u03c6=0) and small-oscillation frequencies
        phi_dc0 = solve_phi_dc0(phi_xdc_comp, beta, p["gamma"])
        if phi_dc0 is None:
            return self.weights.w_infeasible, {"feasible": 0.0, "reason": "no_phi_dc0"}

        w_phi, w_dc = small_oscillation_freqs(phi_dc0, beta, p["gamma"])

        # Timing penalty
        timing_pen, timing_meta = self._timing_penalty(w_phi, w_dc, tau)

        # Set \u03c6_x for comp & store
        phi_x_comp = phi_x_min_of_mid(delta_b, phi_dc0)
        # For store, compute \u03c6_dc0_store (central saddle) just for \u03c6_x choice
        phi_dc0_store = solve_phi_dc0(phi_xdc_store, beta, p["gamma"])
        if phi_dc0_store is None:
            return self.weights.w_infeasible, {"feasible": 0.0, "reason": "no_phi_dc0_store"}
        phi_x_store = phi_x_min_of_max(delta_b, phi_dc0_store)

        # Simple dispersion proxy: prefer larger curvature in \u03c6 under V_comp (reduces spreading)
        lam1, lam2 = hessian_diagonals_at_central(phi_dc0, beta, p["gamma"])
        dispersion_pen = float(1.0 / (1e-6 + lam1))  # smaller lam1 -> more dispersion -> higher penalty

        # Asymmetry proxy (\u03b4\u03b2 too large harms fidelity & sub-Landauer regions per Fig. 6 commentary)
        asym_pen = float(abs(delta_b))

        # Distance from bifurcation proxy (avoid too far too, as it worsens harmonicity)
        dist_pen = float(abs(DeltaC - 0.3) + abs(DeltaS - 0.16))

        # Aggregate energy (dimensionless "cost", not true W). Calibrate weights as needed.
        energy = (
            three_min_pen
            + self.weights.w_timing * timing_pen
            + self.weights.w_asym * asym_pen
            + self.weights.w_dispersion * dispersion_pen
            + self.weights.w_distance_to_bifurcation * dist_pen
        )

        metrics = {
            "feasible": 1.0,
            "beta": float(beta),
            "delta_beta": float(delta_b),
            "phi_xdc_c": float(phi_xdc_c),
            "phi_dc0_comp": float(phi_dc0),
            "phi_x_comp": float(phi_x_comp),
            "phi_x_store": float(phi_x_store),
            "w_phi": float(w_phi),
            "w_dc": float(w_dc),
            "timing_pen": float(timing_pen),
            "dispersion_pen": float(dispersion_pen),
            "asym_pen": float(asym_pen),
            "dist_pen": float(dist_pen),
            **timing_meta,
        }
        return float(energy), metrics

    # Helper used by THRML sampler to score *all* categories for a single variable
    def energy_over_candidates(self, current_indices: Sequence[int], var_id: int, num_classes: int) -> np.ndarray:
        cand = np.array(current_indices, dtype=np.int32)
        energies = np.zeros((num_classes,), dtype=np.float32)
        for c in range(num_classes):
            cand[var_id] = c
            e, _ = self.evaluate(cand)
            energies[c] = e
        return energies

# --------------------------
# THRML glue: categorical nodes and a sampler that asks the oracle
# --------------------------

class CatNode(AbstractNode):
    pass

class OracleInteraction(eqx.Module):
    """An interaction that delegates per-category energies for the head var to a Python oracle.

    For a head variable j with C_j categories, and fixed tails (other variables),
    this returns an energy vector E[c] over c in {0..C_j-1}.
    """
    oracle: EnergyOracle
    var_id: int
    num_classes: int

class OracleGibbsSampler(AbstractConditionalSampler):
    num_classes: int  # filled per-block
    beta: float

    def init(self) -> _SamplerState:
        return None

    def sample(
        self,
        key: jax.Array,
        interactions: List[OracleInteraction],
        active_flags: List[jax.Array],
        states: List[List[_State]],
        sampler_state: _SamplerState,
        output_sd: jax.ShapeDtypeStruct,
    ) -> Tuple[jax.Array, _SamplerState]:
        # There should be exactly one interaction per variable/block.
        assert len(interactions) == 1, "Expected one OracleInteraction per block."
        inter = interactions[0]
        # Reconstruct the full design index vector from tail states + head placeholder.
        # 'states' corresponds to tail_nodes for this interaction group.
        # Each tail block holds a single node; states[0] is a list of arrays for each tail.
        tail_scalars = []
        for tail in states[0]:
            tail_scalars.append(jnp.reshape(tail, (-1,))[0])
        tail_vec = jnp.stack(tail_scalars, axis=0).astype(jnp.int32)

        num_vars_total = len(inter.oracle.spaces.size_per_var())

        def _oracle_energy_host(tail_vals):
            tail_vals_np = np.asarray(tail_vals, dtype=np.int32)
            current_indices = np.zeros((num_vars_total,), dtype=np.int32)
            t_iter = iter(tail_vals_np.tolist())
            for vid in range(num_vars_total):
                if vid == inter.var_id:
                    continue
                current_indices[vid] = next(t_iter)
            E_host = inter.oracle.energy_over_candidates(current_indices, inter.var_id, inter.num_classes)
            return np.asarray(E_host, dtype=np.float32)

        E = io_callback(
            _oracle_energy_host,
            jax.ShapeDtypeStruct((inter.num_classes,), jnp.float32),
            tail_vec,
        )
        E = jnp.asarray(E)
        energies = jnp.broadcast_to(E[None, :], (output_sd.shape[0], inter.num_classes))
        logits = -self.beta * energies
        new_state = jax.random.categorical(key, logits, axis=-1).astype(output_sd.dtype)
        new_state = jnp.reshape(new_state, output_sd.shape)
        return new_state, sampler_state

# --------------------------
# Program construction
# --------------------------

def build_program_and_spaces() -> Tuple[BlockSamplingProgram, SamplingSchedule, List[np.ndarray], List[np.ndarray], DesignSpaces, EnergyOracle]:
    # Discrete grids (feel free to adjust/extend)
    L_vals = np.linspace(0.3e-9, 1.0e-9, 8)         # Henries, 0.3..1.0 nH
    gamma_vals = np.linspace(6.0, 12.0, 13)         # 6..12
    I_minus_vals = np.array([7e-9, 35e-9, 70e-9])   # As in Appendix E, three symmetry levels
    DeltaS_vals = np.array([0.12, 0.16, 0.20, 0.24, 0.30])
    DeltaC_vals = np.array([0.25, 0.30, 0.35, 0.40, 0.50])
    tau_vals = np.linspace(4.0, 5.8, 10)            # \u03c4 in sqrt(LC) units (Sec. III.B / Fig. 4 shows ~4-5.6)

    spaces = DesignSpaces(
        L_values=L_vals,
        gamma_values=gamma_vals,
        I_minus_values=I_minus_vals,
        DeltaS_values=DeltaS_vals,
        DeltaC_values=DeltaC_vals,
        tau_values=tau_vals,
    )

    oracle = EnergyOracle(spaces=spaces)

    # Create one categorical node/block per design variable
    nodes = [CatNode() for _ in range(6)]
    blocks = [Block([n]) for n in nodes]

    # Gibbs spec: all blocks are free, none clamped
    node_shape_dtypes = {CatNode: jax.ShapeDtypeStruct((), jnp.int32)}
    spec = BlockGibbsSpec(free_super_blocks=blocks, clamped_blocks=[], node_shape_dtypes=node_shape_dtypes)

    # Samplers and interactions: one per block
    samplers = []
    inter_groups = []
    sizes = spaces.size_per_var()
    for vid, b in enumerate(blocks):
        samplers.append(OracleGibbsSampler(num_classes=sizes[vid], beta=1.0))
        # Tails = all other blocks except head
        tail_blocks = [blocks[i] for i in range(len(blocks)) if i != vid]
        inter_groups.append(
            InteractionGroup(
                interaction=OracleInteraction(oracle=oracle, var_id=vid, num_classes=sizes[vid]),
                head_nodes=b,
                tail_nodes=tail_blocks,
            )
        )

    prog = BlockSamplingProgram(gibbs_spec=spec, samplers=samplers, interaction_groups=inter_groups)

    # Schedule and initial states
    schedule = SamplingSchedule(n_warmup=50, n_samples=100, steps_per_sample=6)

    # Initial design: choose midpoints
    init = [
        np.array([len(L_vals)//2], dtype=np.int32),
        np.array([len(gamma_vals)//2], dtype=np.int32),
        np.array([0], dtype=np.int32),   # start with smallest I_minus
        np.array([1], dtype=np.int32),   # DeltaS ~ 0.16
        np.array([1], dtype=np.int32),   # DeltaC ~ 0.30
        np.array([len(tau_vals)//2], dtype=np.int32),
    ]
    clamped = []

    return prog, schedule, init, clamped, spaces, oracle

# --------------------------
# Sampling helpers
# --------------------------

def sample_designs(
    prog: BlockSamplingProgram,
    schedule: SamplingSchedule,
    init_state: List[np.ndarray],
    clamped_states: List[np.ndarray],
    n_chains: int = 128,
    seed: int = 0,
    chain_offset: int = 0,
) -> List[np.ndarray]:
    key = jax.random.key(seed + chain_offset * 9973)
    free0 = [jnp.repeat(jnp.asarray(x)[None, ...], repeats=n_chains, axis=0) for x in init_state]
    clamp0 = [jnp.repeat(jnp.asarray(x)[None, ...], repeats=n_chains, axis=0) for x in clamped_states]
    keys = jax.random.split(key, n_chains)

    def one(k, s_free, s_clamp):
        return sample_states(k, prog, schedule, s_free, s_clamp, prog.gibbs_spec.free_blocks)

    per_block_samples: List[List[jnp.ndarray]] = [[] for _ in init_state]
    print(
        f"  -> sampling {n_chains} chains x {schedule.n_samples} samples "
        f"(warmup={schedule.n_warmup}, steps_per_sample={schedule.steps_per_sample}, offset={chain_offset})"
    )
    batched = jax.vmap(one)(keys, free0, clamp0)
    print("  -> sampling complete")
    return batched  # list over blocks; each has shape [n_chains, n_samples, 1]


def score_unique_designs(
    samples: List[np.ndarray],
    oracle: EnergyOracle,
) -> List[Tuple[float, Dict[str, float], List[int]]]:
    arrs = [np.asarray(b).reshape(-1) for b in samples]
    if not arrs:
        return []
    seen = {}
    for i in range(arrs[0].shape[0]):
        idxs = [int(arrs[v][i]) for v in range(len(arrs))]
        energy, meta = oracle.evaluate(idxs)
        key = tuple(idxs)
        if key not in seen or energy < seen[key][0]:
            seen[key] = (energy, meta, idxs)
    return list(seen.values())


def build_energy_heatmap(
    scored_designs: Sequence[Tuple[float, Dict[str, float], List[int]]],
    spaces: DesignSpaces,
    var_x: str,
    var_y: str,
    reduction: str = "min",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    var_x = var_x.strip()
    var_y = var_y.strip()
    idx_x = spaces.var_index(var_x)
    idx_y = spaces.var_index(var_y)
    vals_x = spaces.values_for(var_x)
    vals_y = spaces.values_for(var_y)
    if reduction not in {"min", "mean"}:
        raise ValueError("reduction must be either 'min' or 'mean'")
    grid = np.full((len(vals_x), len(vals_y)), np.inf if reduction == "min" else 0.0, dtype=float)
    counts = np.zeros_like(grid, dtype=np.int32)
    for energy, _, idxs in scored_designs:
        x_idx = int(idxs[idx_x])
        y_idx = int(idxs[idx_y])
        if reduction == "min":
            if energy < grid[x_idx, y_idx]:
                grid[x_idx, y_idx] = energy
        else:
            grid[x_idx, y_idx] += energy
        counts[x_idx, y_idx] += 1
    if reduction == "min":
        grid[counts == 0] = np.nan
    else:
        averaged = np.full_like(grid, np.nan, dtype=float)
        np.divide(grid, counts, out=averaged, where=counts > 0)
        grid = averaged
    return grid, counts, vals_x, vals_y


def save_energy_heatmap(
    grid: np.ndarray,
    vals_x: np.ndarray,
    vals_y: np.ndarray,
    var_x: str,
    var_y: str,
    out_paths: Sequence[Path],
    reduction: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to save heatmaps; install it with `pip install matplotlib`."
        ) from exc
    paths = []
    for raw in out_paths:
        path = Path(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        paths.append(path)
    fig_width = max(4.0, len(vals_x) * 0.4)
    fig_height = max(3.0, len(vals_y) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cmap = mpl.cm.get_cmap("magma").copy()
    cmap.set_bad(color="#d9d9d9")
    data = np.ma.masked_invalid(grid).T
    im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(vals_x)))
    ax.set_xticklabels([_format_axis_tick(var_x, float(v)) for v in vals_x], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(vals_y)))
    ax.set_yticklabels([_format_axis_tick(var_y, float(v)) for v in vals_y])
    ax.set_xlabel(_format_axis_label(var_x))
    ax.set_ylabel(_format_axis_label(var_y))
    ax.set_title(f"{reduction} energy heatmap: {var_x} vs {var_y}")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Energy (a.u.)")
    fig.tight_layout()
    for path in paths:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def summarize_topK(samples: List[np.ndarray], oracle: EnergyOracle, spaces: DesignSpaces,
                   K: int = 10) -> List[Tuple[float, Dict[str, float], List[int]]]:
    scored = score_unique_designs(samples, oracle)
    scored.sort(key=lambda x: x[0])
    return scored[:K]

# --------------------------
# CLI demo
# --------------------------

def _fmt_indices(idxs: Sequence[int], spaces: DesignSpaces) -> str:
    vals = spaces.decode(idxs)
    return (
        f"L={vals['L']*1e9:.3f}nH, gamma={vals['gamma']:.2f}, I-= {vals['I_minus']*1e9:.1f}nA, "
        f"\u0394S={vals['DeltaS']:.2f}, \u0394C={vals['DeltaC']:.2f}, \u03c4={vals['tau']:.2f}\u221a(LC)"
    )

def _partition_chains(total: int, num_workers: int, rank: int) -> tuple[int, int]:
    if num_workers <= 1:
        return 0, total
    base = total // num_workers
    remainder = total % num_workers
    start = rank * base + min(rank, remainder)
    count = base + (1 if rank < remainder else 0)
    return start, count


def _save_worker_outputs(
    samples: List[np.ndarray],
    summary: List[Tuple[float, Dict[str, float], List[int]]],
    out_dir: Path,
    prefix: str,
    worker_rank: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {f"var{i}": np.asarray(arr) for i, arr in enumerate(samples)}
    np.savez(out_dir / f"{prefix}_samples_rank{worker_rank}.npz", **data)
    summary_payload = [
        {"energy": float(e), "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in meta.items()}, "indices": idxs}
        for e, meta, idxs in summary
    ]
    with (out_dir / f"{prefix}_summary_rank{worker_rank}.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2)
    print(f"  -> saved worker results under {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="THRML-based momentum-swap circuit search")
    parser.add_argument("--n-chains", type=int, default=96, help="Total number of parallel chains to run")
    parser.add_argument("--n-samples", type=int, default=800, help="Samples per chain after warmup")
    parser.add_argument("--steps-per-sample", type=int, default=6, help="Gibbs steps between recorded samples")
    parser.add_argument("--n-warmup", type=int, default=100, help="Warmup iterations per chain")
    parser.add_argument("--seed", type=int, default=123, help="Base RNG seed")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of distributed workers")
    parser.add_argument("--worker-rank", type=int, default=0, help="Rank of this worker [0..num_workers-1]")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory to save worker results")
    parser.add_argument("--output-prefix", type=str, default="thrml_momentum_swap", help="Filename prefix for saved outputs")
    parser.add_argument("--topk", type=int, default=12, help="Number of best designs to report")
    parser.add_argument("--no-summary", action="store_true", help="Skip console summary (useful for non-zero workers)")
    parser.add_argument(
        "--heatmap-vars",
        nargs=2,
        metavar=("VAR_X", "VAR_Y"),
        help=f"Generate a heatmap over two variables (options: {', '.join(VAR_NAMES)})",
    )
    parser.add_argument(
        "--heatmap-stat",
        choices=["min", "mean"],
        default="min",
        help="Statistic to aggregate energies per heatmap cell (default: min)",
    )
    parser.add_argument(
        "--heatmap-out",
        type=Path,
        default=None,
        help=f"Additional path to save the heatmap PNG (a copy is always stored under {DEFAULT_HEATMAP_DIR})",
    )
    args = parser.parse_args()

    prog, schedule, init, clamped, spaces, oracle = build_program_and_spaces()
    schedule = dataclasses.replace(
        schedule,
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        steps_per_sample=args.steps_per_sample,
    )

    intro = (
        "THRML momentum-computing bit-swap search\n"
        "----------------------------------------\n"
        "Design variables:\n"
        f"  L:          {len(spaces.L_values)} values (0.3..1.0 nH)\n"
        f"  gamma:      {len(spaces.gamma_values)} values (6..12)\n"
        f"  I_minus:    {len(spaces.I_minus_values)} values (7..70 nA)\n"
        f"  DeltaS:     {len(spaces.DeltaS_values)} values\n"
        f"  DeltaC:     {len(spaces.DeltaC_values)} values\n"
        f"  tau:        {len(spaces.tau_values)} values (4.0..5.8 \u221a(LC))\n"
    )
    print(intro)
    print("Sampling... (this requires thrml + jax)")

    global_start, local_chains = _partition_chains(args.n_chains, args.num_workers, args.worker_rank)
    if local_chains == 0:
        print(f"Worker {args.worker_rank} has no chains assigned; exiting.")
        return

    print(
        f"Worker {args.worker_rank}/{args.num_workers - 1 if args.num_workers > 1 else 0}: "
        f"running {local_chains} chains (global offset {global_start})"
    )
    samples = sample_designs(
        prog,
        schedule,
        init,
        clamped,
        n_chains=local_chains,
        seed=args.seed,
        chain_offset=global_start,
    )

    scored_designs = score_unique_designs(samples, oracle)
    scored_sorted = sorted(scored_designs, key=lambda x: x[0])
    top = scored_sorted[: args.topk]
    if args.output_dir is not None:
        _save_worker_outputs(samples, top, args.output_dir, args.output_prefix, args.worker_rank)
    if args.heatmap_vars is not None:
        var_x, var_y = args.heatmap_vars
        try:
            grid, _counts, vals_x, vals_y = build_energy_heatmap(
                scored_designs, spaces, var_x, var_y, reduction=args.heatmap_stat
            )
        except ValueError as exc:
            print(f"Heatmap skipped: {exc}")
        else:
            heatmap_filename = f"{args.output_prefix}_{var_x}_{var_y}_heatmap.png"
            repo_heatmap_path = DEFAULT_HEATMAP_DIR / heatmap_filename
            save_paths = []
            if args.heatmap_out is not None:
                save_paths.append(Path(args.heatmap_out))
            save_paths.append(repo_heatmap_path)
            unique_paths: List[Path] = []
            for path in save_paths:
                resolved = path.resolve()
                if all(resolved != existing.resolve() for existing in unique_paths):
                    unique_paths.append(path)
            save_energy_heatmap(grid, vals_x, vals_y, var_x, var_y, unique_paths, args.heatmap_stat)
            filled = int(np.count_nonzero(~np.isnan(grid)))
            total = grid.size
            dests = ", ".join(str(p) for p in unique_paths)
            print(
                f"  -> saved heatmap to {dests} ({filled}/{total} cells covered, stat={args.heatmap_stat})"
            )

    if not args.no_summary:
        print("\nTop designs (lower energy = better):\n")
        for rank, (e, meta, idxs) in enumerate(top, 1):
            print(
                f"{rank:2d}. E={e:8.3f} | {_fmt_indices(idxs, spaces)} | "
                f"beta={meta['beta']:.2f}, \u03b4\u03b2={meta['delta_beta']:.3e}, "
                f"w\u03c6={meta['w_phi']:.2f}, w_dc={meta['w_dc']:.2f}, timing_pen={meta['timing_pen']:.4f}, n={int(meta['n_match'])}"
            )

if __name__ == "__main__":
    main()
