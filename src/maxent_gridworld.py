# MaxEnt Gridworld with THRML
# ---------------------------
# Patterned after the THRML Gaussian sampling example you pasted, but for categorical variables:
# - custom CategoricalNode
# - custom categorical Gibbs sampler
# - unary state costs + terminal goal reward (negative energy)
# - tri-way transition consistency factor tying (s_t, a_t, s_{t+1})
#
# Requires: thrml, jax, equinox, networkx, numpy, matplotlib (optional for visualization)

import dataclasses
from typing import Hashable, Mapping, List, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from jaxtyping import Array, Key, PyTree

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
from thrml.factor import AbstractFactor, FactorSamplingProgram
from thrml.interaction import InteractionGroup
from thrml.pgm import AbstractNode

# -----------------------
# Grid + variables
# -----------------------

class CategoricalNode(AbstractNode):
    """A discrete node that holds an integer category (e.g., grid cell index or action id)."""
    pass


def grid_id(i: int, j: int, W: int) -> int:
    return i * W + j


def grid_coords(s: int, W: int) -> Tuple[int, int]:
    return divmod(s, W)


def build_grid(
    H: int, W: int, obstacles: List[Tuple[int, int]]
) -> Tuple[nx.Graph, List[int], Array]:
    """Return a 4-neighbor grid graph, a list of obstacle cell ids, and a binary obstacle mask over cells."""
    G = nx.grid_2d_graph(H, W, periodic=False)
    # flatten node labels to ints 0..(H*W-1)
    mapping = {(i, j): grid_id(i, j, W) for i, j in G.nodes}
    G = nx.relabel_nodes(G, mapping, copy=True)
    obs_ids = [grid_id(i, j, W) for (i, j) in obstacles]
    obs_mask = np.zeros((H * W,), dtype=np.float32)
    obs_mask[obs_ids] = 1.0
    return G, obs_ids, jnp.array(obs_mask)


def generate_random_obstacles(
    H: int,
    W: int,
    num_obstacles: int,
    seed: int | None = None,
    forbidden: Sequence[Tuple[int, int]] | None = None,
) -> List[Tuple[int, int]]:
    """
    Sample `num_obstacles` unique cells uniformly at random from the HxW grid.
    Cells listed in `forbidden` (e.g., start/goal) are never selected.
    """
    total_cells = H * W
    forbidden = tuple(forbidden or ())
    forbidden_ids = {grid_id(i, j, W) for (i, j) in forbidden}

    candidate_ids = np.array(
        [idx for idx in range(total_cells) if idx not in forbidden_ids],
        dtype=np.int32,
    )
    if candidate_ids.size == 0:
        return []

    num = int(np.clip(num_obstacles, 0, candidate_ids.size))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(candidate_ids, size=num, replace=False)
    return [grid_coords(int(idx), W) for idx in chosen]


def plot_gridworld(
    helpers: Mapping[str, Array],
    ax: plt.Axes | None = None,
    show: bool = True,
):
    """
    Visualize the grid layout (obstacles, start, goal) described by the helpers dict
    returned from `build_maxent_gridworld_program`.
    """
    H = int(helpers["H"])
    W = int(helpers["W"])
    obs_mask = np.asarray(helpers["obs_mask"]).reshape(H, W)

    created_ax = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4.0, W / 2), max(4.0, H / 2)))
    else:
        fig = ax.figure

    ax.imshow(
        obs_mask,
        origin="lower",
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        alpha=0.6,
    )
    ax.set_title("MaxEnt Gridworld layout")
    ax.set_xlabel("column")
    ax.set_ylabel("row")

    def _plot_marker(node_id: int | None, color: str, label: str):
        if node_id is None:
            return
        i, j = grid_coords(int(node_id), W)
        ax.scatter(
            j,
            i,
            s=160,
            c=color,
            marker="o",
            edgecolors="black",
            linewidths=1.0,
            label=label,
            zorder=3,
        )

    _plot_marker(helpers.get("start_id"), "tab:green", "start")
    _plot_marker(helpers.get("goal_id"), "tab:red", "goal")

    ax.set_xticks(np.arange(W))
    ax.set_yticks(np.arange(H))
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_aspect("equal")
    ax.grid(which="both", color="lightgray", linewidth=0.5, linestyle="-", alpha=0.5)

    # handles, labels = ax.get_legend_handles_labels()
    # if handles:
    #     ax.legend(loc="upper right", frameon=False)

    if show and created_ax:
        plt.show()

    return ax


# Actions: up, down, left, right
ACTION_DELTAS = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)  # U, D, L, R

def make_next_state_table(H: int, W: int, obs_mask: Array) -> Array:
    """
    Table T[s, a] -> s' (or -1 if invalid).
    Invalid if out of bounds or next is obstacle.
    """
    S = H * W
    A = ACTION_DELTAS.shape[0]
    T = -jnp.ones((S, A), dtype=jnp.int32)
    for s in range(S):
        i, j = divmod(s, W)
        for a in range(A):
            di, dj = int(ACTION_DELTAS[a, 0]), int(ACTION_DELTAS[a, 1])
            ni, nj = i + di, j + dj
            if (0 <= ni < H) and (0 <= nj < W):
                snext = ni * W + nj
                if obs_mask[snext] == 0:
                    T = T.at[s, a].set(int(snext))
    return T


# -----------------------
# Interactions and Sampler
# -----------------------

class CatUnary(eqx.Module):
    """Per-node, per-category energy table E[c] added to the head node's categorical energies."""
    # shape [n, k, C] inside sampler (THRML pads to k), but we store [n, C] and THRML handles padding.
    weights: Array  # [n, C]


class TransitionSNext(eqx.Module):
    """Energy contribution for head = s_{t+1}, tails = [s_t, a_t]."""
    next_table: Array = eqx.field(static=True)  # [S, A] int32, -1 if invalid
    num_classes: int   # C_state
    step_cost: float   # per-step cost
    invalid_penalty: float  # penalty if s_{t+1} != next_table[s_t, a_t]
    obstacle_mask: Array = eqx.field(static=True)    # [C_state], 1 for obstacle cells else 0
    obstacle_penalty: float
    beta: float


class TransitionAction(eqx.Module):
    """Energy contribution for head = a_t, tails = [s_t, s_{t+1}]."""
    next_table: Array = eqx.field(static=True)   # [S, A]
    num_classes: int    # C_action
    invalid_penalty: float
    beta: float


class TransitionSPrev(eqx.Module):
    """Energy contribution for head = s_t, tails = [a_t, s_{t+1}]."""
    next_table: Array = eqx.field(static=True)
    num_classes: int    # C_state
    invalid_penalty: float
    beta: float


class CategoricalGibbsSampler(AbstractConditionalSampler):
    """
    Generic categorical Gibbs sampler.
    Accumulates per-class energies from Interactions and draws from softmax(-beta * energy).
    """
    num_classes: int
    beta: float

    def sample(
        self,
        key: Key,
        interactions: list[PyTree],
        active_flags: list[Array],
        states: list[list[_State]],
        sampler_state: _SamplerState,
        output_sd: jax.ShapeDtypeStruct,
    ) -> tuple[Array, _SamplerState]:
        # output_sd.shape is [n], one category per head node in the block
        n = output_sd.shape[0]
        C = self.num_classes
        # Accumulate energies per category for each head node: shape [n, C]
        E = jnp.zeros((n, C), dtype=jnp.float32)

        for active, inter, tails in zip(active_flags, interactions, states):
            # 'active': [n, k] boolean padding mask
            # inter arrays are already padded to [n, k, ...] by THRML; our modules carry parameters we use below.

            if isinstance(inter, CatUnary):
                # THRML pads unary weights over the interaction arity dimension already: [n, k, C]
                w = inter.weights  # [n, k, C]
                E = E + jnp.sum(jnp.expand_dims(active, -1) * w, axis=1)

            elif isinstance(inter, TransitionSNext):
                # tails: [s_t, a_t], each [n, k]
                s_t, a_t = tails
                # elementwise next state
                next_s = inter.next_table[s_t, a_t]  # [n, k]
                # one-hot over candidate s_{t+1} for allowed next state
                oh = jax.nn.one_hot(next_s, inter.num_classes, dtype=jnp.float32)  # [n, k, C]
                # invalid nexts (value -1) produce all-zeros oh; mark them as invalid by treating as mismatch
                valid = (next_s >= 0).astype(jnp.float32)[..., None]  # [n, k, 1]
                # Energy per category: step_cost + penalty if c != next; invalid treated as mismatch
                mismatch = 1.0 - oh  # 1 where category != allowed next (or all 1 if invalid)
                e_kc = inter.step_cost + inter.invalid_penalty * (1.0 - valid * oh - 0.0 * oh + mismatch * valid)
                # obstacle hardening on candidate s_{t+1}
                e_kc = e_kc + inter.obstacle_penalty * inter.obstacle_mask[None, None, :]
                E = E + jnp.sum(jnp.expand_dims(active, -1) * e_kc, axis=1)

            elif isinstance(inter, TransitionAction):
                # tails: [s_t, s_{t+1}]
                s_t, s_tp1 = tails  # [n, k], [n, k]
                # find the action that maps s_t -> s_{t+1} (unique when connected, else none)
                # We'll compute allowed action by scanning actions: arg where next_table[s_t, a] == s_tp1, else -1
                S, A = inter.next_table.shape
                # Build next states for all actions: [n, k, A]
                all_next = inter.next_table[s_t[..., None], jnp.arange(A)[None, None, :]]
                # match mask across actions
                match = (all_next == s_tp1[..., None]).astype(jnp.float32)  # [n, k, A]
                # If no match (dead-end), treat all as mismatch -> uniform high energy
                # energy per action: 0 for matching action, penalty otherwise
                e_kc = inter.invalid_penalty * (1.0 - match)
                E = E + jnp.sum(jnp.expand_dims(active, -1) * e_kc, axis=1)

            elif isinstance(inter, TransitionSPrev):
                # tails: [a_t, s_{t+1}]
                a_t, s_tp1 = tails  # [n, k]
                # compute predecessor state s_t that under action a_t leads to s_{t+1}
                # We invert by brute-force: for each candidate s, check next_table[s, a_t] == s_tp1
                # Build candidate next for all s: [n, k, S]
                S, A = inter.next_table.shape
                # For efficiency, precompute per-action inverse is ideal; for clarity, do direct test:
                # We'll vectorize: for each head candidate c in [0..S), next_table[c, a_t] == s_tp1?
                # Construct [S,] vector of candidate ids and broadcast
                cand = jnp.arange(S, dtype=jnp.int32)[None, None, :]  # [1,1,S]
                nxt = inter.next_table[cand, a_t[..., None]]  # [n, k, S]
                match = (nxt == s_tp1[..., None]).astype(jnp.float32)  # [n, k, S]
                e_kc = inter.invalid_penalty * (1.0 - match)  # penalty when candidate s_t doesn't map correctly
                E = E + jnp.sum(jnp.expand_dims(active, -1) * e_kc, axis=1)

        # Draw a categorical update: logits = -beta * E
        logits = -self.beta * E
        new_state = jax.random.categorical(key, logits, axis=-1).astype(output_sd.dtype)  # [n]
        return new_state, sampler_state

    def init(self) -> _SamplerState:
        return None


# -----------------------
# Factors
# -----------------------

class StateUnaryFactor(AbstractFactor):
    """Unary energies for state variables (e.g., obstacle penalties per time step; terminal goal bonus)."""
    weights: Array  # [num_nodes, C_state]

    def __init__(self, weights: Array, block: Block):
        super().__init__([block])
        self.weights = weights

    def to_interaction_groups(self) -> list[InteractionGroup]:
        return [
            InteractionGroup(
                interaction=CatUnary(self.weights),
                head_nodes=self.node_groups[0],
                tail_nodes=[],
            )
        ]


class TransitionFactor(AbstractFactor):
    """
    A single factor that adds tri-way consistency energy across time:
      - head s_{t+1} with tails [s_t, a_t]
      - head a_t with tails [s_t, s_{t+1}]
      - head s_t   with tails [a_t, s_{t+1}]
    Blocks must be aligned over t (equal length).
    """
    next_table: Array
    step_cost: float
    invalid_penalty: float
    obstacle_mask: Array
    obstacle_penalty: float
    beta: float
    C_state: int
    C_action: int

    def __init__(
        self,
        blocks: Tuple[Block, Block, Block],  # (S_next_block, S_t_block, A_t_block)
        next_table: Array,
        step_cost: float,
        invalid_penalty: float,
        obstacle_mask: Array,
        obstacle_penalty: float,
        beta: float,
        C_state: int,
        C_action: int,
    ):
        super().__init__(list(blocks))
        self.next_table = next_table
        self.step_cost = step_cost
        self.invalid_penalty = invalid_penalty
        self.obstacle_mask = obstacle_mask
        self.obstacle_penalty = obstacle_penalty
        self.beta = beta
        self.C_state = C_state
        self.C_action = C_action

    def to_interaction_groups(self) -> list[InteractionGroup]:
        Snext_block, S_block, A_block = self.node_groups

        return [
            # s_{t+1} | s_t, a_t
            InteractionGroup(
                interaction=TransitionSNext(
                    next_table=self.next_table,
                    num_classes=self.C_state,
                    step_cost=self.step_cost,
                    invalid_penalty=self.invalid_penalty,
                    obstacle_mask=self.obstacle_mask,
                    obstacle_penalty=self.obstacle_penalty,
                    beta=self.beta,
                ),
                head_nodes=Snext_block,
                tail_nodes=[S_block, A_block],
            ),
            # a_t | s_t, s_{t+1}
            InteractionGroup(
                interaction=TransitionAction(
                    next_table=self.next_table,
                    num_classes=self.C_action,
                    invalid_penalty=self.invalid_penalty,
                    beta=self.beta,
                ),
                head_nodes=A_block,
                tail_nodes=[S_block, Snext_block],
            ),
            # s_t | a_t, s_{t+1}
            InteractionGroup(
                interaction=TransitionSPrev(
                    next_table=self.next_table,
                    num_classes=self.C_state,
                    invalid_penalty=self.invalid_penalty,
                    beta=self.beta,
                ),
                head_nodes=S_block,
                tail_nodes=[A_block, Snext_block],
            ),
        ]


# -----------------------
# Build a MaxEnt Gridworld program
# -----------------------

def build_maxent_gridworld_program(
    H: int = 8,
    W: int = 8,
    obstacles: List[Tuple[int, int]] = ((3, 3), (3, 4), (4, 3), (4, 4)),
    start: Tuple[int, int] = (0, 0),
    goal: Tuple[int, int] = (7, 7),
    T: int = 16,
    step_cost: float = 1.0,
    goal_bonus: float = 12.0,       # negative energy on s_T == goal
    invalid_penalty: float = 100.0, # harsh penalty for inconsistent transition
    obstacle_penalty: float = 1e3,  # big cost to occupy an obstacle
    beta: float = 1.0,
    seed: int = 0,
):
    """
    Returns: (prog, gibbs_spec, init_state (free blocks), clamped_states (clamped blocks), helper dict)
    """
    key = jax.random.key(seed)

    # grid + tables
    G, obs_ids, obs_mask = build_grid(H, W, list(obstacles))
    S = H * W
    A = ACTION_DELTAS.shape[0]
    next_table = make_next_state_table(H, W, obs_mask)  # [S, A]
    start_id = grid_id(*start, W)
    goal_id = grid_id(*goal, W)

    # Nodes: states s_0..s_T and actions a_0..a_{T-1}
    S_nodes = [CategoricalNode() for _ in range(T + 1)]
    A_nodes = [CategoricalNode() for _ in range(T)]

    # Blocks: free state nodes exclude s_0 (we clamp it), all actions are free
    block_S_free = Block(S_nodes[1:])  # s_1..s_T
    block_A_free = Block(A_nodes)      # a_0..a_{T-1}
    block_S0_clamped = Block([S_nodes[0]])

    free_blocks = [block_S_free, block_A_free]
    clamped_blocks = [block_S0_clamped]

    # node dtypes
    node_shape_dtypes = {CategoricalNode: jax.ShapeDtypeStruct((), jnp.int32)}
    spec = BlockGibbsSpec(free_blocks, clamped_blocks, node_shape_dtypes)

    # Unary energies for all s_t: obstacle penalties
    # weights[t, c] = obstacle_penalty * 1_{c is obstacle}; we will also add a terminal goal bonus at t = T
    weights_per_state = np.tile(obs_mask[None, :], (T + 1, 1)).astype(np.float32) * obstacle_penalty

    # Terminal goal bonus: negative energy on s_T == goal
    weights_per_state[T, goal_id] -= goal_bonus

    state_unary_factor = StateUnaryFactor(jnp.array(weights_per_state), Block(S_nodes))

    # Transition factor across all time steps t=0..T-1
    trans_factor = TransitionFactor(
        blocks=(Block(S_nodes[1:]), Block(S_nodes[:-1]), Block(A_nodes)),
        next_table=next_table,
        step_cost=step_cost,
        invalid_penalty=invalid_penalty,
        obstacle_mask=obs_mask,
        obstacle_penalty=obstacle_penalty,
        beta=beta,
        C_state=S,
        C_action=A,
    )

    # Samplers: one for each free block
    sampler_states = CategoricalGibbsSampler(num_classes=S, beta=beta)
    sampler_actions = CategoricalGibbsSampler(num_classes=A, beta=beta)

    prog = FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=[sampler_states, sampler_actions],
        factors=[state_unary_factor, trans_factor],
        other_interaction_groups=[],
    )

    # (Optional) Equivalent construction via InteractionGroups
    # groups = []
    # for fac in [state_unary_factor, trans_factor]:
    #     groups += fac.to_interaction_groups()
    # prog_alt = BlockSamplingProgram(gibbs_spec=spec, samplers=[sampler_states, sampler_actions], interaction_groups=groups)

    # Sampling schedule
    schedule = SamplingSchedule(n_warmup=0, n_samples=1000, steps_per_sample=5)

    # Initial free-block states: random valid categories
    key, kS, kA = jax.random.split(key, 3)
    init_Sfree = jax.random.randint(kS, (len(block_S_free.nodes),), 0, S)  # we will vmap over batches later
    init_Afree = jax.random.randint(kA, (len(block_A_free.nodes),), 0, A)
    init_state = [init_Sfree, init_Afree]

    # Clamped s_0 to start
    clamped_S0 = jnp.array([start_id])  # shape [1]
    clamped_states = [clamped_S0]

    helpers = dict(
        H=H, W=W, T=T, S_nodes=S_nodes, A_nodes=A_nodes,
        start_id=start_id, goal_id=goal_id, next_table=next_table, obs_mask=obs_mask
    )

    return prog, schedule, init_state, clamped_states, helpers


# -----------------------
# Run many parallel chains and (optionally) visualize occupancy
# -----------------------

def run_sampling(
    prog: FactorSamplingProgram,
    schedule: SamplingSchedule,
    init_state: List[Array],
    clamped_states: List[Array],
    n_batches: int = 256,
    seed: int = 123,
):
    key = jax.random.key(seed)

    # Expand initial states and clamped states across batches
    free0 = [jnp.repeat(x[None, ...], repeats=n_batches, axis=0) for x in init_state]
    clamp0 = [jnp.repeat(x[None, ...], repeats=n_batches, axis=0) for x in clamped_states]
    keys = jax.random.split(key, n_batches)
    nodes_to_sample = prog.gibbs_spec.free_blocks

    # sample_states signature mirrors sample_with_observation but without an observer
    def one_chain(k, s_free, s_clamp):
        return sample_states(k, prog, schedule, s_free, s_clamp, nodes_to_sample)

    batched_samples = jax.vmap(one_chain)(keys, free0, clamp0)
    # `batched_samples` is a list mirroring `prog.gibbs_spec.free_blocks`.
    return tuple(batched_samples)


def compute_state_visit_probs(
    state_samples: Array,
    helpers: Mapping[str, Array],
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert sampled states into per-timestep visit probabilities over the grid.

    Args:
        state_samples: Array with shape (n_batches, n_samples, T_free) containing categorical states.
        helpers: Dict returned from `build_maxent_gridworld_program`.
        normalize: If True, normalize each timestep slice to form a probability distribution.

    Returns:
        np.ndarray of shape (T_free, H, W) with visit probabilities per timestep.
        Note that index 0 corresponds to s_1 (s_0 is clamped).
    """
    H = int(helpers["H"])
    W = int(helpers["W"])
    total_states = H * W

    arr = np.asarray(state_samples)
    if arr.ndim != 3:
        raise ValueError(f"Expected state_samples to have 3 dims, got {arr.shape}")
    flat = arr.reshape(-1, arr.shape[-1])  # [n_batches * n_samples, T_free]

    counts = np.zeros((flat.shape[1], total_states), dtype=np.float64)
    for t in range(flat.shape[1]):
        counts[t] = np.bincount(flat[:, t], minlength=total_states)

    visit = counts.reshape(flat.shape[1], H, W)
    if normalize:
        totals = visit.sum(axis=(1, 2), keepdims=True)
        totals = np.where(totals == 0, 1.0, totals)
        visit = visit / totals
    return visit


def plot_state_visit_heatmaps(
    helpers: Mapping[str, Array],
    visit_probs: np.ndarray,
    timesteps: Sequence[int] | None = None,
    axes: Sequence[plt.Axes] | plt.Axes | None = None,
    include_overall: bool = True,
    cmap: str = "magma",
) -> tuple[plt.Figure | None, list[plt.Axes]]:
    """
    Overlay visit probabilities on the grid layout for selected timesteps.

    Args:
        helpers: Helper dict from program construction.
        visit_probs: Array of shape (T_free, H, W).
        timesteps: Iterable of timestep indices to visualize (0-based for s_1).
        axes: Optional axes to draw on. If None, new axes are created.
        include_overall: Whether to append an additional panel showing aggregate visits.
        cmap: Matplotlib colormap to use for heat values.

    Returns:
        (figure, axes_list)
    """
    num_steps = visit_probs.shape[0]
    if timesteps is None:
        timesteps = (0, num_steps // 2, num_steps - 1)
    timesteps = tuple(int(t) for t in timesteps if num_steps > 0)
    num_panels = len(timesteps) + (1 if include_overall else 0)

    created_fig = False
    fig: plt.Figure | None
    axes_list: list[plt.Axes]

    if axes is None:
        fig, axes_created = plt.subplots(1, num_panels, figsize=(4 * num_panels, 4))
        axes_list = list(np.atleast_1d(axes_created))
        created_fig = True
    else:
        if isinstance(axes, plt.Axes):
            axes_list = [axes]
        else:
            axes_list = list(axes)
        fig = axes_list[0].figure if axes_list else None

    if len(axes_list) != num_panels:
        raise ValueError(f"Expected {num_panels} axes, received {len(axes_list)}")

    vmax = np.max(visit_probs) if visit_probs.size else 1.0
    vmax = max(vmax, 1e-9)

    for ax, step in zip(axes_list[: len(timesteps)], timesteps):
        idx = step % num_steps if num_steps else 0
        plot_gridworld(helpers, ax=ax, show=False)
        im = ax.imshow(
            visit_probs[idx],
            origin="lower",
            cmap=cmap,
            alpha=0.85,
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(f"s_{idx + 1}")
        if created_fig:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    if include_overall:
        overall = visit_probs.sum(axis=0)
        overall_max = np.max(overall)
        if overall_max > 0:
            overall = overall / overall_max
        ax = axes_list[-1]
        plot_gridworld(helpers, ax=ax, show=False)
        ax.imshow(overall, origin="lower", cmap=cmap, alpha=0.85, vmin=0.0, vmax=1.0)
        ax.set_title("aggregate")

    if created_fig:
        fig.tight_layout()
    else:
        fig = axes_list[0].figure if axes_list else None
    return fig, axes_list


# ---------------
# Example usage
# ---------------

if __name__ == "__main__":
    H, W = 16, 16
    start = (0, 0)
    goal = (H - 1, W - 1)
    obstacles = generate_random_obstacles(
        H,
        W,
        num_obstacles=8,
        seed=123,
        forbidden=[start, goal],
    )

    betas = (0.7)
    T = 128
    timesteps_to_plot = (0, T // 2, T - 1)

    fig, axes = plt.subplots(len(betas), len(timesteps_to_plot) + 1, figsize=(4 * (len(timesteps_to_plot) + 1), 4 * len(betas)))
    axes = np.atleast_2d(axes)

    for row, beta in enumerate(betas):
        prog, schedule, init_state, clamped_states, helpers = build_maxent_gridworld_program(
            H=H,
            W=W,
            obstacles=obstacles,
            start=start,
            goal=goal,
            T=T,
            step_cost=0.01,
            goal_bonus=100.0,
            invalid_penalty=2.0,
            obstacle_penalty=1e3,
            beta=beta,
            seed=row,
        )

        schedule = dataclasses.replace(schedule, n_samples=1000, steps_per_sample=50)

        state_samples, _ = run_sampling(
            prog,
            schedule,
            init_state,
            clamped_states,
            n_batches=64,
            seed=42 + row,
        )

        visit_probs = compute_state_visit_probs(state_samples, helpers)
        plot_state_visit_heatmaps(
            helpers,
            visit_probs,
            timesteps=timesteps_to_plot,
            axes=axes[row],
            include_overall=True,
        )
        axes[row][0].set_ylabel(f"beta={beta:.1f}", rotation=90, fontsize=10)

    plt.tight_layout()
    plt.show()
