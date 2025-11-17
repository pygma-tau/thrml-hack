"""MaxEnt Sokoban gridworld using THRML."""

import dataclasses
from typing import List, Mapping, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Key, PyTree

from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, SamplingSchedule, sample_states
from thrml.conditional_samplers import AbstractConditionalSampler, _SamplerState, _State
from thrml.factor import AbstractFactor, FactorSamplingProgram
from thrml.interaction import InteractionGroup
from thrml.pgm import AbstractNode

# ------------------------------------------------------------
# Basic grid helpers
# ------------------------------------------------------------


class CategoricalNode(AbstractNode):
    pass


def grid_id(i: int, j: int, W: int) -> int:
    return i * W + j


def grid_coords(idx: int, W: int) -> Tuple[int, int]:
    return divmod(idx, W)


ACTION_DELTAS = jnp.array(
    [[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32
)  # up, down, left, right

# ------------------------------------------------------------
# Sokoban layout + transitions
# ------------------------------------------------------------


def build_sokoban_layout(
    H: int,
    W: int,
    walls: Sequence[Tuple[int, int]],
    goals: Sequence[Tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return wall and goal masks flattened over cells."""
    wall_mask = np.zeros((H * W,), dtype=np.float32)
    for i, j in walls:
        wall_mask[grid_id(i, j, W)] = 1.0
    goal_mask = np.zeros((H * W,), dtype=np.float32)
    for i, j in goals:
        goal_mask[grid_id(i, j, W)] = 1.0
    return wall_mask, goal_mask


def make_sokoban_transition_tables(
    H: int,
    W: int,
    wall_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build deterministic transition lookup tables for player and crate.
    Tables have shape [S_player, S_crate, A] and contain the next indices or -1 if invalid.
    """
    S = H * W
    A = ACTION_DELTAS.shape[0]
    player_next = -np.ones((S, S, A), dtype=np.int32)
    crate_next = -np.ones((S, S, A), dtype=np.int32)

    def valid_cell(idx: int) -> bool:
        return 0 <= idx < S and not wall_mask[idx]

    for sp in range(S):
        if wall_mask[sp]:
            continue
        pi, pj = grid_coords(sp, W)
        for sc in range(S):
            if wall_mask[sc] or sp == sc:
                continue
            ci, cj = grid_coords(sc, W)
            for a, (di, dj) in enumerate(np.asarray(ACTION_DELTAS)):
                ni, nj = pi + int(di), pj + int(dj)
                if not (0 <= ni < H and 0 <= nj < W):
                    continue
                target = grid_id(ni, nj, W)
                if wall_mask[target]:
                    continue
                # Simple move into empty space
                if target != sc:
                    player_next[sp, sc, a] = target
                    crate_next[sp, sc, a] = sc
                else:
                    # Attempting to push the crate
                    bi, bj = ni + int(di), nj + int(dj)
                    if not (0 <= bi < H and 0 <= bj < W):
                        continue
                    crate_target = grid_id(bi, bj, W)
                    if wall_mask[crate_target]:
                        continue
                    # pushing into crate's spot moves player into crate's previous cell (sc)
                    player_next[sp, sc, a] = sc
                    crate_next[sp, sc, a] = crate_target

    return player_next, crate_next


# ------------------------------------------------------------
# Interactions + sampler
# ------------------------------------------------------------


class CatUnary(eqx.Module):
    weights: Array


class SokobanPlayerTransition(eqx.Module):
    player_next_table: Array = eqx.field(static=True)
    num_state_classes: int
    step_cost: float
    invalid_penalty: float


class SokobanCrateTransition(eqx.Module):
    crate_next_table: Array = eqx.field(static=True)
    num_state_classes: int
    invalid_penalty: float


class SokobanActionConsistency(eqx.Module):
    player_next_table: Array = eqx.field(static=True)
    crate_next_table: Array = eqx.field(static=True)
    num_action_classes: int
    invalid_penalty: float


class CategoricalGibbsSampler(AbstractConditionalSampler):
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
        n = output_sd.shape[0]
        C = self.num_classes
        E = jnp.zeros((n, C), dtype=jnp.float32)

        for active, inter, tails in zip(active_flags, interactions, states):
            mask = jnp.expand_dims(active, -1)

            def _select_scalar(x):
                zeros = jnp.zeros_like(x)
                return jnp.sum(jnp.where(active, x, zeros), axis=1)

            if isinstance(inter, CatUnary):
                contrib = jnp.sum(mask * inter.weights, axis=1)
                E = E + contrib.reshape((n, C))

            elif isinstance(inter, SokobanPlayerTransition):
                player_prev, crate_prev, action = tails
                player_prev = _select_scalar(player_prev)
                crate_prev = _select_scalar(crate_prev)
                action = _select_scalar(action)
                next_idx = inter.player_next_table[player_prev, crate_prev, action]
                oh = jax.nn.one_hot(next_idx, inter.num_state_classes, dtype=jnp.float32)
                valid = (next_idx >= 0).astype(jnp.float32)[..., None]
                mismatch = 1.0 - oh
                energies = inter.step_cost + inter.invalid_penalty * (
                    valid * mismatch + (1.0 - valid)
                )
                contrib = jnp.sum(mask * energies, axis=1)
                E = E + contrib.reshape((n, C))

            elif isinstance(inter, SokobanCrateTransition):
                player_prev, crate_prev, action = tails
                player_prev = _select_scalar(player_prev)
                crate_prev = _select_scalar(crate_prev)
                action = _select_scalar(action)
                next_idx = inter.crate_next_table[player_prev, crate_prev, action]
                oh = jax.nn.one_hot(next_idx, inter.num_state_classes, dtype=jnp.float32)
                valid = (next_idx >= 0).astype(jnp.float32)[..., None]
                mismatch = 1.0 - oh
                energies = inter.invalid_penalty * (valid * mismatch + (1.0 - valid))
                contrib = jnp.sum(mask * energies, axis=1)
                E = E + contrib.reshape((n, C))

            elif isinstance(inter, SokobanActionConsistency):
                player_prev, crate_prev, player_next, crate_next = tails
                player_prev = _select_scalar(player_prev)
                crate_prev = _select_scalar(crate_prev)
                player_next = _select_scalar(player_next)
                crate_next = _select_scalar(crate_next)
                pred_player = inter.player_next_table[player_prev, crate_prev]
                pred_crate = inter.crate_next_table[player_prev, crate_prev]
                valid = ((pred_player >= 0) & (pred_crate >= 0)).astype(jnp.float32)
                match_player = (pred_player == player_next[:, None]).astype(jnp.float32)
                match_crate = (pred_crate == crate_next[:, None]).astype(jnp.float32)
                matches = match_player * match_crate
                mismatch = 1.0 - matches
                energies = inter.invalid_penalty * (valid * mismatch + (1.0 - valid))
                contrib = jnp.sum(mask * energies, axis=1)
                E = E + contrib.reshape((n, C))

        logits = -self.beta * E
        new_state = jax.random.categorical(key, logits, axis=-1).astype(output_sd.dtype)
        return new_state, sampler_state

    def init(self) -> _SamplerState:
        return None


# ------------------------------------------------------------
# Factors
# ------------------------------------------------------------


class PlayerUnaryFactor(AbstractFactor):
    weights: Array

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


class CrateUnaryFactor(AbstractFactor):
    weights: Array

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


class SokobanTransitionFactor(AbstractFactor):
    player_next_table: Array
    crate_next_table: Array
    step_cost: float
    invalid_penalty: float
    num_states: int
    num_actions: int

    def __init__(
        self,
        blocks: Tuple[Block, Block, Block, Block, Block],
        player_next_table: Array,
        crate_next_table: Array,
        step_cost: float,
        invalid_penalty: float,
        num_states: int,
        num_actions: int,
    ):
        super().__init__(list(blocks))
        self.player_next_table = player_next_table
        self.crate_next_table = crate_next_table
        self.step_cost = step_cost
        self.invalid_penalty = invalid_penalty
        self.num_states = num_states
        self.num_actions = num_actions

    def to_interaction_groups(self) -> list[InteractionGroup]:
        player_next_block, crate_next_block, player_block, crate_block, action_block = self.node_groups

        return [
            InteractionGroup(
                interaction=SokobanPlayerTransition(
                    player_next_table=self.player_next_table,
                    num_state_classes=self.num_states,
                    step_cost=self.step_cost,
                    invalid_penalty=self.invalid_penalty,
                ),
                head_nodes=player_next_block,
                tail_nodes=[player_block, crate_block, action_block],
            ),
            InteractionGroup(
                interaction=SokobanCrateTransition(
                    crate_next_table=self.crate_next_table,
                    num_state_classes=self.num_states,
                    invalid_penalty=self.invalid_penalty,
                ),
                head_nodes=crate_next_block,
                tail_nodes=[player_block, crate_block, action_block],
            ),
            InteractionGroup(
                interaction=SokobanActionConsistency(
                    player_next_table=self.player_next_table,
                    crate_next_table=self.crate_next_table,
                    num_action_classes=self.num_actions,
                    invalid_penalty=self.invalid_penalty,
                ),
                head_nodes=action_block,
                tail_nodes=[player_block, crate_block, player_next_block, crate_next_block],
            ),
        ]


# ------------------------------------------------------------
# Program builder + helpers
# ------------------------------------------------------------


def build_maxent_sokoban_program(
    H: int = 8,
    W: int = 8,
    walls: Sequence[Tuple[int, int]] = (),
    goals: Sequence[Tuple[int, int]] = ((6, 6),),
    player_start: Tuple[int, int] = (0, 0),
    crate_start: Tuple[int, int] = (3, 3),
    T: int = 12,
    step_cost: float = 1.0,
    invalid_penalty: float = 100.0,
    obstacle_penalty: float = 1e3,
    goal_bonus: float = 12.0,
    beta: float = 1.0,
    seed: int = 0,
) -> tuple[FactorSamplingProgram, SamplingSchedule, list[Array], list[Array], Mapping[str, Array]]:
    key = jax.random.key(seed)
    wall_mask, goal_mask = build_sokoban_layout(H, W, walls, goals)
    player_next_table, crate_next_table = make_sokoban_transition_tables(H, W, wall_mask)

    S = H * W
    A = ACTION_DELTAS.shape[0]
    player_nodes = [CategoricalNode() for _ in range(T + 1)]
    crate_nodes = [CategoricalNode() for _ in range(T + 1)]
    action_nodes = [CategoricalNode() for _ in range(T)]

    block_player_free = Block(player_nodes[1:])
    block_crate_free = Block(crate_nodes[1:])
    block_action_free = Block(action_nodes)
    block_player0 = Block([player_nodes[0]])
    block_crate0 = Block([crate_nodes[0]])

    free_blocks = [block_player_free, block_crate_free, block_action_free]
    clamped_blocks = [block_player0, block_crate0]

    node_shape_dtypes = {CategoricalNode: jax.ShapeDtypeStruct((), jnp.int32)}
    spec = BlockGibbsSpec(free_blocks, clamped_blocks, node_shape_dtypes)

    wall_weights = wall_mask[None, :] * obstacle_penalty
    player_weights = np.tile(wall_weights, (T + 1, 1))[:, None, :]
    crate_weights = np.tile(wall_weights, (T + 1, 1))[:, None, :]
    goal_ids = np.where(goal_mask > 0)[0]
    if goal_ids.size > 0:
        crate_weights[-1, 0, goal_ids] -= goal_bonus

    player_factor = PlayerUnaryFactor(jnp.array(player_weights), Block(player_nodes))
    crate_factor = CrateUnaryFactor(jnp.array(crate_weights), Block(crate_nodes))

    transition_factor = SokobanTransitionFactor(
        blocks=(
            Block(player_nodes[1:]),
            Block(crate_nodes[1:]),
            Block(player_nodes[:-1]),
            Block(crate_nodes[:-1]),
            Block(action_nodes),
        ),
        player_next_table=jnp.array(player_next_table),
        crate_next_table=jnp.array(crate_next_table),
        step_cost=step_cost,
        invalid_penalty=invalid_penalty,
        num_states=S,
        num_actions=A,
    )

    sampler_player = CategoricalGibbsSampler(num_classes=S, beta=beta)
    sampler_crate = CategoricalGibbsSampler(num_classes=S, beta=beta)
    sampler_action = CategoricalGibbsSampler(num_classes=A, beta=beta)

    prog = FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=[sampler_player, sampler_crate, sampler_action],
        factors=[player_factor, crate_factor, transition_factor],
        other_interaction_groups=[],
    )

    schedule = SamplingSchedule(n_warmup=0, n_samples=128, steps_per_sample=5)
    key, kp, kc, ka = jax.random.split(key, 4)
    init_player = jax.random.randint(kp, (len(block_player_free.nodes),), 0, S)
    init_crate = jax.random.randint(kc, (len(block_crate_free.nodes),), 0, S)
    init_action = jax.random.randint(ka, (len(block_action_free.nodes),), 0, A)
    init_state = [init_player, init_crate, init_action]

    player0 = jnp.array([grid_id(*player_start, W)], dtype=jnp.int32)
    crate0 = jnp.array([grid_id(*crate_start, W)], dtype=jnp.int32)
    clamped_states = [player0, crate0]

    helpers = dict(
        H=H,
        W=W,
        T=T,
        wall_mask=wall_mask,
        goal_mask=goal_mask,
        player_start=player_start,
        crate_start=crate_start,
        goal_ids=goal_ids,
    )
    return prog, schedule, init_state, clamped_states, helpers


def run_sampling(
    prog: FactorSamplingProgram,
    schedule: SamplingSchedule,
    init_state: list[Array],
    clamped_states: list[Array],
    n_batches: int = 64,
    seed: int = 123,
) -> tuple[Array, Array, Array]:
    key = jax.random.key(seed)
    free0 = [jnp.repeat(x[None, ...], repeats=n_batches, axis=0) for x in init_state]
    clamp0 = [jnp.repeat(x[None, ...], repeats=n_batches, axis=0) for x in clamped_states]
    keys = jax.random.split(key, n_batches)
    nodes_to_sample = prog.gibbs_spec.free_blocks

    def one_chain(k, s_free, s_clamp):
        return sample_states(k, prog, schedule, s_free, s_clamp, nodes_to_sample)

    samples = jax.vmap(one_chain)(keys, free0, clamp0)
    return tuple(samples)


def compute_visit_probs(samples: Array, helpers: Mapping[str, Array]) -> np.ndarray:
    """
    Convert sampled categorical states into per-timestep visit probabilities.

    Args:
        samples: Array with shape (n_batches, n_samples, T_free).
        helpers: Helper dict with grid size info.
    """
    H = int(helpers["H"])
    W = int(helpers["W"])
    arr = np.asarray(samples)
    if arr.ndim != 3:
        raise ValueError(f"Expected samples with 3 dims, got {arr.shape}")
    flat = arr.reshape(-1, arr.shape[-1])
    counts = np.zeros((flat.shape[1], H * W), dtype=np.float64)
    for t in range(flat.shape[1]):
        counts[t] = np.bincount(flat[:, t], minlength=H * W)
    visit = counts.reshape(flat.shape[1], H, W)
    totals = visit.sum(axis=(1, 2), keepdims=True)
    totals = np.where(totals == 0, 1.0, totals)
    return visit / totals


def plot_sokoban_layout(
    helpers: Mapping[str, Array],
    ax: plt.Axes | None = None,
    show: bool = True,
):
    H = int(helpers["H"])
    W = int(helpers["W"])
    wall_mask = np.asarray(helpers["wall_mask"]).reshape(H, W)
    goal_mask = np.asarray(helpers["goal_mask"]).reshape(H, W)
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, W / 2), max(4, H / 2)))
        created = True
    else:
        fig = ax.figure
    ax.imshow(wall_mask, origin="lower", cmap="Greys", alpha=0.7, vmin=0.0, vmax=1.0)
    ax.imshow(
        goal_mask,
        origin="lower",
        cmap="viridis",
        alpha=0.4,
        vmin=0.0,
        vmax=1.0,
    )
    ps = helpers["player_start"]
    cs = helpers["crate_start"]
    ax.scatter(ps[1], ps[0], c="tab:blue", s=120, edgecolors="black", label="player start")
    ax.scatter(cs[1], cs[0], c="tab:orange", s=120, edgecolors="black", label="crate start")
    ax.set_title("Sokoban layout")
    ax.set_xticks(np.arange(W))
    ax.set_yticks(np.arange(H))
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_aspect("equal")
    ax.grid(color="lightgray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper right", frameon=False)
    if show and created:
        plt.show()
    return ax


def plot_player_crate_heatmaps(
    helpers: Mapping[str, Array],
    player_visit: np.ndarray,
    crate_visit: np.ndarray,
    timesteps: Sequence[int] | None = None,
    include_overall: bool = True,
    cmap_player: str = "Blues",
    cmap_crate: str = "Oranges",
) -> tuple[plt.Figure, list[list[plt.Axes]]]:
    """
    Plot player and crate visit probabilities side-by-side across timesteps.

    Returns:
        (fig, axes_grid) where axes_grid is a 2D list [row][col].
    """
    num_steps = player_visit.shape[0]
    if timesteps is None:
        timesteps = (0, num_steps // 2, num_steps - 1)
    timesteps = tuple(int(max(0, min(num_steps - 1, t))) for t in timesteps if num_steps > 0)
    num_cols = len(timesteps) + (1 if include_overall else 0)
    fig, axes = plt.subplots(2, num_cols, figsize=(4 * num_cols, 8))
    axes = np.atleast_2d(axes)

    def _plot_panel(ax, data, step, title, cmap):
        plot_sokoban_layout(helpers, ax=ax, show=False)
        vmax = np.max(data[step])
        vmax = vmax if vmax > 0 else 1.0
        ax.imshow(data[step], origin="lower", cmap=cmap, alpha=0.85, vmin=0.0, vmax=vmax)
        ax.set_title(title)

    for col, step in enumerate(timesteps):
        _plot_panel(axes[0, col], player_visit, step, f"player t={step + 1}", cmap_player)
        _plot_panel(axes[1, col], crate_visit, step, f"crate t={step + 1}", cmap_crate)

    if include_overall:
        overall_player = player_visit.sum(axis=0)
        overall_crate = crate_visit.sum(axis=0)
        if np.max(overall_player) > 0:
            overall_player = overall_player / np.max(overall_player)
        if np.max(overall_crate) > 0:
            overall_crate = overall_crate / np.max(overall_crate)
        last_col = num_cols - 1
        plot_sokoban_layout(helpers, ax=axes[0, last_col], show=False)
        axes[0, last_col].imshow(
            overall_player, origin="lower", cmap=cmap_player, alpha=0.85, vmin=0.0, vmax=1.0
        )
        axes[0, last_col].set_title("player aggregate")
        plot_sokoban_layout(helpers, ax=axes[1, last_col], show=False)
        axes[1, last_col].imshow(
            overall_crate, origin="lower", cmap=cmap_crate, alpha=0.85, vmin=0.0, vmax=1.0
        )
        axes[1, last_col].set_title("crate aggregate")

    axes[0, 0].set_ylabel("Player", fontsize=12)
    axes[1, 0].set_ylabel("Crate", fontsize=12)
    fig.tight_layout()
    return fig, axes.tolist()


if __name__ == "__main__":
    H, W = 12, 12
    walls = [(1, 1), (1, 2), (2, 2), (5, 5), (5, 6), (6, 5)]
    goals = [(10, 10)]
    player_start = (0, 0)
    crate_start = (3, 3)
    prog, schedule, init_state, clamped_states, helpers = build_maxent_sokoban_program(
        H=H,
        W=W,
        walls=walls,
        goals=goals,
        player_start=player_start,
        crate_start=crate_start,
        T=12,
        beta=1.0,
    )

    player_samples, crate_samples, action_samples = run_sampling(
        prog,
        schedule,
        init_state,
        clamped_states,
        n_batches=1,
        seed=42,
    )

    player_visit = compute_visit_probs(player_samples, helpers)
    crate_visit = compute_visit_probs(crate_samples, helpers)
    num_steps = crate_visit.shape[0]
    time_indices = (0, max(0, num_steps // 2), max(0, num_steps - 1))
    fig, _ = plot_player_crate_heatmaps(
        helpers,
        player_visit,
        crate_visit,
        timesteps=time_indices,
        include_overall=True,
    )
    fig.savefig("sokoban_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
