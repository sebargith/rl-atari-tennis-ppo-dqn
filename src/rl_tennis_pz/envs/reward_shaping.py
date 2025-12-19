from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import supersuit as ss
from pettingzoo.atari import tennis_v3
from stable_baselines3.common.vec_env import VecEnvWrapper


# Reward shaping wrappers

# Penaliza moverse por inercia después de terminar un punto
class PostPointNoDriftPenalty(VecEnvWrapper):
    def __init__(
        self,
        venv,
        move_action_idxs: Iterable[int],
        post_point_steps: int = 60,
        move_penalty: float = 1e-3,
    ):
        super().__init__(venv)
        self.move_action_idxs = np.array(sorted(set(move_action_idxs)), dtype=np.int32)
        self.post_point_steps = int(post_point_steps)
        self.move_penalty = float(move_penalty)

        n = self.venv.num_envs
        self._cooldown = np.zeros(n, dtype=np.int32)
        self._last_actions = None

    def reset(self, **kwargs):
        obs = self.venv.reset(**kwargs)
        self._cooldown[:] = 0
        self._last_actions = None
        return obs

    def step_async(self, actions):
        self._last_actions = np.array(actions, dtype=np.int32).copy()
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        # Si el punto terminó (reward != 0), activar cooldown
        ended_point = (np.abs(rewards) > 1e-12)
        self._cooldown = np.where(ended_point, self.post_point_steps, self._cooldown)

        # Durante cooldown, penalizar acciones de movimiento
        if self._last_actions is not None and self.move_action_idxs.size > 0:
            in_cd = (self._cooldown > 0)
            is_move = np.isin(self._last_actions, self.move_action_idxs)
            rewards = rewards - (self.move_penalty * (in_cd & is_move).astype(np.float32))

        # Decrementar cooldown y resetear en done
        self._cooldown = np.where(self._cooldown > 0, self._cooldown - 1, 0)
        self._cooldown = np.where(dones, 0, self._cooldown)

        return obs, rewards, dones, infos


# Action index helpers

# Grupos de acciones por defecto para un action set típico (18)
def _fallback_action_groups(action_n: int) -> Tuple[list[int], list[int], list[int], int]:
    noop = 0
    right = [3, 6, 8, 11, 14, 16]
    left = [4, 7, 9, 12, 15, 17]

    move = [i for i in range(action_n) if i != noop]
    return left, right, move, noop


# Concat VecEnv robusto para distintas firmas de Supersuit
def _robust_concat_vec_envs(vec_env, num_vec_envs: int, num_cpus: int):
    try:
        return ss.concat_vec_envs_v1(
            vec_env,
            num_vec_envs=num_vec_envs,
            num_cpus=num_cpus,
            base_class="stable_baselines3",
        )
    except TypeError:
        obs_space = getattr(vec_env, "observation_space", None)
        act_space = getattr(vec_env, "action_space", None)

        if obs_space is not None and act_space is not None:
            try:
                return ss.concat_vec_envs_v1(
                    vec_env,
                    num_vec_envs,
                    num_cpus,
                    obs_space,
                    act_space,
                    base_class="stable_baselines3",
                )
            except TypeError:
                try:
                    return ss.concat_vec_envs_v1(
                        vec_env,
                        num_vec_envs=num_vec_envs,
                        num_cpus=num_cpus,
                        obs_space=obs_space,
                        act_space=act_space,
                        base_class="stable_baselines3",
                    )
                except TypeError:
                    pass

        print("[env] concat_vec_envs_v1 multi-cpu falló; fallback a num_cpus=1")
        return ss.concat_vec_envs_v1(
            vec_env,
            num_vec_envs=num_vec_envs,
            num_cpus=1,
            base_class="stable_baselines3",
        )


# Construye un VecEnv listo para SB3 con wrappers tipo Atari y shaping post-punto
def make_tennis_vec_env(
    num_envs: int = 1,
    num_cpus: int = 1,
    seed: Optional[int] = 0,
    max_cycles: int = 5000,
    apply_post_point_shaping: bool = True,
):
    # Entorno parallel de PettingZoo
    par_env = tennis_v3.parallel_env(
        obs_type="rgb_image",
        full_action_space=False,
        max_cycles=int(max_cycles),
        auto_rom_install_path=None,
    )
    par_env.reset(seed=seed)

    # Preprocesamiento estilo Atari
    par_env = ss.max_observation_v0(par_env, 2)
    par_env = ss.sticky_actions_v0(par_env, repeat_action_probability=0.25)
    par_env = ss.frame_skip_v0(par_env, 4)
    par_env = ss.clip_reward_v0(par_env, lower_bound=-1.0, upper_bound=1.0)
    par_env = ss.color_reduction_v0(par_env, mode="B")
    par_env = ss.resize_v1(par_env, 84, 84)
    par_env = ss.frame_stack_v1(par_env, 4)

    # PettingZoo -> VecEnv y concatenación para SB3
    vec_env = ss.pettingzoo_env_to_vec_env_v1(par_env)
    vec_env = _robust_concat_vec_envs(vec_env, num_vec_envs=num_envs, num_cpus=num_cpus)

    # Grupos de acciones (fallback)
    action_n = vec_env.action_space.n
    left_idxs, right_idxs, move_idxs, noop_idx = _fallback_action_groups(action_n)
    print(f"[env] action_n={action_n}, left={left_idxs}, right={right_idxs}, noop={noop_idx}")

    # Penalización post-punto para evitar drift
    if apply_post_point_shaping:
        vec_env = PostPointNoDriftPenalty(
            vec_env,
            move_action_idxs=move_idxs,
            post_point_steps=60,
            move_penalty=1e-3,
        )

    return vec_env
