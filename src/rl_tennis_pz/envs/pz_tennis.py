from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
import supersuit as ss
from pettingzoo.atari import tennis_v3
from stable_baselines3.common.vec_env import VecEnvWrapper


class PostPointNoDriftPenalty(VecEnvWrapper):  #penalizar movimiento lateral
    def __init__(
        self,
        venv,
        noop_action_idx: int = 0,
        post_point_steps: int = 60,
        move_penalty: float = 1e-3,
    ):
        super().__init__(venv)
        self.noop_action_idx = int(noop_action_idx)
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

        # Activar cooldown cuando termina un punto (reward != 0)
        ended_point = (np.abs(rewards) > 1e-12)
        self._cooldown = np.where(ended_point, self.post_point_steps, self._cooldown)

        # Durante cooldown, penalizar moverse (acción != NOOP)
        if self._last_actions is not None:
            in_cd = (self._cooldown > 0)
            is_move = (self._last_actions != self.noop_action_idx)
            rewards = rewards - self.move_penalty * (in_cd & is_move).astype(np.float32)

        # Decrementar cooldown y resetear en done
        self._cooldown = np.where(self._cooldown > 0, self._cooldown - 1, 0)
        self._cooldown = np.where(dones, 0, self._cooldown)

        return obs, rewards, dones, infos


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
            # firma antigua posicional
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
                # firma con keywords
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


def make_tennis_vec_env(
    num_envs: int = 1,
    num_cpus: int = 1,
    seed: Optional[int] = 0,
    max_cycles: int = 5000,
    reward_shaping: bool = True,
    post_point_steps: int = 60,
    post_point_move_penalty: float = 1e-3,
    **_unused_kwargs,   
):

    # PettingZoo parallel env
    env = tennis_v3.parallel_env(
        obs_type="rgb_image",
        full_action_space=False,
        max_cycles=int(max_cycles),
        auto_rom_install_path=None,
    )
    env.reset(seed=seed)

    # Wrappers Atari
    env = ss.max_observation_v0(env, 2)
    env = ss.sticky_actions_v0(env, repeat_action_probability=0.25)
    env = ss.frame_skip_v0(env, 4)
    env = ss.clip_reward_v0(env, lower_bound=-1.0, upper_bound=1.0)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, 84, 84)
    env = ss.frame_stack_v1(env, 4)

    # Convertir a VecEnv y concatenar
    vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
    vec_env = _robust_concat_vec_envs(vec_env, num_vec_envs=num_envs, num_cpus=num_cpus)

    # Acción NOOP asumida 0
    action_n = vec_env.action_space.n
    print(f"[env] action_n={action_n}, noop=0")

    if reward_shaping:
        vec_env = PostPointNoDriftPenalty(
            vec_env,
            noop_action_idx=0,
            post_point_steps=post_point_steps,
            move_penalty=post_point_move_penalty,
        )

    return vec_env
