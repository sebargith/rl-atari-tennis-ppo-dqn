from typing import Optional, Dict

import numpy as np
import supersuit as ss
from pettingzoo.atari import tennis_v3
from stable_baselines3 import PPO

import imageio


def _make_eval_env(render_mode: Optional[str]):
    # Crea el entorno AEC Tennis con el mismo preprocesamiento que en entrenamiento
    env = tennis_v3.env(
        obs_type="rgb_image",
        full_action_space=False,
        max_cycles=20_000,
        auto_rom_install_path=None,
        render_mode=render_mode,
    )

    # Aplica wrappers Atari-style de Supersuit
    env = ss.max_observation_v0(env, 2)
    env = ss.sticky_actions_v0(env, repeat_action_probability=0.25)
    env = ss.frame_skip_v0(env, 4)
    env = ss.clip_reward_v0(env, lower_bound=-1.0, upper_bound=1.0)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, 84, 84)
    env = ss.frame_stack_v1(env, 4)

    return env


def _obs_hwc_to_chw(obs: np.ndarray) -> np.ndarray:
    # Convierte observación HWC a CHW para coincidir con VecTransposeImage en SB3
    if obs.ndim != 3:
        raise ValueError(f"Esperaba obs con 3 dims (HWC), recibí shape={obs.shape}")
    return np.transpose(obs, (2, 0, 1))


def _maybe_write_frame(env, writer):
    # Escribe un frame al video si hay writer y el render devuelve un array
    if writer is None:
        return
    frame = env.render()
    if isinstance(frame, np.ndarray):
        writer.append_data(frame)


def watch_selfplay(
    model_path: str = "models/ppo_tennis_selfplay.zip",
    episodes: int = 3,
    seed: Optional[int] = 0,
    render: bool = False,
    record_path: Optional[str] = None,
):
    # Define render_mode según si se graba video o se intenta mostrar ventana
    if record_path is not None:
        render_mode = "rgb_array"
    elif render:
        render_mode = "human"
    else:
        render_mode = None

    # Carga el modelo PPO y crea el entorno
    model = PPO.load(model_path)
    env = _make_eval_env(render_mode=render_mode)

    # Crea writer si se grabará video
    writer = None
    if record_path is not None:
        writer = imageio.get_writer(record_path, fps=30)

    # Ejecuta episodios en self-play (misma política para ambos agentes)
    for ep in range(episodes):
        print(f"=== Self-play: episodio {ep + 1}/{episodes} ===")
        env.reset(seed=(seed + ep) if seed is not None else None)

        # Acumula recompensas por agente durante el episodio
        ep_rewards: Dict[str, float] = {agent: 0.0 for agent in env.possible_agents}

        # Escribe el primer frame del episodio si se graba
        _maybe_write_frame(env, writer)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            done = termination or truncation

            # Acumula recompensas del paso para todos los agentes
            for a, r in env.rewards.items():
                ep_rewards[a] += r

            # Selecciona acción o pasa None si el agente terminó
            if done:
                action = None
            else:
                obs_chw = _obs_hwc_to_chw(obs)
                action, _ = model.predict(obs_chw, deterministic=True)

            # Aplica la acción al entorno
            env.step(action)

            # Escribe frame o renderiza si corresponde
            if record_path is not None:
                _maybe_write_frame(env, writer)
            elif render and env.render_mode == "human":
                env.render()

        # Reporta recompensas finales del episodio
        print("Recompensas episodio:")
        for a, r in ep_rewards.items():
            print(f"  {a}: {r:.3f}")

    # Cierra writer si se grabó video
    if writer is not None:
        writer.close()
        print(f"Video guardado en: {record_path}")

    # Cierra el entorno
    env.close()


def watch_vs_random(
    model_path: str = "models/ppo_tennis_selfplay_2M.zip",
    episodes: int = 3,
    seed: Optional[int] = 0,
    render: bool = False,
    record_path: Optional[str] = None,
):
    # Define render_mode según si se graba video o se intenta mostrar ventana
    if record_path is not None:
        render_mode = "rgb_array"
    elif render:
        render_mode = "human"
    else:
        render_mode = None

    # Carga el modelo PPO y crea el entorno
    model = PPO.load(model_path)
    env = _make_eval_env(render_mode=render_mode)

    # Crea writer si se grabará video
    writer = None
    if record_path is not None:
        writer = imageio.get_writer(record_path, fps=30)

    # Ejecuta episodios: agente entrenado vs agente aleatorio
    for ep in range(episodes):
        print(f"=== VS random: episodio {ep + 1}/{episodes} ===")
        env.reset(seed=(seed + ep) if seed is not None else None)

        # Acumula recompensas por agente durante el episodio
        ep_rewards: Dict[str, float] = {agent: 0.0 for agent in env.possible_agents}
        trained_agent = env.possible_agents[0]

        # Escribe el primer frame del episodio si se graba
        _maybe_write_frame(env, writer)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            done = termination or truncation

            # Acumula recompensas del paso para todos los agentes
            for a, r in env.rewards.items():
                ep_rewards[a] += r

            # Selecciona acción o pasa None si el agente terminó
            if done:
                action = None
            else:
                if agent == trained_agent:
                    obs_chw = _obs_hwc_to_chw(obs)
                    action, _ = model.predict(obs_chw, deterministic=True)
                else:
                    action = env.action_space(agent).sample()

            # Aplica la acción al entorno
            env.step(action)

            # Escribe frame o renderiza si corresponde
            if record_path is not None:
                _maybe_write_frame(env, writer)
            elif render and env.render_mode == "human":
                env.render()

        # Reporta recompensas finales del episodio
        print("Recompensas episodio:")
        for a, r in ep_rewards.items():
            print(f"  {a}: {r:.3f}")

    # Cierra writer si se grabó video
    if writer is not None:
        writer.close()
        print(f"Video guardado en: {record_path}")

    # Cierra el entorno
    env.close()
