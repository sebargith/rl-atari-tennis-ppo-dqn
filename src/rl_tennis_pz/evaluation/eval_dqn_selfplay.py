from typing import Optional, Dict

import numpy as np
from stable_baselines3 import DQN

from rl_tennis_pz.evaluation.eval_ppo_selfplay import (
    _make_eval_env,
    _obs_hwc_to_chw,
    _maybe_write_frame,
)

# _make_eval_env(render_mode) crea el entorno Tennis con wrappers Supersuit
# _obs_hwc_to_chw convierte observaciones HWC a CHW para el modelo
# _maybe_write_frame escribe un frame al writer si existe


def watch_dqn_selfplay(
    model_path: str = "models/dqn_tennis_selfplay.zip",
    episodes: int = 3,
    seed: Optional[int] = 0,
    render: bool = False,
    record_path: Optional[str] = None,
):
    # Define el modo de render según si se graba video o se muestra ventana
    if record_path is not None:
        render_mode = "rgb_array"
    elif render:
        render_mode = "human"
    else:
        render_mode = None

    # Carga el modelo DQN y crea el entorno de evaluación
    model = DQN.load(model_path)
    env = _make_eval_env(render_mode=render_mode)

    # Crea el writer de video si se indicó un path de salida
    writer = None
    if record_path is not None:
        import imageio
        writer = imageio.get_writer(record_path, fps=30)

    # Ejecuta varios episodios de self-play usando la misma política en ambos agentes
    for ep in range(episodes):
        print(f"=== [DQN] Self-play: episodio {ep + 1}/{episodes} ===")

        # Resetea el entorno con seed por episodio
        env.reset(seed=(seed + ep) if seed is not None else None)

        # Inicializa acumulador de recompensas por agente
        ep_rewards: Dict[str, float] = {agent: 0.0 for agent in env.possible_agents}

        # Guarda el primer frame del episodio si se está grabando
        _maybe_write_frame(env, writer)

        # Loop de turnos PettingZoo
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            done = termination or truncation

            # Acumula recompensas del paso para todos los agentes
            for a, r in env.rewards.items():
                ep_rewards[a] += r

            # Si el agente ya terminó, se pasa acción None
            if done:
                action = None
            else:
                # Convierte obs a CHW y predice acción con el modelo
                if not isinstance(obs, np.ndarray):
                    raise ValueError(f"Obs no es np.ndarray, es {type(obs)}")
                obs_chw = _obs_hwc_to_chw(obs)
                action, _ = model.predict(obs_chw, deterministic=True)

            # Aplica la acción al entorno
            env.step(action)

            # Escribe frame o renderiza según el modo
            if record_path is not None:
                _maybe_write_frame(env, writer)
            elif render and env.render_mode == "human":
                env.render()

        # Imprime recompensas acumuladas del episodio
        print("Recompensas episodio:")
        for a, r in ep_rewards.items():
            print(f"  {a}: {r:.3f}")

    # Cierra writer si se grabó video
    if writer is not None:
        writer.close()
        print(f"Video DQN self-play guardado en: {record_path}")

    # Cierra el entorno
    env.close()


def watch_dqn_vs_random(
    model_path: str = "models/dqn_tennis_selfplay.zip",
    episodes: int = 3,
    seed: Optional[int] = 0,
    render: bool = False,
    record_path: Optional[str] = None,
):
    # Define el modo de render según si se graba video o se muestra ventana
    if record_path is not None:
        render_mode = "rgb_array"
    elif render:
        render_mode = "human"
    else:
        render_mode = None

    # Carga el modelo DQN y crea el entorno de evaluación
    model = DQN.load(model_path)
    env = _make_eval_env(render_mode=render_mode)

    # Crea el writer de video si se indicó un path de salida
    writer = None
    if record_path is not None:
        import imageio
        writer = imageio.get_writer(record_path, fps=30)

    # Ejecuta varios episodios: agente entrenado vs agente aleatorio
    for ep in range(episodes):
        print(f"=== [DQN] VS random: episodio {ep + 1}/{episodes} ===")

        # Resetea el entorno con seed por episodio
        env.reset(seed=(seed + ep) if seed is not None else None)

        # Inicializa acumulador de recompensas por agente
        ep_rewards: Dict[str, float] = {agent: 0.0 for agent in env.possible_agents}

        # Define roles fijos: entrenado = first_0, random = second_0
        trained_agent = env.possible_agents[0]
        random_agent = env.possible_agents[1]

        # Guarda el primer frame del episodio si se está grabando
        _maybe_write_frame(env, writer)

        # Loop de turnos PettingZoo
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            done = termination or truncation

            # Acumula recompensas del paso para todos los agentes
            for a, r in env.rewards.items():
                ep_rewards[a] += r

            # Si el agente ya terminó, se pasa acción None
            if done:
                action = None
            else:
                # El agente entrenado usa el modelo, el otro elige acción al azar
                if agent == trained_agent:
                    if not isinstance(obs, np.ndarray):
                        raise ValueError(f"Obs no es np.ndarray, es {type(obs)}")
                    obs_chw = _obs_hwc_to_chw(obs)
                    action, _ = model.predict(obs_chw, deterministic=True)
                else:
                    action = env.action_space(agent).sample()

            # Aplica la acción al entorno
            env.step(action)

            # Escribe frame o renderiza según el modo
            if record_path is not None:
                _maybe_write_frame(env, writer)
            elif render and env.render_mode == "human":
                env.render()

        # Imprime recompensas acumuladas del episodio y resumen entrenado vs random
        print("Recompensas episodio:")
        for a, r in ep_rewards.items():
            print(f"  {a}: {r:.3f}")
        print(f"  -> entrenado: {ep_rewards[trained_agent]:.3f}, random: {ep_rewards[random_agent]:.3f}")

    # Cierra writer si se grabó video
    if writer is not None:
        writer.close()
        print(f"Video DQN vs random guardado en: {record_path}")

    # Cierra el entorno
    env.close()
