import argparse
import os
import sys
from typing import Dict

import numpy as np
import supersuit as ss
from pettingzoo.atari import tennis_v3
from stable_baselines3 import DQN, PPO

# Agrega src/ al sys.path para importar rl_tennis_pz.* al ejecutar desde /scripts
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Helper para transponer observaciones HWC -> CHW
from rl_tennis_pz.evaluation.eval_ppo_selfplay import _obs_hwc_to_chw  # noqa: E402


def make_eval_env(max_cycles: int, render_mode=None):
    # Crea el entorno AEC de PettingZoo Tennis con wrappers Atari-style
    env = tennis_v3.env(
        obs_type="rgb_image",
        full_action_space=False,
        max_cycles=max_cycles,
        auto_rom_install_path=None,
        render_mode=render_mode,
    )

    # Aplica wrappers de Supersuit para preprocesar observaciones y recompensas
    env = ss.max_observation_v0(env, 2)
    env = ss.sticky_actions_v0(env, repeat_action_probability=0.25)
    env = ss.frame_skip_v0(env, 4)
    env = ss.clip_reward_v0(env, lower_bound=-1.0, upper_bound=1.0)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, 84, 84)
    env = ss.frame_stack_v1(env, 4)

    return env


def load_model(algo: str, path: str, device: str):
    # Carga un modelo SB3 según el algoritmo indicado
    algo = algo.upper()
    if algo == "PPO":
        return PPO.load(path, device=device)
    if algo == "DQN":
        return DQN.load(path, device=device)
    raise ValueError(f"algo inválido: {algo} (usa PPO o DQN)")


def maybe_write_frame(env, writer):
    # Escribe un frame al video si existe writer y el env está en rgb_array
    if writer is None:
        return
    frame = env.render()
    if isinstance(frame, np.ndarray):
        writer.append_data(frame)


def eval_vs_model(
    algo_a: str,
    model_a_path: str,
    algo_b: str,
    model_b_path: str,
    episodes: int,
    seed: int,
    max_cycles: int,
    device: str,
    deterministic: bool,
    render: bool,
    record_path: str | None,
) -> Dict[str, float]:
    # Carga modelos A y B
    model_a = load_model(algo_a, model_a_path, device)
    model_b = load_model(algo_b, model_b_path, device)

    # Define render_mode según ventana o grabación
    if record_path is not None:
        render_mode = "rgb_array"
    elif render:
        render_mode = "human"
    else:
        render_mode = None

    # Crea el entorno para evaluación
    env = make_eval_env(max_cycles=max_cycles, render_mode=render_mode)

    # Asigna roles por orden en possible_agents
    agent_a = env.possible_agents[0]
    agent_b = env.possible_agents[1]

    # Acumuladores globales
    total_a = 0.0
    total_b = 0.0
    win_a = 0
    win_b = 0
    ties = 0

    # Inicializa writer si se está grabando
    writer = None
    if record_path is not None:
        import imageio

        os.makedirs(os.path.dirname(record_path) or ".", exist_ok=True)
        writer = imageio.get_writer(record_path, fps=30)

    for ep in range(episodes):
        # Resetea el episodio con seed reproducible
        env.reset(seed=seed + ep)

        # Retorno acumulado por agente en el episodio
        ep_rewards = {a: 0.0 for a in env.possible_agents}

        # Primer frame del episodio si se graba
        maybe_write_frame(env, writer)

        # Loop AEC: cada iteración corresponde al turno de un agente
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            done = termination or truncation

            # Acumula recompensas del paso para cada agente
            for a, r in env.rewards.items():
                ep_rewards[a] += float(r)

            # Si el agente terminó, se pasa action=None
            if done:
                action = None
            else:
                # Valida formato de observación
                if not isinstance(obs, np.ndarray):
                    raise ValueError(f"Obs no es np.ndarray, es {type(obs)}")

                # Convierte obs a CHW para el modelo
                obs_chw = _obs_hwc_to_chw(obs)

                # Predice la acción con el modelo correspondiente al agente
                if agent == agent_a:
                    action, _ = model_a.predict(obs_chw, deterministic=deterministic)
                else:
                    action, _ = model_b.predict(obs_chw, deterministic=deterministic)

            # Avanza el entorno
            env.step(action)

            # Escribe frame o renderiza en ventana
            if writer is not None:
                maybe_write_frame(env, writer)
            elif render and env.render_mode == "human":
                env.render()

        # Calcula retornos del episodio para A y B
        ra = float(ep_rewards[agent_a])
        rb = float(ep_rewards[agent_b])

        total_a += ra
        total_b += rb

        # Actualiza conteo de wins según retorno del episodio
        if ra > rb:
            win_a += 1
        elif rb > ra:
            win_b += 1
        else:
            ties += 1

        print(f"Ep {ep+1:02d}/{episodes} | A({algo_a.upper()})={ra:.3f}  B({algo_b.upper()})={rb:.3f}")

    # Cierra writer si se grabó
    if writer is not None:
        writer.close()
        print(f"Video guardado en: {record_path}")

    # Cierra el entorno
    env.close()

    # Retorna métricas agregadas
    return {
        "reward_a_mean": total_a / episodes,
        "reward_b_mean": total_b / episodes,
        "win_a": win_a,
        "win_b": win_b,
        "ties": ties,
    }


def main():
    # Parsea argumentos CLI para ejecutar A vs B
    p = argparse.ArgumentParser(description="Evaluación A vs B (PPO/DQN) en PettingZoo Tennis")

    # Define algoritmo y ruta del modelo A
    p.add_argument("--algo-a", required=True, choices=["PPO", "DQN", "ppo", "dqn"])
    p.add_argument("--model-a", required=True)

    # Define algoritmo y ruta del modelo B
    p.add_argument("--algo-b", required=True, choices=["PPO", "DQN", "ppo", "dqn"])
    p.add_argument("--model-b", required=True)

    # Define parámetros de evaluación
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-cycles", type=int, default=2000)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--stochastic", action="store_true")

    # Define opciones de render o grabación
    p.add_argument("--render", action="store_true")
    p.add_argument("--record", type=str, default=None)

    args = p.parse_args()

    # Ejecuta la evaluación con los argumentos recibidos
    stats = eval_vs_model(
        algo_a=args.algo_a,
        model_a_path=args.model_a,
        algo_b=args.algo_b,
        model_b_path=args.model_b,
        episodes=args.episodes,
        seed=args.seed,
        max_cycles=args.max_cycles,
        device=args.device,
        deterministic=(not args.stochastic),
        render=args.render,
        record_path=args.record,
    )

    # Imprime resumen final
    print("\n==============================")
    print(" RESUMEN A vs B")
    print("==============================")
    print(f"A: {args.model_a} ({args.algo_a.upper()})")
    print(f"  mean reward: {stats['reward_a_mean']:.3f}")
    print(f"  wins: {stats['win_a']} / {args.episodes}")
    print()
    print(f"B: {args.model_b} ({args.algo_b.upper()})")
    print(f"  mean reward: {stats['reward_b_mean']:.3f}")
    print(f"  wins: {stats['win_b']} / {args.episodes}")
    print()
    print(f"Ties: {stats['ties']} / {args.episodes}")
    print("==============================\n")


if __name__ == "__main__":
    main()
