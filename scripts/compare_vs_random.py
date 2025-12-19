import argparse
import os
import sys
from typing import Dict, Optional

import numpy as np
from stable_baselines3 import PPO

# Agrega src/ al sys.path para importar rl_tennis_pz.* al ejecutar desde /scripts
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Importa helpers de evaluación del entorno y preprocesado de observaciones
from rl_tennis_pz.evaluation.eval_ppo_selfplay import (  # noqa: E402
    _make_eval_env,
    _obs_hwc_to_chw,
)


def eval_model_vs_random(
    model_path: str,
    episodes: int = 20,
    seed: Optional[int] = 0,
) -> Dict[str, float]:
    # Evalúa un modelo PPO contra un oponente aleatorio
    print(f"\n=== Evaluando modelo: {model_path} ===")

    # Carga el modelo PPO desde el archivo .zip
    model = PPO.load(model_path)

    # Crea el entorno sin render para acelerar la evaluación
    env = _make_eval_env(render_mode=None)

    # Acumuladores globales
    total_trained = 0.0
    total_random = 0.0
    win_trained = 0
    win_random = 0
    ties = 0

    # Asigna roles: primer agente entrenado, segundo agente random
    trained_agent = env.possible_agents[0]
    random_agent = env.possible_agents[1]

    for ep in range(episodes):
        # Resetea el entorno con una seed distinta por episodio
        env.reset(seed=(seed + ep) if seed is not None else None)

        # Acumula retorno por agente durante el episodio
        ep_rewards: Dict[str, float] = {agent: 0.0 for agent in env.possible_agents}

        # Itera el entorno en modo AEC (turnos por agente)
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            done = termination or truncation

            # Suma recompensas del paso para cada agente
            for a, r in env.rewards.items():
                ep_rewards[a] += float(r)

            # Si el agente está terminado, se debe pasar action=None
            if done:
                action = None
            else:
                if agent == trained_agent:
                    # Convierte obs de HWC a CHW para calzar con el input del modelo
                    if not isinstance(obs, np.ndarray):
                        raise ValueError(f"Obs no es np.ndarray, es {type(obs)}")
                    obs_chw = _obs_hwc_to_chw(obs)

                    # Predice la acción del modelo en modo determinista
                    action, _ = model.predict(obs_chw, deterministic=True)
                else:
                    # Muestrea una acción aleatoria para el oponente
                    action = env.action_space(agent).sample()

            # Avanza el entorno con la acción elegida
            env.step(action)

        # Obtiene retornos finales del episodio
        rt = float(ep_rewards[trained_agent])
        rr = float(ep_rewards[random_agent])

        total_trained += rt
        total_random += rr

        # Determina ganador comparando retornos del episodio
        if rt > rr:
            win_trained += 1
        elif rr > rt:
            win_random += 1
        else:
            ties += 1

        print(f"Ep {ep + 1:02d}/{episodes} | trained={rt:.3f} random={rr:.3f}")

    # Cierra el entorno para liberar recursos
    env.close()

    # Calcula promedios
    reward_trained_mean = total_trained / episodes
    reward_random_mean = total_random / episodes

    # Imprime resumen
    print(f"\nResumen modelo: {model_path}")
    print(f"  Reward medio entrenado: {reward_trained_mean:.3f}")
    print(f"  Reward medio random:    {reward_random_mean:.3f}")
    print(f"  Wins entrenado: {win_trained} / {episodes}")
    print(f"  Wins random:    {win_random} / {episodes}")
    print(f"  Empates:        {ties} / {episodes}")

    return {
        "reward_trained_mean": reward_trained_mean,
        "reward_random_mean": reward_random_mean,
        "win_trained": win_trained,
        "win_random": win_random,
        "ties": ties,
    }


def main():
    # Parsea argumentos para comparar dos modelos PPO contra random
    parser = argparse.ArgumentParser(
        description="Comparar dos modelos PPO de Tennis vs rival aleatorio."
    )
    parser.add_argument(
        "--model-a",
        type=str,
        default="models/ppo_tennis_selfplay.zip",
        help="Ruta del modelo A.",
    )
    parser.add_argument(
        "--model-b",
        type=str,
        default="models/ppo_tennis_selfplay_2M.zip",
        help="Ruta del modelo B.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Número de episodios por modelo.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed base para los episodios.",
    )

    args = parser.parse_args()

    # Evalúa el modelo A vs random
    stats_a = eval_model_vs_random(
        model_path=args.model_a,
        episodes=args.episodes,
        seed=args.seed,
    )

    # Evalúa el modelo B vs random con un offset de seed
    stats_b = eval_model_vs_random(
        model_path=args.model_b,
        episodes=args.episodes,
        seed=args.seed + 10,
    )

    # Imprime comparación final
    print("\n======================================")
    print(" COMPARACIÓN FINAL VS RANDOM")
    print("======================================")
    print(f"Modelo A: {args.model_a}")
    print(f"  Reward medio entrenado: {stats_a['reward_trained_mean']:.3f}")
    print(f"  Wins entrenado: {stats_a['win_trained']} / {args.episodes}")
    print()
    print(f"Modelo B: {args.model_b}")
    print(f"  Reward medio entrenado: {stats_b['reward_trained_mean']:.3f}")
    print(f"  Wins entrenado: {stats_b['win_trained']} / {args.episodes}")
    print("======================================")


if __name__ == "__main__":
    main()
