import argparse
import os
import sys
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import supersuit as ss
from pettingzoo.atari import tennis_v3
from stable_baselines3 import DQN, PPO

# Agrega src/ al sys.path para importar rl_tennis_pz.* al ejecutar sin instalar el paquete
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Convierte observaciones HWC -> CHW para calzar con el input esperado por la CNN
from rl_tennis_pz.evaluation.eval_ppo_selfplay import _obs_hwc_to_chw  # noqa: E402


def _make_eval_env_fast(render_mode: Optional[str], max_cycles: int):
    # Crea el entorno AEC de Tennis con max_cycles configurable
    env = tennis_v3.env(
        obs_type="rgb_image",
        full_action_space=False,
        max_cycles=max_cycles,
        auto_rom_install_path=None,
        render_mode=render_mode,
    )

    # Aplica wrappers Atari-style para preprocesamiento
    env = ss.max_observation_v0(env, 2)
    env = ss.sticky_actions_v0(env, repeat_action_probability=0.25)
    env = ss.frame_skip_v0(env, 4)
    env = ss.clip_reward_v0(env, lower_bound=-1.0, upper_bound=1.0)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, 84, 84)
    env = ss.frame_stack_v1(env, 4)

    return env


def _load_model(algo: str, model_path: str, device: str):
    # Carga un modelo SB3 según el algoritmo indicado
    algo = algo.upper()
    if algo == "PPO":
        return PPO.load(model_path, device=device)
    if algo == "DQN":
        return DQN.load(model_path, device=device)
    raise ValueError(f"Algo no soportado: {algo}. Usa PPO o DQN.")


def eval_model_vs_random(
    algo: str,
    model_path: str,
    episodes: int,
    seed: Optional[int],
    max_cycles: int,
    device: str,
    deterministic: bool = True,
    verbose: bool = True,
) -> Dict:
    # Evalúa un modelo contra un oponente aleatorio, acumulando rewards por episodio
    print(f"\n=== [{algo.upper()}] Evaluando vs random: {model_path} (max_cycles={max_cycles}) ===")
    model = _load_model(algo, model_path, device=device)

    # Crea el entorno sin render para acelerar la evaluación
    env = _make_eval_env_fast(render_mode=None, max_cycles=max_cycles)

    # Roles por convención: entrenado = possible_agents[0], random = possible_agents[1]
    trained_agent = env.possible_agents[0]
    random_agent = env.possible_agents[1]

    # Series por episodio para graficar y guardar
    ep_trained = []
    ep_random = []
    ep_steps = []

    # Conteos de wins/ties según comparación de retorno total por episodio
    win_trained = 0
    win_random = 0
    ties = 0

    for ep in range(episodes):
        # Resetea el episodio con seed offseteado por episodio
        env.reset(seed=(seed + ep) if seed is not None else None)

        # Retorno acumulado por agente
        ep_rewards = {a: 0.0 for a in env.possible_agents}
        steps = 0

        # Loop AEC: cada iteración es el turno de un agente
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            done = termination or truncation

            # Acumula rewards del paso para cada agente
            for a, r in env.rewards.items():
                ep_rewards[a] += float(r)

            # Selecciona acción del agente actual
            if done:
                action = None
            else:
                if agent == trained_agent:
                    # Transpone obs HWC -> CHW para el modelo
                    if not isinstance(obs, np.ndarray):
                        raise ValueError(f"Obs no es np.ndarray, es {type(obs)}")
                    obs_chw = _obs_hwc_to_chw(obs)
                    action, _ = model.predict(obs_chw, deterministic=deterministic)
                else:
                    # Acción aleatoria para el oponente
                    action = env.action_space(agent).sample()

            # Aplica acción y avanza el entorno
            env.step(action)
            steps += 1

        # Calcula retornos finales del episodio
        rt = float(ep_rewards[trained_agent])
        rr = float(ep_rewards[random_agent])

        ep_trained.append(rt)
        ep_random.append(rr)
        ep_steps.append(steps)

        # Actualiza conteos de resultado por episodio
        if rt > rr:
            win_trained += 1
        elif rr > rt:
            win_random += 1
        else:
            ties += 1

        if verbose:
            print(f"Ep {ep+1:02d}/{episodes} | trained={rt:.3f} random={rr:.3f} | steps={steps}")

    env.close()

    # Convierte a arrays para estadísticas y guardado
    ep_trained = np.array(ep_trained, dtype=np.float32)
    ep_random = np.array(ep_random, dtype=np.float32)
    ep_steps = np.array(ep_steps, dtype=np.int32)

    return {
        "algo": algo.upper(),
        "model_path": model_path,
        "episodes": episodes,
        "seed": seed,
        "max_cycles": max_cycles,
        "device": str(model.device),
        "deterministic": deterministic,
        "reward_trained_mean": float(ep_trained.mean()),
        "reward_random_mean": float(ep_random.mean()),
        "win_trained": int(win_trained),
        "win_random": int(win_random),
        "ties": int(ties),
        "ep_reward_trained": ep_trained,
        "ep_reward_random": ep_random,
        "ep_steps": ep_steps,
    }


def _plot_compare(stats_a: Dict, stats_b: Optional[Dict], out_png: str):
    # Grafica reward por episodio para A y opcionalmente B
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(stats_a["ep_reward_trained"], label=f"trained ({os.path.basename(stats_a['model_path'])})")
    ax.plot(stats_a["ep_reward_random"], label="random")

    if stats_b is not None:
        ax.plot(stats_b["ep_reward_trained"], label=f"B trained ({os.path.basename(stats_b['model_path'])})")
        ax.plot(stats_b["ep_reward_random"], label="B random")

    ax.set_title("Tennis vs Random: reward por episodio")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _save_npz(stats_a: Dict, stats_b: Optional[Dict], out_npz: str):
    # Guarda series por episodio y resúmenes básicos en un npz comprimido
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)

    payload = {
        "A_ep_reward_trained": stats_a["ep_reward_trained"],
        "A_ep_reward_random": stats_a["ep_reward_random"],
        "A_ep_steps": stats_a["ep_steps"],
        "A_reward_trained_mean": np.array([stats_a["reward_trained_mean"]], dtype=np.float32),
        "A_reward_random_mean": np.array([stats_a["reward_random_mean"]], dtype=np.float32),
        "A_win_trained": np.array([stats_a["win_trained"]], dtype=np.int32),
        "A_win_random": np.array([stats_a["win_random"]], dtype=np.int32),
        "A_ties": np.array([stats_a["ties"]], dtype=np.int32),
    }

    if stats_b is not None:
        payload.update(
            {
                "B_ep_reward_trained": stats_b["ep_reward_trained"],
                "B_ep_reward_random": stats_b["ep_reward_random"],
                "B_ep_steps": stats_b["ep_steps"],
                "B_reward_trained_mean": np.array([stats_b["reward_trained_mean"]], dtype=np.float32),
                "B_reward_random_mean": np.array([stats_b["reward_random_mean"]], dtype=np.float32),
                "B_win_trained": np.array([stats_b["win_trained"]], dtype=np.int32),
                "B_win_random": np.array([stats_b["win_random"]], dtype=np.int32),
                "B_ties": np.array([stats_b["ties"]], dtype=np.int32),
            }
        )

    np.savez_compressed(out_npz, **payload)


def main():
    # Parsea argumentos CLI para evaluar A vs random y opcionalmente B vs random
    p = argparse.ArgumentParser(description="Comparar modelos vs random (rápido) + plots")
    p.add_argument("--algo", type=str, choices=["PPO", "DQN", "ppo", "dqn"], required=True)
    p.add_argument("--model-a", type=str, required=True)
    p.add_argument("--model-b", type=str, default=None)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-cycles", type=int, default=2000)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--out-prefix", type=str, default="plots/compare_vs_random_fast")

    args = p.parse_args()

    # Define modo deterministic para predict
    det = not args.stochastic

    # Evalúa el modelo A
    stats_a = eval_model_vs_random(
        algo=args.algo,
        model_path=args.model_a,
        episodes=args.episodes,
        seed=args.seed,
        max_cycles=args.max_cycles,
        device=args.device,
        deterministic=det,
        verbose=True,
    )

    # Evalúa el modelo B si fue provisto
    stats_b = None
    if args.model_b is not None:
        stats_b = eval_model_vs_random(
            algo=args.algo,
            model_path=args.model_b,
            episodes=args.episodes,
            seed=args.seed + 10,
            max_cycles=args.max_cycles,
            device=args.device,
            deterministic=det,
            verbose=True,
        )

    # Imprime resumen final
    print("\n======================================")
    print(" RESUMEN VS RANDOM (FAST)")
    print("======================================")
    print(f"Modelo A: {args.model_a}")
    print(f"  mean reward trained: {stats_a['reward_trained_mean']:.3f}")
    print(f"  wins trained: {stats_a['win_trained']} / {args.episodes}")
    if stats_b is not None:
        print()
        print(f"Modelo B: {args.model_b}")
        print(f"  mean reward trained: {stats_b['reward_trained_mean']:.3f}")
        print(f"  wins trained: {stats_b['win_trained']} / {args.episodes}")
    print("======================================")

    # Genera plot y npz con prefijo definido
    out_png = args.out_prefix + ".png"
    out_npz = args.out_prefix + ".npz"
    _plot_compare(stats_a, stats_b, out_png)
    _save_npz(stats_a, stats_b, out_npz)

    print(f"\nPlot guardado en: {out_png}")
    print(f"Datos guardados en: {out_npz}")


if __name__ == "__main__":
    main()
