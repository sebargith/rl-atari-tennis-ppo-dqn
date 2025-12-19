import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict

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


@dataclass(frozen=True)
class ModelSpec:
    name: str
    algo: str
    path: str


def make_eval_env(max_cycles: int):
    # Crea un entorno AEC con wrappers Atari-style y max_cycles configurable
    env = tennis_v3.env(
        obs_type="rgb_image",
        full_action_space=False,
        max_cycles=max_cycles,
        auto_rom_install_path=None,
        render_mode=None,
    )
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
    a = algo.upper()
    if a == "PPO":
        return PPO.load(path, device=device)
    if a == "DQN":
        return DQN.load(path, device=device)
    raise ValueError(f"Algo inválido: {algo}")


def play_match(
    col: ModelSpec,
    row: ModelSpec,
    episodes: int,
    seed: int,
    max_cycles: int,
    device: str,
    deterministic: bool = True,
    swap_sides: bool = True,
) -> Dict[str, float]:
    # Corre episodios col vs row y retorna win_rate del modelo col
    model_col = load_model(col.algo, col.path, device=device)
    model_row = load_model(row.algo, row.path, device=device)

    env = make_eval_env(max_cycles=max_cycles)
    a0 = env.possible_agents[0]
    a1 = env.possible_agents[1]

    win_col = 0
    win_row = 0
    ties = 0

    for ep in range(episodes):
        env.reset(seed=seed + ep)

        # Alterna lados para reducir sesgo por saque/lado
        if swap_sides and (ep % 2 == 1):
            first_model, first_is_col = model_row, False
            second_model, second_is_col = model_col, True
        else:
            first_model, first_is_col = model_col, True
            second_model, second_is_col = model_row, False

        ep_rewards = {a0: 0.0, a1: 0.0}

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            done = termination or truncation

            # Acumula rewards del paso
            for a, r in env.rewards.items():
                ep_rewards[a] += float(r)

            # Selecciona acción del modelo que controla al agente en turno
            if done:
                action = None
            else:
                if not isinstance(obs, np.ndarray):
                    raise ValueError(f"Obs no es np.ndarray: {type(obs)}")
                obs_chw = _obs_hwc_to_chw(obs)

                if agent == a0:
                    action, _ = first_model.predict(obs_chw, deterministic=deterministic)
                else:
                    action, _ = second_model.predict(obs_chw, deterministic=deterministic)

            env.step(action)

        r0 = float(ep_rewards[a0])
        r1 = float(ep_rewards[a1])

        # Mapea retornos a (col,row) según la asignación de lados
        if first_is_col:
            r_col, r_row = r0, r1
        else:
            r_col, r_row = r1, r0

        if r_col > r_row:
            win_col += 1
        elif r_row > r_col:
            win_row += 1
        else:
            ties += 1

    env.close()

    return {
        "win_col": win_col,
        "win_row": win_row,
        "ties": ties,
        "win_rate_col": win_col / float(episodes),
    }


def plot_matrix(names, mat_sign, mat_text, out_png):
    # Genera una matriz visual de resultados con texto por celda
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    n = len(names)
    data = np.array(mat_sign, dtype=float)

    # Mapea: pierde->0, empate->1, gana->2
    mapped = np.ones((n, n), dtype=int)
    mapped[data < 0] = 0
    mapped[data > 0] = 2
    mapped[np.eye(n, dtype=bool)] = 1

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#d73027", "#ffffff", "#1a9850"])

    fig, ax = plt.subplots(figsize=(1.6 * n + 3, 1.2 * n + 2))
    ax.imshow(mapped, cmap=cmap, vmin=0, vmax=2)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)

    ax.set_xlabel("Modelo (columna)")
    ax.set_ylabel("Oponente (fila)")

    # Dibuja grilla por celda
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Escribe texto dentro de cada celda no diagonal
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ax.text(j, i, mat_text[i][j], ha="center", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    # Parsea argumentos y construye la matriz de enfrentamientos
    p = argparse.ArgumentParser(description="Matriz de enfrentamientos modelo vs modelo + PNG/NPZ")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-cycles", type=int, default=2000)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--out-prefix", type=str, default="plots/matchups_4models")

    args = p.parse_args()

    models = [
        ModelSpec("PPO_500k", "PPO", "models/ppo_tennis_selfplay.zip"),
        ModelSpec("PPO_2M", "PPO", "models/ppo_tennis_selfplay_2M.zip"),
        ModelSpec("DQN_1M", "DQN", "models/dqn_tennis_selfplay_1M.zip"),
        ModelSpec("DQN_vsRand_200k", "DQN", "models/dqn_vs_random_200k.zip"),
    ]

    names = [m.name for m in models]
    n = len(models)

    mat_sign = [[np.nan for _ in range(n)] for _ in range(n)]
    mat_text = [["" for _ in range(n)] for _ in range(n)]

    det = not args.stochastic

    for i_row in range(n):
        for j_col in range(n):
            if i_row == j_col:
                continue

            row = models[i_row]
            col = models[j_col]

            stats = play_match(
                col=col,
                row=row,
                episodes=args.episodes,
                seed=args.seed + 1000 * i_row + 10 * j_col,
                max_cycles=args.max_cycles,
                device=args.device,
                deterministic=det,
                swap_sides=True,
            )

            wr = stats["win_rate_col"]
            if wr > 0.5:
                s = 1
            elif wr < 0.5:
                s = -1
            else:
                s = 0

            mat_sign[i_row][j_col] = s
            mat_text[i_row][j_col] = f"{int(stats['win_col'])}-{int(stats['win_row'])}\n{wr*100:.0f}%"

            print(
                f"[{row.name} (fila) vs {col.name} (col)] "
                f"=> col winrate={wr:.3f} ({stats['win_col']}-{stats['win_row']}, ties={stats['ties']})"
            )

    out_png = args.out_prefix + ".png"
    out_npz = args.out_prefix + ".npz"

    plot_matrix(names, mat_sign, mat_text, out_png)

    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez_compressed(
        out_npz,
        names=np.array(names, dtype=object),
        mat_sign=np.array(mat_sign, dtype=float),
        mat_text=np.array(mat_text, dtype=object),
        episodes=np.array([args.episodes], dtype=int),
        seed=np.array([args.seed], dtype=int),
        max_cycles=np.array([args.max_cycles], dtype=int),
    )

    print(f"\nPNG guardado en: {out_png}")
    print(f"NPZ guardado en: {out_npz}")


if __name__ == "__main__":
    main()
