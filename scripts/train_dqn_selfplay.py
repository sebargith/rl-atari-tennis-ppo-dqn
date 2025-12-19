import argparse
import os
import sys

# Calcula la ruta raíz del repo (2 niveles arriba de este script)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Agrega la carpeta src/ al sys.path para poder importar rl_tennis_pz.*
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Importa la función de entrenamiento DQN (self-play)
from rl_tennis_pz.training import train_dqn_selfplay  # noqa: E402


def main():
    # Define argumentos por línea de comandos para configurar el entrenamiento
    parser = argparse.ArgumentParser(description="Entrenar DQN self-play en PettingZoo Tennis")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-cpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="logs/dqn_tennis")
    parser.add_argument("--model-path", type=str, default="models/dqn_tennis_selfplay")
    args = parser.parse_args()

    # Llama al entrenamiento con los parámetros elegidos
    train_dqn_selfplay(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_cpus=args.num_cpus,
        seed=args.seed,
        log_dir=args.log_dir,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
