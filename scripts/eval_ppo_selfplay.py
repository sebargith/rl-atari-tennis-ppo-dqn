import argparse
import os
import sys

# Agrega la carpeta src/ al sys.path para importar rl_tennis_pz.* al ejecutar desde /scripts
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Importa funciones para ver jugar al modelo PPO en modo self-play o vs random
from rl_tennis_pz.evaluation import watch_selfplay, watch_vs_random  # noqa: E402


def main():
    # Lee argumentos de línea de comandos para elegir modelo, modo y opciones de evaluación
    parser = argparse.ArgumentParser(
        description="Ver jugar al modelo PPO en Tennis (self-play o vs random)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/ppo_tennis_selfplay.zip",
        help="Ruta del modelo PPO a evaluar (archivo .zip de SB3).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Cantidad de episodios a simular.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed base; por episodio se usa seed + ep.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["selfplay", "vs_random"],
        default="selfplay",
        help="selfplay: modelo vs modelo; vs_random: modelo vs política aleatoria.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Abre una ventana para render (requiere GUI en WSL).",
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Ruta de salida para grabar video (ej: videos/selfplay.mp4).",
    )

    args = parser.parse_args()

    # Ejecuta evaluación según el modo elegido
    if args.mode == "selfplay":
        watch_selfplay(
            model_path=args.model_path,
            episodes=args.episodes,
            seed=args.seed,
            render=args.render,
            record_path=args.record,
        )
    else:
        watch_vs_random(
            model_path=args.model_path,
            episodes=args.episodes,
            seed=args.seed,
            render=args.render,
            record_path=args.record,
        )


if __name__ == "__main__":
    main()
