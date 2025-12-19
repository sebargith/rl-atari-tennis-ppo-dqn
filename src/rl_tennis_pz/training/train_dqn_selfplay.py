import os
from typing import Optional

from stable_baselines3 import DQN
from stable_baselines3.dqn import CnnPolicy
from stable_baselines3.common.utils import set_random_seed

from rl_tennis_pz.envs import make_tennis_vec_env
from rl_tennis_pz.callbacks.tennis_metrics import TennisMetricsCallback


def train_dqn_selfplay(
    total_timesteps: int = 1_000_000,
    num_envs: int = 1,
    num_cpus: int = 1,
    seed: Optional[int] = 0,
    log_dir: str = "logs/dqn_tennis",
    model_path: str = "models/dqn_tennis_selfplay",
):
    # Entrena DQN con CnnPolicy en self-play simétrico sobre PettingZoo Tennis
    # No se pasa seed al constructor de DQN para evitar llamadas a env.seed()
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Fija semillas globales para reproducibilidad
    if seed is not None:
        set_random_seed(seed)

    # Crea el VecEnv con wrappers de preprocesamiento y shaping configurado
    env = make_tennis_vec_env(
        num_envs=num_envs,
        num_cpus=num_cpus,
        seed=seed,
        max_cycles=20_000,
        reward_shaping=True,
        base_horiz_penalty=5e-5,
        streak_penalty=2e-4,
        streak_k=60,
        bias_decay=0.99,
        bias_penalty=1e-6,
        noop_penalty=0.0,
    )

    # Instancia el modelo DQN con hiperparámetros tipo Atari
    model = DQN(
        policy=CnnPolicy,
        env=env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=20_000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        target_update_interval=10_000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        tensorboard_log=log_dir,
        device="auto",
    )

    # Callback para registrar métricas agregadas y guardarlas al final
    metrics_cb = TennisMetricsCallback(
        quick_threshold=50,
        plot_path=f"plots/dqn_selfplay_metrics_seed{seed}.npz",
    )

    # Ejecuta el entrenamiento por total_timesteps
    model.learn(total_timesteps=total_timesteps, callback=metrics_cb)

    # Guarda métricas (npz) y un gráfico (png)
    metrics_cb.save_npz()
    metrics_cb.save_plot_png(f"plots/dqn_selfplay_metrics_seed{seed}.png")

    # Guarda el modelo entrenado y cierra el entorno
    model.save(model_path)
    env.close()

    # Confirma rutas de outputs
    print(f"Modelo DQN guardado en: {model_path}.zip")
    print(f"Métricas guardadas en: plots/dqn_selfplay_metrics_seed{seed}*.png/.npz")
