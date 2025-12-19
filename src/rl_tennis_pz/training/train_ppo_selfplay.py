import os
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.utils import set_random_seed

from rl_tennis_pz.envs import make_tennis_vec_env


def train_ppo_selfplay(
    total_timesteps: int = 500_000,
    num_envs: int = 4,
    num_cpus: int = 1,
    seed: Optional[int] = 0,
    log_dir: str = "logs/ppo_tennis",
    model_path: str = "models/ppo_tennis_selfplay",
):
    # Entrena PPO con una política CNN compartida para self-play simétrico
    # No se pasa seed al constructor del modelo para evitar llamadas a env.seed()
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Fija seeds globales (numpy/torch) para reproducibilidad
    if seed is not None:
        set_random_seed(seed)

    # Crea el VecEnv de Tennis con wrappers de preprocesamiento y seed en reset
    env = make_tennis_vec_env(num_envs=num_envs, num_cpus=num_cpus, seed=seed)

    # Instancia PPO con CnnPolicy para observaciones de píxeles
    model = PPO(
        policy=CnnPolicy,
        env=env,
        verbose=1,
        gamma=0.99,
        n_steps=256,
        ent_coef=0.01,
        learning_rate=2.5e-4,
        vf_coef=0.5,
        max_grad_norm=0.5,
        gae_lambda=0.95,
        n_epochs=4,
        clip_range=0.1,
        batch_size=256,
        tensorboard_log=log_dir,
    )

    # Ejecuta el entrenamiento por total_timesteps y guarda el modelo final
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)

    # Cierra el entorno para liberar recursos
    env.close()

    # Confirma la ruta del modelo guardado
    print(f"Modelo guardado en: {model_path}.zip")
