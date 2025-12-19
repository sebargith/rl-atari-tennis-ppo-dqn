# Atari Tennis RL (PettingZoo) — PPO & DQN + Self-Play

Proyecto de **Aprendizaje por Refuerzo** para entrenar agentes que juegan **Tennis (Atari)** usando el entorno multiagente **PettingZoo (tennis_v3)**, con modelos **PPO** y **DQN** (Stable-Baselines3), evaluación **vs random** y **modelo vs modelo**, además de grabación de videos.

## Qué incluye
- `src/rl_tennis_pz/envs/`  
  Creación del entorno (wrappers de SuperSuit) y lógica de **reward shaping**.
- `src/rl_tennis_pz/training/`  
  Entrenamiento **PPO self-play** y **DQN self-play**.
- `src/rl_tennis_pz/evaluation/`  
  Evaluación y visualización: **self-play**, **vs random**, **modelo vs modelo**.
- `src/rl_tennis_pz/callbacks/tennis_metrics.py`  
  Callback para registrar métricas de juego (por ejemplo: quick events, eventos, etc.) y generar plots.
- `scripts/`  
  Entradas CLI para entrenar, evaluar y grabar videos.

## Requisitos
- Python 3.10+ (recomendado 3.11)
- Linux/WSL2 recomendado
- ROM de Atari Tennis instalada (vía AutoROM o equivalente)

## Instalación (venv)
Desde la carpeta raíz del repo:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Estructura esperada del repo
```text
.
├── src/
│   └── rl_tennis_pz/
├── scripts/
├── models/
├── logs/
├── plots/
└── videos/
```

---

# Uso

## Activar entorno y ubicarse en el repo
```bash
cd rl-tennis-pz
source .venv/bin/activate
```

---

## Entrenar

### PPO self-play
```bash
python scripts/train_ppo_selfplay.py \
  --total-timesteps 500000 \
  --num-envs 4 \
  --num-cpus 1 \
  --seed 0 \
  --model-path models/ppo_tennis_selfplay \
  --log-dir logs/ppo_tennis
```

### DQN self-play
```bash
python scripts/train_dqn_selfplay.py \
  --total-timesteps 200000 \
  --num-envs 1 \
  --num-cpus 1 \
  --seed 0 \
  --model-path models/dqn_tennis_selfplay \
  --log-dir logs/dqn_tennis
```

**Outputs esperados**
- Modelos: `models/*.zip`
- TensorBoard: `logs/*/`

---

# Evaluación

## 1) Evaluación cuantitativa vs random (rápido + plots)
Script: `scripts/eval_vs_random_summary.py`

**Ejemplo PPO (modelo A y B):**
```bash
python scripts/eval_vs_random_summary.py \
  --algo PPO \
  --model-a models/ppo_tennis_selfplay.zip \
  --model-b models/ppo_tennis_selfplay_2M.zip \
  --episodes 20 \
  --seed 0 \
  --max-cycles 2000 \
  --out-prefix plots/ppo_vs_random_summary
```

**Ejemplo DQN (solo A):**
```bash
python scripts/eval_vs_random_summary.py \
  --algo DQN \
  --model-a models/dqn_tennis_selfplay_1M.zip \
  --episodes 20 \
  --seed 0 \
  --max-cycles 2000 \
  --out-prefix plots/dqn_vs_random_summary
```

Genera:
- `plots/<out-prefix>.png`
- `plots/<out-prefix>.npz`

---

## 2) Modelo vs modelo (A vs B) + video opcional
Script: `scripts/eval_model_vs_model.py`

**PPO vs PPO (2M vs 500k) + video:**
```bash
python scripts/eval_model_vs_model.py \
  --algo-a PPO \
  --model-a models/ppo_tennis_selfplay_2M.zip \
  --algo-b PPO \
  --model-b models/ppo_tennis_selfplay.zip \
  --episodes 10 \
  --seed 0 \
  --max-cycles 2000 \
  --record videos/ppo2m_vs_ppo500k.mp4
```

**PPO (A) vs DQN (B) + video:**
```bash
python scripts/eval_model_vs_model.py \
  --algo-a PPO \
  --model-a models/ppo_tennis_selfplay_2M.zip \
  --algo-b DQN \
  --model-b models/dqn_tennis_selfplay_1M.zip \
  --episodes 10 \
  --seed 0 \
  --max-cycles 2000 \
  --record videos/ppo2m_vs_dqn1m.mp4
```
---

## 3) Ver jugar / grabar videos (self-play o vs random)
### PPO
```bash
python scripts/eval_ppo_selfplay.py \
  --mode selfplay \
  --model-path models/ppo_tennis_selfplay.zip \
  --episodes 1 \
  --seed 0 \
  --record videos/ppo_selfplay.mp4
```

```bash
python scripts/eval_ppo_selfplay.py \
  --mode vs_random \
  --model-path models/ppo_tennis_selfplay.zip \
  --episodes 1 \
  --seed 0 \
  --record videos/ppo_vs_random.mp4
```

### DQN
```bash
python scripts/eval_dqn_selfplay.py \
  --mode selfplay \
  --model-path models/dqn_tennis_selfplay_1M.zip \
  --episodes 1 \
  --seed 0 \
  --record videos/dqn_selfplay.mp4
```

```bash
python scripts/eval_dqn_selfplay.py \
  --mode vs_random \
  --model-path models/dqn_vs_random_200k.zip \
  --episodes 1 \
  --seed 0 \
  --record videos/dqn_vs_random.mp4
```

---

# TensorBoard

```bash
tensorboard --logdir logs --port 6006
```

Abrir:
- `http://localhost:6006`

---

# Modelos incluidos (ejemplo)
- `models/ppo_tennis_selfplay.zip`
- `models/ppo_tennis_selfplay_2M.zip`
- `models/dqn_tennis_selfplay_1M.zip`
- `models/dqn_vs_random_200k.zip`

---

