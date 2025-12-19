from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


@dataclass
class EpisodeStats:
    timesteps: int
    ep_len: int
    ep_return: float
    events: int
    steps_per_event_mean: float
    steps_per_event_median: float
    quick_event_rate: float
    pos_event_rate: float


# Callback para registrar métricas por episodio y guardar npz/plots al final
# Evento = reward != 0
# steps_per_event_* mide pasos entre eventos
# quick_event_rate mide proporción de eventos con pasos < quick_threshold
# pos_event_rate mide proporción de eventos con reward > 0
class TennisMetricsCallback(BaseCallback):
    def __init__(
        self,
        quick_threshold: int = 50,
        plot_path: str = "plots/dqn_tennis_metrics.npz",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.quick_threshold = quick_threshold
        self.plot_path = plot_path

        # Estado por env
        self._steps_since_event: Optional[np.ndarray] = None
        self._ep_len: Optional[np.ndarray] = None
        self._ep_return: Optional[np.ndarray] = None
        self._ep_events_steps: Optional[List[List[int]]] = None
        self._ep_pos_events: Optional[np.ndarray] = None
        self._ep_total_events: Optional[np.ndarray] = None

        # Historial para guardado y plots
        self.history: List[EpisodeStats] = []

    def _init_storage(self):
        n_envs = self.training_env.num_envs
        self._steps_since_event = np.zeros(n_envs, dtype=np.int32)
        self._ep_len = np.zeros(n_envs, dtype=np.int32)
        self._ep_return = np.zeros(n_envs, dtype=np.float32)
        self._ep_events_steps = [[] for _ in range(n_envs)]
        self._ep_pos_events = np.zeros(n_envs, dtype=np.int32)
        self._ep_total_events = np.zeros(n_envs, dtype=np.int32)

    def _on_training_start(self) -> None:
        self._init_storage()

    def _on_step(self) -> bool:
        rewards = np.array(self.locals.get("rewards"))
        dones = np.array(self.locals.get("dones"))

        if self._steps_since_event is None:
            self._init_storage()

        # Actualiza acumuladores por env
        for i in range(len(rewards)):
            self._steps_since_event[i] += 1
            self._ep_len[i] += 1
            self._ep_return[i] += float(rewards[i])

            # Detecta evento y registra distancia entre eventos
            if rewards[i] != 0.0:
                self._ep_events_steps[i].append(int(self._steps_since_event[i]))
                self._steps_since_event[i] = 0
                self._ep_total_events[i] += 1
                if rewards[i] > 0.0:
                    self._ep_pos_events[i] += 1

            # Cierra episodio y registra métricas
            if dones[i]:
                ev_steps = self._ep_events_steps[i]
                if len(ev_steps) > 0:
                    ev_arr = np.array(ev_steps, dtype=np.float32)
                    mean_steps = float(ev_arr.mean())
                    median_steps = float(np.median(ev_arr))
                    quick_rate = float((ev_arr < self.quick_threshold).mean())
                    pos_rate = float(self._ep_pos_events[i] / max(1, self._ep_total_events[i]))
                    events = int(self._ep_total_events[i])
                else:
                    mean_steps = 0.0
                    median_steps = 0.0
                    quick_rate = 0.0
                    pos_rate = 0.0
                    events = 0

                # Log a TensorBoard
                self.logger.record("tennis/ep_len", int(self._ep_len[i]))
                self.logger.record("tennis/ep_return", float(self._ep_return[i]))
                self.logger.record("tennis/events_per_ep", events)
                self.logger.record("tennis/steps_per_event_mean", mean_steps)
                self.logger.record("tennis/steps_per_event_median", median_steps)
                self.logger.record("tennis/quick_event_rate", quick_rate)
                self.logger.record("tennis/pos_event_rate", pos_rate)

                # Guarda en historial
                self.history.append(
                    EpisodeStats(
                        timesteps=int(self.num_timesteps),
                        ep_len=int(self._ep_len[i]),
                        ep_return=float(self._ep_return[i]),
                        events=events,
                        steps_per_event_mean=mean_steps,
                        steps_per_event_median=median_steps,
                        quick_event_rate=quick_rate,
                        pos_event_rate=pos_rate,
                    )
                )

                # Resetea acumuladores del episodio
                self._steps_since_event[i] = 0
                self._ep_len[i] = 0
                self._ep_return[i] = 0.0
                self._ep_events_steps[i].clear()
                self._ep_pos_events[i] = 0
                self._ep_total_events[i] = 0

        return True

    def save_npz(self):
        os.makedirs(os.path.dirname(self.plot_path), exist_ok=True)

        t = np.array([h.timesteps for h in self.history], dtype=np.int32)
        ep_return = np.array([h.ep_return for h in self.history], dtype=np.float32)
        ep_len = np.array([h.ep_len for h in self.history], dtype=np.int32)
        events = np.array([h.events for h in self.history], dtype=np.int32)
        spe_mean = np.array([h.steps_per_event_mean for h in self.history], dtype=np.float32)
        spe_median = np.array([h.steps_per_event_median for h in self.history], dtype=np.float32)
        quick = np.array([h.quick_event_rate for h in self.history], dtype=np.float32)
        pos = np.array([h.pos_event_rate for h in self.history], dtype=np.float32)

        np.savez(
            self.plot_path,
            timesteps=t,
            ep_return=ep_return,
            ep_len=ep_len,
            events=events,
            steps_per_event_mean=spe_mean,
            steps_per_event_median=spe_median,
            quick_event_rate=quick,
            pos_event_rate=pos,
        )

    def save_plot_png(self, png_path: str = "plots/dqn_tennis_metrics.png"):
        import matplotlib.pyplot as plt

        if len(self.history) == 0:
            print("No hay episodios en history; no se generó plot.")
            return

        os.makedirs(os.path.dirname(png_path), exist_ok=True)

        t = np.array([h.timesteps for h in self.history], dtype=np.int32)
        spe = np.array([h.steps_per_event_mean for h in self.history], dtype=np.float32)
        quick = np.array([h.quick_event_rate for h in self.history], dtype=np.float32)
        ret = np.array([h.ep_return for h in self.history], dtype=np.float32)

        # Plot steps_per_event_mean
        plt.figure()
        plt.plot(t, spe)
        plt.title("steps_per_event_mean")
        plt.xlabel("timesteps")
        plt.ylabel("steps/event")
        plt.tight_layout()
        plt.savefig(png_path.replace(".png", "_steps_per_event.png"))
        plt.close()

        # Plot quick_event_rate
        plt.figure()
        plt.plot(t, quick)
        plt.title(f"quick_event_rate (< {self.quick_threshold} steps)")
        plt.xlabel("timesteps")
        plt.ylabel("rate")
        plt.tight_layout()
        plt.savefig(png_path.replace(".png", "_quick_rate.png"))
        plt.close()

        # Plot episode return
        plt.figure()
        plt.plot(t, ret)
        plt.title("episode return")
        plt.xlabel("timesteps")
        plt.ylabel("return")
        plt.tight_layout()
        plt.savefig(png_path.replace(".png", "_return.png"))
        plt.close()
