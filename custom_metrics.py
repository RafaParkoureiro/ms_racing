import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CustomMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomMetricsCallback, self).__init__(verbose)
        self.metrics = {
            "episode_length": [],
            "episode_reward": [],
            "proximity_time": [],
            "on_track_time": [],
            "off_track_time": [],
            "collision": [],
        }

    def _on_step(self) -> bool:
        # Collect environment info
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode_length" in info:
                self.metrics["episode_length"].append(info["episode_length"])
                self.metrics["episode_reward"].append(info["episode_reward"])
                self.metrics["proximity_time"].append(info["proximity_time"])
                self.metrics["on_track_time"].append(info["on_track_time"])
                self.metrics["off_track_time"].append(info["off_track_time"])
                self.metrics["collision"].append(info["collision"])

        # Log every 50 steps
        if self.n_calls % 50 == 0:
            if self.metrics["episode_length"]:
                self.logger.record(
                    "custom/mean_episode_length",
                    np.mean(self.metrics["episode_length"]),
                )
                self.logger.record(
                    "custom/mean_episode_reward",
                    np.mean(self.metrics["episode_reward"]),
                )
                self.logger.record(
                    "custom/mean_proximity_time",
                    np.mean(self.metrics["proximity_time"]),
                )
                self.logger.record(
                    "custom/mean_on_track_time",
                    np.mean(self.metrics["on_track_time"]),
                )
                self.logger.record(
                    "custom/mean_off_track_time",
                    np.mean(self.metrics["off_track_time"]),
                )
                self.logger.record(
                    "custom/collision_percentage",
                    np.sum(self.metrics["collision"]) * 100 / 250,
                )
            # Clear the metrics
            self.metrics = {key: [] for key in self.metrics}
        return True
