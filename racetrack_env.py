from highway_env.envs.common.abstract import AbstractEnv
from highway_env.vehicle.behavior import IDMVehicle
from track_builder import make_road
from highway_env.road.lane import CircularLane
import numpy as np


class RacetrackEnv(AbstractEnv):
    @classmethod
    def default_config(cls) -> dict:
        '''
        Environment configs (most important/ most changed)
        -duration: episode duration in ms
        -collision_reward: penalty for colliding
        -lane_centering_cost:
        -lane_centering_reward:
        -controlled_vehicles: agent's vehicles
        -other_vehicles: bot vehicles
        -on_road_reward: reward for being on the track

        Custom configs:
        -proximity_penalty: penalty for getting too close to front vehicle
        -lane_change_reward: reward for changing lane if too close to front vehicle
        -off_track_penalty_rate: incremental penalty for each off-track step
        -off_track_penalty_threshold: threshold for truncating the episode
        '''      
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "OccupancyGrid",
                    "features": ["presence", "on_road"],
                    "grid_size": [[-25, 25], [-25, 25]],
                    "grid_step": [1, 1],
                    "as_image": False,
                    "align_to_vehicle_axes": True,
                },
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": False,
                    "lateral": True,
                    "target_speeds": [0, 5, 10],
                },
                "simulation_frequency": 15,
                "policy_frequency": 10,
                "duration": 60,
                "collision_reward": -500,
                "lane_centering_cost": 1.5,
                "lane_centering_reward": 2,
                "action_reward": -0.3,
                "controlled_vehicles": 1,
                "other_vehicles": 5,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.5],
                "on_road_reward": 1,
                "proximity_penalty": -5,
                "lane_change_reward": 2.5,
                "off_track_penalty": -1,
                "off_track_threshold": 6
            }
        )
        return config
    
    def _init_metrics(self):
        """
        Initialize metrics for the episode.
        """
        self.episode_reward = 0
        self.episode_length = 0
        self.proximity_time = 0
        self.on_track_time = 0
        self.off_track_time = 0
        self.off_track = 0

    def _update_metrics(self, reward, proximity_penalty: float):
        """
        Custom metrics function.
        """
        self.episode_reward += reward
        self.episode_length += 1 / self.config["policy_frequency"]
        if self.vehicle.on_road:
            self.on_track_time += 1 / self.config["policy_frequency"]
        else:
            self.off_track_time += 1 / self.config["policy_frequency"]
        if proximity_penalty != 0:
            self.proximity_time += 1 / self.config["policy_frequency"]

    def _reward(self, action: np.ndarray) -> float:
        rewards = self._rewards(action)
        total_reward = sum(rewards.values())
        self._update_metrics(total_reward,rewards.get("proximity_penalty", 0),)
        return total_reward

    def _rewards(self, action: np.ndarray) -> dict[str, float]:
        '''
        Custom rewards function.
        Applies reward for lane changing when collision is eminent.
        Penalty for proximity to front car.
        Off track penalty.
        '''
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        front_vehicle, distance_to_front = self._get_closest_vehicle_in_lane(self.vehicle)
        lane_change_reward = 0
        proximity_penalty = 0
        off_track_penalty = 0
        lateral_action = action[1] if len(action) > 1 else 0

        if front_vehicle and self.vehicle.lane_index == front_vehicle.lane_index:
            if distance_to_front <= 10:
                # "Semi Filter" of
                if lateral_action != 0:  # Reward only for lateral moves
                    lane_change_reward = 5  # Reward lane change
                proximity_penalty = 10 / (1 + distance_to_front)
        
        if not self.vehicle.on_road:
            self.off_track += 1 / self.config["policy_frequency"]
            off_track_penalty = self.off_track
        else:
            self.off_track = 0
            off_track_penalty = 0

        return {
            "lane_centering_reward": (1 / (1 + self.config["lane_centering_cost"] * lateral**2)) * self.config["lane_centering_reward"],
            "action_reward": np.linalg.norm(action) * self.config["action_reward"],
            "on_road_reward": self.vehicle.on_road * self.config["on_road_reward"],
            "proximity_penalty": proximity_penalty * self.config["proximity_penalty"],
            "lane_change_reward": lane_change_reward * self.config["lane_change_reward"],
            "collision_reward": self.vehicle.crashed * self.config["collision_reward"],
            "off_track_penalty": off_track_penalty * self.config["off_track_penalty"],
        }
        
    def _get_closest_vehicle_in_lane(self, vehicle):
        """
        Get the distance to the closest vehicle in the same lane, ignoring negative distances.
        """
        lane_index = vehicle.lane_index
        vehicles_in_lane = [
            v for v in self.road.vehicles
            if v.lane_index == lane_index and v != vehicle and v.on_road
        ]

        if not vehicles_in_lane:
            return None, float("inf")

        closest_vehicle = None
        min_distance = float("inf")

        for other_vehicle in vehicles_in_lane:
            raw_distance = self._longitudinal_distance(vehicle, other_vehicle)
            if raw_distance >= 0 and raw_distance < min_distance:
                closest_vehicle = other_vehicle
                min_distance = raw_distance

        return closest_vehicle, min_distance

    def _longitudinal_distance(self, vehicle1, vehicle2):
        """
        Calculate the longitudinal distance between two vehicles along the same lane.
        Handles wrapping effects for circular and straight lanes.
        """
        lane = self.road.network.get_lane(vehicle1.lane_index)
        
        # Ensure both vehicles are on the same lane
        if vehicle1.lane_index != vehicle2.lane_index:
            return float("inf")

        pos1 = lane.local_coordinates(vehicle1.position)[0]
        pos2 = lane.local_coordinates(vehicle2.position)[0]
        lane_length = lane.length
        raw_distance = pos2 - pos1

        if isinstance(lane, CircularLane):
            if raw_distance > lane_length / 2:
                raw_distance -= lane_length
            elif raw_distance < -lane_length / 2:
                raw_distance += lane_length

        return raw_distance
    
    def _info(self, obs, action=None):
        """
        Return additional metrics in the info dictionary.
        """
        info = super()._info(obs, action)
        info.update({
            "episode_reward": self.episode_reward,
            "episode_length": self.episode_length,
            "proximity_time": self.proximity_time,
            "on_track_time": self.on_track_time,
            "off_track_time": self.off_track_time,
        })
        return info

    def _is_terminated(self) -> bool:
        if self.vehicle.crashed == True:
            return True
        return False

    def _is_truncated(self) -> bool:
        if self.time >= self.config["duration"] or self.off_track_time >= self.config["off_track_threshold"]:
            return True
        return False

    def _is_terminal(self) -> bool:
        return self._is_terminated() or self._is_truncated()

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()
        self._init_metrics()

    def _make_road(self) -> None:
        self.road = make_road(self.np_random, show_trajectories=self.config["show_trajectories"])

    def _make_vehicles(self) -> None:
        rng = self.np_random

        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            lane_index = (
                ("a", "b", rng.integers(0, 2))
                if i == 0
                else self.road.network.random_lane_index(rng)
            )
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(
                self.road, lane_index, speed=10, longitudinal=rng.uniform(20, 50)
            )
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

        if self.config["other_vehicles"] > 0:
            vehicle = IDMVehicle.make_on_lane(
                self.road,
                ("b", "c", lane_index[-1]),
                longitudinal=rng.uniform(
                    low=0, high=self.road.network.get_lane(("b", "c", 0)).length
                ),
                speed=6 + rng.uniform(high=2),
            )
            self.road.vehicles.append(vehicle)

            for _ in range(rng.integers(0, self.config["other_vehicles"])):
                random_lane_index = self.road.network.random_lane_index(rng)
                vehicle = IDMVehicle.make_on_lane(
                    self.road,
                    random_lane_index,
                    longitudinal=rng.uniform(
                        low=0, high=self.road.network.get_lane(random_lane_index).length
                    ),
                    speed=6 + rng.uniform(high=2),
                )
                for v in self.road.vehicles:
                    if np.linalg.norm(vehicle.position - v.position) < 20:
                        break
                else:
                    self.road.vehicles.append(vehicle)
