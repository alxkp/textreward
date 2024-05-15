import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

import mujoco

class QuadrotorSwarmEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, waypoint_threshold = 0.4, completion_bonus=10.0, step_limit=1000, **kwargs):
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            os.path.join(os.getcwd(), "quadrotor/assets/task.xml"),
            5,
            observation_space=observation_space,
            width=1200,
            height=900,
            default_camera_config={"distance": 10.0},
            **kwargs
        )
        self.waypoint_threshold = waypoint_threshold
        self.current_waypoint_index = 0
        self.completion_bonus = completion_bonus
        self.step_limit = step_limit
        self.step_counter = 0

        # Get the initial positions of the waypoints from the MuJoCo model
        self.waypoints = []
        for i in range(1, 12):  # Assuming you have 11 waypoints (wp1 to wp11)
            waypoint_name = f"wp{i}"
            waypoint_pos = self.model.body(waypoint_name).pos
            self.waypoints.append(waypoint_pos)



    def _get_obs(self):
        data = [
                self.data.body("x2_1").xpos,  # Position of quadrotor 1
                self.data.sensor("linear_velocity_1").data, # Linear velocity of quadrotor 1
                self.data.sensor("angular_velocity_1").data, # Angular velocity of quadrotor 1
                self.data.sensor("Control_1").data,  # Control inputs
                self.data.body("x2_1").xquat[:2],  # Orientation of quadrotor 1 (first 2 elements of quaternion)

                # ------------------------

                self.data.body("x2_2").xpos,  # Position of quadrotor 2
                self.data.sensor("linear_velocity_2").data, # Linear velocity of quadrotor 2
                self.data.sensor("angular_velocity_2").data, # Angular velocity of quadrotor 2
                self.data.sensor("Control_2").data,  # Control inputs
                self.data.body("x2_2").xquat[:2],  # Orientation of quadrotor 2 (first 2 elements of quaternion)
            ]
        return np.concatenate(data)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        self.step_counter += 1
        obs = self._get_obs()
        reward = self._get_reward(obs)
        done = self._get_done(obs)
        info = self._get_info(obs)
        return obs, reward, done, False, info

    def _get_reward(self, obs):
        # Get the current positions of quadrotor 1 and quadrotor 2
        quad1_pos = obs[:3]
        quad2_pos = obs[16:19]

        # Get the position of the current waypoint
        current_waypoint = self.waypoints[self.current_waypoint_index]

        # Calculate the linear distances between quadrotors and the current waypoint
        distance1 = np.linalg.norm(quad1_pos - current_waypoint)
        distance2 = np.linalg.norm(quad2_pos - current_waypoint)

        # Calculate the reward based on the distances
        reward = -(distance1 + distance2)

        # Check if both quadrotors have reached the current waypoint
        if distance1 < self.waypoint_threshold and distance2 < self.waypoint_threshold:
            # Increment the current waypoint index
            self.current_waypoint_index += 1

            # Check if all waypoints have been reached
            if self.current_waypoint_index >= len(self.waypoints):
                # Reset the current waypoint index
                self.current_waypoint_index = 0
                # Optionally, you can add a bonus reward for completing all waypoints
                reward += self.completion_bonus

        return reward

    def _get_done(self, obs):
        # TODO: Add termination beyond step count limits
        done = self.step_counter >= self.step_limit
        return done

    def _get_info(self, obs):
        info = {}
        return info

    def reset_model(self):
        # Reset the state of the environment to an initial state and reset state counter
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        obs = self._get_obs()
        self.step_counter = 0
        return obs

if __name__ == "__main__":
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = QuadrotorSwarmEnv(step_limit=10, render_mode="rgb_array", camera_name="free")

    env = DummyVecEnv([lambda: env])

    obs = env.reset()

    while True:
        action = env.action_space.sample().reshape(1, -1)
        obs, reward, done, info = env.step(action)
        env.render(mode="human")
        if done:
            obs = env.reset()

    print("Simulation finished")
    env.close()
