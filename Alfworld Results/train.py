import os

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor


from quadrotor.environment import QuadrotorSwarmEnv
from utils import NormalizeActionWrapper

NUM_EPISODES=100000

# Create the environment
env = QuadrotorSwarmEnv(step_limit=200, render_mode="rgb_array", camera_name="free")
env = NormalizeActionWrapper(env)
env = Monitor(env, allow_early_resets=True)
check_env(env)

env = DummyVecEnv([lambda: env])

# Set up the PPO model with default values
ppo_model = PPO("MlpPolicy", env, verbose=1)
ppo_model.learn(total_timesteps=NUM_EPISODES)

# Evaluate the PPO model
ppo_mean_reward, _ = evaluate_policy(ppo_model, env, n_eval_episodes=10)
print(f"PPO Mean Reward: {ppo_mean_reward:.2f}")

# Save the PPO model
ppo_model.save("ppo_quadrotor_swarm")

# Reset the environment before running the trained model
obs = env.reset()

for _ in range(1000):
    action, _ = ppo_model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render(mode="human")
    if done:
        obs = env.reset()

env.close()
