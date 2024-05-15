import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Create the environment
env_id = "InvertedPendulum-v4"
env = gym.make(env_id, render_mode="rgb_array")

# Wrap the environment in a vectorized environment
env = DummyVecEnv([lambda: env])

# Create the PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
total_timesteps = 50000
model.learn(total_timesteps=total_timesteps)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Render the environment with the trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    env.render(mode="human")
    if done:
        obs = env.reset()

env.close()
