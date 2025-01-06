from stable_baselines3 import PPO, A2C, SAC, TD3
import os
import numpy as np
from racetrack_env import RacetrackEnv
from gymnasium.envs.registration import EnvSpec

# Set up directories
current_folder = os.path.dirname(os.path.abspath(__file__))
models_folder = os.path.join(current_folder, "models")

# List available models
def list_models():
    models = [f for f in os.listdir(models_folder) if os.path.isfile(os.path.join(models_folder, f))]
    return models

# Function to create a custom Racetrack environment
def create_custom_racetrack_env():
    return RacetrackEnv(render_mode="rgb_array")

if __name__ == "__main__":
    # List models and prompt for selection
    models = list_models()
    if not models:
        print("No models found in the MODELS directory.")
        exit()
    
    print("Available models:")
    for idx, model_name in enumerate(models, start=1):
        print(f"{idx}. {model_name}")
    
    try:
        choice = int(input("Select a model by entering the corresponding number: ").strip()) - 1
        if choice < 0 or choice >= len(models):
            raise ValueError("Invalid selection.")
    except ValueError as e:
        print(f"Error: {e}")
        exit()
    
    selected_model = models[choice]
    model_path = os.path.join(models_folder, selected_model)

    # Prompt for algorithm type
    algos = {"PPO": PPO, "A2C": A2C, "SAC": SAC, "TD3": TD3}
    algo = input("Enter the algorithm type (PPO, A2C, SAC, TD3): ").strip().upper()
    if algo not in algos:
        print(f"Algorithm type '{algo}' not recognized.")
        exit()

    # Prompt for device
    device = input("Enter the device to use (cuda/cpu): ").strip().lower()
    if device not in ["cuda", "cpu"]:
        print(f"Device '{device}' not recognized. Please enter 'cuda' or 'cpu'.")
        exit()

    # Set up environment
    env = create_custom_racetrack_env()
    env.spec = EnvSpec(id="RacetrackEnv-v0")  # Mock spec for compatibility

    # Load the model
    try:
        model = algos[algo].load(model_path, env=env, device=device)
        print(f"Model '{selected_model}' loaded successfully on {device}.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    # Number of episodes for visualization
    try:
        n_episodes = int(input("Enter the number of episodes to visualize: ").strip())
    except ValueError:
        print("Invalid input. Please enter an integer.")
        exit()
    
    episode_rewards = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        step=0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step+=1
            #rewards_truncated = {key: round(value, 3) if isinstance(value, (int, float)) else value for key, value in info.get("rewards", {}).items()}
            print("STEP", step, "(reward",reward,"):",info.get("rewards"))
            if done:
                print(f"Episode ended: Terminated={terminated}, Truncated={truncated}")
            env.render()  # Visualize the environment

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    # Print average statistics
    print(f"\nAverage reward across {n_episodes} episodes: {np.mean(episode_rewards)}")
    env.close()
