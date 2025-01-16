from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from racetrack_env import RacetrackEnv
import os
from custom_metrics import CustomMetricsCallback

# Set up directories
current_folder = os.path.dirname(os.path.abspath(__file__))
logs_folder = os.path.join(current_folder, "logs")
models_folder = os.path.join(current_folder, "models")

# Function to create parallel environments
def create_custom_racetrack_env():
    return RacetrackEnv()

if __name__ == "__main__":
    # User input for training configuration
    algos = ["PPO", "A2C", "SAC", "TD3"]
    print(f"Available algorithms: {', '.join(algos)}")
    algo = input("Enter the algorithm to train (PPO, A2C, SAC, TD3): ").strip().upper()
    if algo not in algos:
        raise ValueError(f"Invalid algorithm selected: {algo}")
    checkpoint_name = input("Enter checkpoint name (leave blank to start fresh): ").strip()
    run_name = input("Enter run name: ").strip()
    device = input("Enter the device to use (cuda/cpu): ").strip()
    total_timesteps = int(input("Enter total timesteps for training: ").strip())
    n_envs = int(input("Enter the number of parallel environments: ").strip())

    # Set up paths
    tensorboard_log = os.path.join(logs_folder, run_name)
    model_save_path = os.path.join(models_folder, run_name)

    print(f"Setting up {n_envs} parallel environments...")
    env = SubprocVecEnv([create_custom_racetrack_env for _ in range(n_envs)])

    model = None
    if checkpoint_name:
        checkpoint_path = os.path.join(models_folder, checkpoint_name)
        try:
            model = eval(algo).load(checkpoint_path, env=env, device=device)
            print(f"Checkpoint '{checkpoint_name}' loaded successfully. Continuing training...")
        except FileNotFoundError:
            print(f"Checkpoint '{checkpoint_name}' not found. Starting fresh...")

    if not model:
        if algo == "PPO":
            model = PPO(
                "MlpPolicy",
                env,
                verbose=2,
                tensorboard_log=tensorboard_log,
                learning_rate=4e-5,   # Smaller learning rate
                n_steps=2048,         # Larger steps per update
                gamma=0.985,          # Slightly lower gamma
                gae_lambda=0.8,       # Adjust GAE lambda
                clip_range=0.2,
                vf_coef=0.4,          # Reduce weight of value loss
                normalize_advantage=True,
                device=device,
            )
        elif algo == "A2C":
            model = A2C(
                "MlpPolicy",
                env,
                verbose=2,
                tensorboard_log=tensorboard_log,
                learning_rate=3e-4,
                n_steps=50,
                gamma=0.985,
                gae_lambda=0.95,
                max_grad_norm=0.3,
                device=device,
            )
        elif algo == "SAC":
            model = SAC(
                "MlpPolicy",
                env,
                verbose=2,
                tensorboard_log=tensorboard_log,
                learning_rate=2e-4,
                gamma=0.99,
                tau=0.005,
                ent_coef="auto",
                target_update_interval=1,
                device=device,
            )
        elif algo == "TD3":
            model = TD3(
                "MlpPolicy",
                env,
                verbose=2,
                tensorboard_log=tensorboard_log,
                learning_rate=2e-4,
                gamma=0.99,
                tau=0.005,
                train_freq=1,
                normalize_advantage=True,
                gradient_steps=1,
                device=device,
            )

    # Training
    print(f"Training {algo} for {total_timesteps} timesteps...")
    custom_callback = CustomMetricsCallback(verbose=1)
    model.learn(total_timesteps=total_timesteps, tb_log_name=run_name, callback=custom_callback)

    # Save the trained model
    model.save(model_save_path)
    print(f"Model saved successfully as {run_name}.")

    # Close environments
    env.close()
