from crane_env import CraneStabilityEnv
from gymnasium.utils.env_checker import check_env


env = CraneStabilityEnv()

# Validate our custom Gymnasium environment
check_env(env)

print("Environment passes Gymnasium checks!")

obs, info = env.reset(seed=42)

print("\nInitial observation:")
print(obs)

action = env.action_space.sample()
print("\nRandom action:")
print(action)

obs, reward, terminated, truncated, info = env.step(action)

print("\nObservation after one step:")
print(obs)

print("\nReward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)
print("Info:", info)