from gymnasium.utils.env_checker import check_env

from crane_env import CraneStabilityEnv


print("=" * 60)
print("FINAL GYMNASIUM ENVIRONMENT VALIDATION")
print("=" * 60)

env = CraneStabilityEnv()

try:
    check_env(
        env,
        skip_render_check=True
    )

    print("\nVALIDATION PASSED")
    print("CraneStabilityEnv follows Gymnasium API requirements.")

except Exception as error:

    print("\nVALIDATION FAILED")
    print(type(error).__name__)
    print(error)

finally:
    env.close()