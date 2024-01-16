""":mod:`gymnasium` tests."""
from torchrl.envs.libs.gym import GymEnv


def test_init_env() -> None:
    """Test :class:`GymEnv` initialization.

    Current latest version of :mod:`opencv-python` raises an error when
    instantiating a :class:`GymEnv`. Remove this test when the issue is
    resolved.
    """
    GymEnv(env_name="CartPole-v0")
