"""`gymnasium <https://gymnasium.farama.org/>`_ tests."""

from torchrl.envs.libs.gym import GymEnv


def test_init_env() -> None:
    """Test initialization of a ``gymnasium`` environment.

    Current latest version of
    `opencv-python <https://github.com/opencv/opencv-python>`_ raises an
    error when instantiating a ``gymnasium`` environment.
    Remove this test when the issue is resolved.
    """
    GymEnv(env_name="CartPole-v0")
