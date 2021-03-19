import pytest

def test_train():
    from stable_baselines3 import A2C
    model = A2C('MlpPolicy', 'CartPole-v1').learn(1000)
    assert True