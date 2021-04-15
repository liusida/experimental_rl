import gym

def make_env(env_id, rank, seed, render, render_index=0):
    def _init():
        # only render one environment
        _render = render and rank in [render_index]

        env = gym.make(env_id, render=_render)

        assert rank < 100, "seed * 100 + rank is assuming rank <100"
        env.seed(seed*100 + rank)

        return env
    return _init