import numpy as np
from src.acrobot_env import AcrobotEnv
from src.rrt import rrt
import time

if __name__ == "__main__":
    env = AcrobotEnv(is_render=True, dt=0.05, n_steps=10)
    actions = rrt(np.array([0, 0, 0, 0]), np.array([np.pi, 0, 0, 0]), env)

    # Execute each action 10 times but render frames separately
    env = AcrobotEnv(is_render=True, dt=0.05, n_steps=1)
    curr_state = env.reset()
    for i in range(len(actions) * 10):
        curr_state = env.step(actions[i // 10])
        env.render()
        time.sleep(0.1)
