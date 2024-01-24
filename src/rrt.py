"""
RRT implemented for acrobot
"""
from typing import List
import numpy as np
from .acrobot_env import AcrobotEnv


class Node:
    """
    Node class for RRT
    Just acts as a container for state, action, and parent
    """

    def __init__(
        self,
        state: np.ndarray,
        action: float = None,
        parent: "Node" = None,
    ):
        self.state = state
        self.action = action
        self.parent = parent


def rrt(
    start_config: np.ndarray,
    target_config: np.ndarray,
    env: AcrobotEnv,
    max_iters: int = 100000,
    target_threshold: float = 0.6,
):
    """
    RRT algorithm

    Parameters
    ----------
    start_config : np.ndarray
        Starting configuration of the acrobot
    target_config : np.ndarray
        Target configuration of the acrobot
    env : AcrobotEnv
        Acrobot environment
    max_iters : int
        Maximum number of iterations
    target_threshold : float
        Threshold distance to target (l2 norm)

    Returns
    -------
    list
        List of actions to reach target
        Empty list if target is not reached
    """
    current_nodes: List[Node] = [Node(start_config.astype(float))]

    for _ in range(max_iters):
        # Select random state
        rand_idx = np.random.randint(len(current_nodes))
        env.state = np.copy(current_nodes[rand_idx].state)

        # Create new next state
        random_action = np.random.uniform(-4, 4)
        new_state = env.step(random_action)
        new_node = Node(new_state, random_action, current_nodes[rand_idx])
        current_nodes.append(new_node)

        # Check if new state is close enough to target
        if np.linalg.norm(new_node.state - target_config) < target_threshold:
            # Extract path
            curr_node = current_nodes[-1]
            actions = []
            while curr_node.parent is not None:
                actions.append(curr_node.action)
                curr_node = curr_node.parent

            return actions[::-1]

    return []
