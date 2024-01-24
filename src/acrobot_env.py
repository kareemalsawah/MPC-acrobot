"""
AcrobotEnv with an interface similar to the openAI Gym interface
Physical model implemented based on https://github.com/LeeLinJun/mpc-mpnet-py
Which was based on the model described in:
M. W. Spong, “Underactuated mechanical systems,” 
in Control problems in robotics and automation. Springer, 1998
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import pygame
import numpy as np

# pylint: disable=too-many-instance-attributes, too-many-locals, too-many-arguments


@dataclass
class AcrobotParams:
    """
    Physical parameters for acrobot simulations
    """

    mass: float = 1  # Mass of each link in the robot, assume both has the same mass

    # Length of links
    len_1: float = 1
    len_2: float = 1

    # Moment of inertia of each link
    mom_inert_1: float = 0.2
    mom_inert_2: float = 1

    # Center of mass of each link
    cent_mass_1: float = 0.5
    cent_mass_2: float = 0.25

    g_acc: float = 9.81  # gravity acceleration
    damping_factor: float = 0.1


@dataclass
class AcrobotConstraints:
    """
    Constraints on states and controls
    """

    theta1: Tuple[float] = (-np.pi, np.pi)
    theta2: Tuple[float] = (-np.pi, np.pi)
    theta_dot1: Tuple[float] = (-6, 6)
    theta_dot2: Tuple[float] = (-6, 6)
    control_torque: Tuple[float] = (-4, 4)


def pi_to_pi(angle: float) -> float:
    """
    Convert angle to be within [-pi, pi]

    Parameters
    ----------
    angle : float
        Angle to convert

    Returns
    -------
    angle : float
        Converted angle
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


class AcrobotEnv:
    """
    Acrobot environment

    Parameters
    ----------
    is_render : bool, by default False
        Whether to render the environment
    physical_params : Optional[AcrobotParams]
        Physical parameters of the acrobot
    constraints : Optional[AcrobotConstraints]
        Constraints on states and controls
    dt : float, by default 0.02
        Time step of the simulation
    n_steps : int, by default 1
        Number of steps to take in each call to step
    """

    def __init__(
        self,
        is_render: bool = False,
        physical_params: Optional[AcrobotParams] = None,
        constraints: Optional[AcrobotConstraints] = None,
        dt: float = 0.02,
        n_steps: int = 1,
    ):
        self.is_render = is_render
        self.physical_params = (
            physical_params if physical_params is not None else AcrobotParams()
        )
        self.constraints = (
            constraints if constraints is not None else AcrobotConstraints()
        )
        self.dt = dt
        self.n_steps = n_steps

        self.state = np.zeros(4)

        if self.is_render:
            pygame.display.init()
            self.screen = pygame.display.set_mode([500, 500])

    def reset(self) -> np.ndarray:
        """
        Reset environment
        """
        self.state = np.zeros(4).astype(float)
        if self.is_render:
            pass
        return self.state

    def step(self, action: float):
        """
        Take a step in the environment

        Parameters
        ----------
        action : float
            Torque to apply to the second joint

        Returns
        -------
        state : np.ndarray
            New state of the environment
        """
        for _ in range(self.n_steps):
            action = np.clip(
                action,
                self.constraints.control_torque[0],
                self.constraints.control_torque[1],
            )
            deriv = get_state_derivate(self.state, action, self.physical_params)
            self.state += np.array(deriv) * self.dt
        self.state[0] = pi_to_pi(self.state[0])
        self.state[1] = pi_to_pi(self.state[1])

        return self.state

    def render(self):
        """
        Render the environment
        """
        self.screen.fill((255, 255, 255))  # white background

        # Draw the first link
        x1 = 250 + 100 * np.sin(self.state[0])
        y1 = 250 + 100 * np.cos(self.state[0])
        pygame.draw.line(self.screen, (0, 0, 0), (250, 250), (x1, y1), 5)

        # Draw the second link
        x2 = x1 + 100 * np.sin(self.state[1] + self.state[0])
        y2 = y1 + 100 * np.cos(self.state[1] + self.state[0])
        pygame.draw.line(self.screen, (0, 0, 0), (x1, y1), (x2, y2), 5)

        pygame.display.flip()

        pygame.time.Clock().tick(60)  # run at 60 fps

    def close(self):
        """
        Close the environment
        """
        if self.is_render:
            pygame.quit()

    def interactive(self):
        """
        Interactive playing of the acrobot
        """
        if not self.is_render:
            raise ValueError("Cannot play without rendering")

        key_left = False
        key_right = False

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        key_left = True
                    if event.key == pygame.K_RIGHT:
                        key_right = True
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT:
                        key_left = False
                    if event.key == pygame.K_RIGHT:
                        key_right = False
            if key_left:
                self.step(self.constraints.control_torque[0])
            elif key_right:
                self.step(self.constraints.control_torque[1])
            else:
                self.step(0)

            self.render()


def get_state_derivate(
    state: np.array,
    action: float,
    physical_params: AcrobotParams,
) -> np.ndarray:
    """
    Derivative of the state
    Taken from https://github.com/LeeLinJun/mpc-mpnet-py

    Parameters
    ----------
    state: np.ndarray
        State of the environment, shape=(4)
    action : float
        Torque to apply to the second joint
        Action must be within constraints.control_torque
    physical_params : AcrobotParams
        Physical parameters of the acrobot

    Returns
    -------
    deriv : List
        Derivative of the state, shape=(4)
    """
    theta1 = state[0] - np.pi / 2

    m = physical_params.mass
    l = physical_params.len_1
    l2 = physical_params.len_2
    lc = physical_params.cent_mass_1
    lc2 = physical_params.cent_mass_2
    mom_inert_1 = physical_params.mom_inert_1
    mom_inert_2 = physical_params.mom_inert_2
    g = physical_params.g_acc
    damp_factor = physical_params.damping_factor

    d11 = (
        m * lc2
        + m * (l2 + lc2 + 2 * l * lc * np.cos(state[1]))
        + mom_inert_1
        + mom_inert_2
    )
    d22 = m * lc2 + mom_inert_2
    d12 = m * (lc2 + l * lc * np.cos(state[1])) + mom_inert_2
    d21 = d12

    c1 = -m * l * lc * state[3] * state[3] * np.sin(state[1]) - (
        2 * m * l * lc * state[2] * state[3] * np.sin(state[1])
    )
    c2 = m * l * lc * state[2] * state[2] * np.sin(state[1])
    g1 = (m * lc + m * l) * g * np.cos(theta1) + (
        m * lc * g * np.cos(theta1 + state[1])
    )
    g2 = m * lc * g * np.cos(theta1 + state[1])

    deriv = [0, 0, 0, 0]
    deriv[0] = state[2]
    deriv[1] = state[3]

    u2 = action - damp_factor * state[3]
    u1 = -1 * damp_factor * state[2]
    deriv[2] = (d22 * (u1 - c1 - g1) - d12 * (u2 - c2 - g2)) / (d11 * d22 - d12 * d21)
    deriv[3] = (d11 * (u2 - c2 - g2) - d21 * (u1 - c1 - g1)) / (d11 * d22 - d12 * d21)
    return deriv
