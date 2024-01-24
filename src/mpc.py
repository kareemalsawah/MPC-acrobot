"""
Model Predictive Control (MPC) module.
"""
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

import casadi
import numpy as np

from .acrobot_env import AcrobotParams, AcrobotConstraints, get_state_derivate


@dataclass
class SimpleMPCParams:
    """
    Parameters for SimpleMPC
    """

    horizon: int = 200
    dt: float = 0.1
    solver_options: Dict[str, Any] = field(
        default_factory=lambda: {
            "ipopt.max_iter": 1000,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.acceptable_tol": 0.00001,
            "ipopt.acceptable_obj_change_tol": 0.000001,
        }
    )


class SimpleMPC:
    """
    MPC used for acrobot to reach a target state
    Assumes no obstacles in the environment
    """

    def __init__(
        self,
        target_state: np.ndarray,
        target_weights: np.ndarray,
        acrobot_params: AcrobotParams,
        acrobot_constraints: AcrobotConstraints,
        params: Optional[SimpleMPCParams] = None,
    ):
        self.target_state = target_state
        self.target_weights = target_weights
        self.acrobot_params = acrobot_params
        self.acrobot_constraints = acrobot_constraints
        self.params = params if params is not None else SimpleMPCParams()

        self.states = np.zeros((self.params.horizon + 1, 4))
        self.controls = np.zeros((self.params.horizon, 1))

        self.forward_model = None
        self.solver = None
        self.constraints = None

        self._setup()

    def _setup(self):
        """
        Setup MPC
        """
        self._build_forward_model()
        self._build_horizon()
        self._build_constraints()

    def _build_forward_model(self):
        """
        Build forward model
        """
        theta1 = casadi.SX.sym("theta1")
        theta2 = casadi.SX.sym("theta2")
        theta_dot1 = casadi.SX.sym("theta_dot1")
        theta_dot2 = casadi.SX.sym("theta_dot2")
        control_torque = casadi.SX.sym("control_torque")

        states = casadi.vertcat(theta1, theta2, theta_dot1, theta_dot2)

        derivative = get_state_derivate(states, control_torque, self.acrobot_params)
        derivative = casadi.vertcat(*derivative)
        rhs = states + self.params.dt * derivative

        self.forward_model = casadi.Function(
            "forward_model", [states, control_torque], [rhs]
        )

    def _build_horizon(self):
        """
        Build horizon
        Uses multiple shooting
        """
        actions = casadi.SX.sym("actions", (self.params.horizon, 1))
        states = casadi.SX.sym("states", (self.params.horizon + 1, 4))
        initial_state = casadi.SX.sym("initial_state", (1, 4))

        equality_constraints = [states[0, :] - initial_state]
        for i in range(self.params.horizon):
            equality_constraints.append(
                states[i + 1, :]
                - casadi.reshape(self.forward_model(states[i, :], actions[i, :]), 1, 4)
            )

        equality_constraints = casadi.vertcat(*equality_constraints)

        # objective = np.sum(
        #     [
        #         casadi.mtimes(
        #             [
        #                 (
        #                     casadi.reshape(states[i, :], 4, 1)
        #                     - self.target_state.reshape(4, 1)
        #                 ).T,
        #                 np.diag(self.target_weights),
        #                 casadi.reshape(states[i, :], 4, 1)
        #                 - self.target_state.reshape(4, 1),
        #             ]
        #         )
        #         for i in range(self.params.horizon + 1)
        #     ]
        # )
        objective = casadi.mtimes(
            [
                (
                    casadi.reshape(states[-1, :], 4, 1)
                    - self.target_state.reshape(4, 1)
                ).T,
                np.diag(self.target_weights),
                casadi.reshape(states[-1, :], 4, 1) - self.target_state.reshape(4, 1),
            ]
        )
        opt_variables = casadi.vertcat(
            casadi.reshape(states, -1, 1), casadi.reshape(actions, -1, 1)
        )

        nlp_problem = {
            "f": objective,
            "x": opt_variables,
            "g": casadi.reshape(equality_constraints, -1, 1),
            "p": initial_state,
        }
        self.solver = casadi.nlpsol(
            "solver", "ipopt", nlp_problem, self.params.solver_options
        )

    def _build_constraints(self):
        """
        Build constraints
        """
        lbg = np.zeros((self.params.horizon + 1) * 4)
        ubg = np.zeros((self.params.horizon + 1) * 4)

        # State constraints
        lbx = [
            self.acrobot_constraints.theta1[0],
            self.acrobot_constraints.theta2[0],
            self.acrobot_constraints.theta_dot1[0],
            self.acrobot_constraints.theta_dot2[0],
        ] * (self.params.horizon + 1)
        ubx = [
            self.acrobot_constraints.theta1[1],
            self.acrobot_constraints.theta2[1],
            self.acrobot_constraints.theta_dot1[1],
            self.acrobot_constraints.theta_dot2[1],
        ] * (self.params.horizon + 1)

        # Control constraints
        lbx += [self.acrobot_constraints.control_torque[0]] * self.params.horizon
        ubx += [self.acrobot_constraints.control_torque[1]] * self.params.horizon

        self.constraints = {"lbg": lbg, "ubg": ubg, "lbx": lbx, "ubx": ubx}

    def run(self, state: np.array, init_zeros: bool = False) -> float:
        """
        Run MPC

        Parameters
        ----------
        state : np.array
            Current state of the system
        init_zeros : bool, by default False
            If false, uses previous states and controls as initial guess

        Returns
        -------
        float
            Control to be applied to the system
        """
        if init_zeros:
            states = np.zeros((self.params.horizon + 1, 4))
            actions = np.zeros((self.params.horizon, 1))
        else:
            states = self.states
            actions = self.controls

        initial_params = casadi.vertcat(
            casadi.reshape(states, -1, 1), casadi.reshape(actions, -1, 1)
        )

        sol = self.solver(
            x0=initial_params,
            p=state.reshape(1, 4),
            lbg=self.constraints["lbg"],
            ubg=self.constraints["ubg"],
            lbx=self.constraints["lbx"],
            ubx=self.constraints["ubx"],
        )

        opt_variables = sol["x"].full().flatten()
        self.states = opt_variables[: (self.params.horizon + 1) * 4].reshape(
            self.params.horizon + 1, 4
        )
        self.controls = opt_variables[(self.params.horizon + 1) * 4 :].reshape(
            self.params.horizon, 1
        )

        return self.controls[0, 0]
