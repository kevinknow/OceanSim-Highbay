import numpy as np

from isaacsim.core.utils.rotations import matrix_to_euler_angles, quat_to_rot_matrix


class UnderwaterHydrodynamics:
    """Minimal body-frame hydrodynamics for a neutrally buoyant ROV."""

    def __init__(
        self,
        linear_command_tau: float = 0.35,
        angular_command_tau: float = 0.25,
        linear_drag: np.ndarray = np.array([8.0, 14.0, 18.0]),
        quadratic_linear_drag: np.ndarray = np.array([6.0, 10.0, 12.0]),
        angular_drag: np.ndarray = np.array([8.0, 8.0, 6.0]),
        quadratic_angular_drag: np.ndarray = np.array([3.0, 3.0, 2.5]),
        attitude_stiffness: np.ndarray = np.array([9.0, 9.0, 0.0]),
        attitude_damping: np.ndarray = np.array([3.5, 3.5, 0.0]),
        max_attitude_error: np.ndarray = np.array([0.45, 0.45, 0.0]),
        max_angular_velocity: np.ndarray = np.array([1.2, 1.2, 1.2]),
        max_smoothed_torque_cmd: np.ndarray = np.array([5.0, 5.0, 4.0]),
        max_total_torque: np.ndarray = np.array([10.0, 10.0, 8.0]),
    ) -> None:
        self._linear_command_tau = linear_command_tau
        self._angular_command_tau = angular_command_tau
        self._linear_drag = np.array(linear_drag, dtype=np.float64)
        self._quadratic_linear_drag = np.array(quadratic_linear_drag, dtype=np.float64)
        self._angular_drag = np.array(angular_drag, dtype=np.float64)
        self._quadratic_angular_drag = np.array(quadratic_angular_drag, dtype=np.float64)
        self._attitude_stiffness = np.array(attitude_stiffness, dtype=np.float64)
        self._attitude_damping = np.array(attitude_damping, dtype=np.float64)
        self._max_attitude_error = np.array(max_attitude_error, dtype=np.float64)
        self._max_angular_velocity = np.array(max_angular_velocity, dtype=np.float64)
        self._max_smoothed_torque_cmd = np.array(max_smoothed_torque_cmd, dtype=np.float64)
        self._max_total_torque = np.array(max_total_torque, dtype=np.float64)

        self._smoothed_force_cmd_body = np.zeros(3, dtype=np.float64)
        self._smoothed_torque_cmd_body = np.zeros(3, dtype=np.float64)

    def _first_order_filter(
        self,
        current_value: np.ndarray,
        target_value: np.ndarray,
        tau: float,
        step: float,
    ) -> np.ndarray:
        if tau <= 1e-6:
            return np.array(target_value, dtype=np.float64)
        alpha = 1.0 - np.exp(-step / tau)
        return current_value + alpha * (target_value - current_value)

    def compute_wrench(
        self,
        step: float,
        world_orientation: np.ndarray,
        world_linear_velocity: np.ndarray,
        world_angular_velocity: np.ndarray,
        desired_force_cmd_body: np.ndarray,
        desired_torque_cmd_body: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rot_body_to_world = quat_to_rot_matrix(np.array(world_orientation, dtype=np.float64))
        rot_world_to_body = rot_body_to_world.T

        linear_velocity_body = rot_world_to_body @ np.array(world_linear_velocity, dtype=np.float64)
        angular_velocity_body = rot_world_to_body @ np.array(world_angular_velocity, dtype=np.float64)
        angular_velocity_body = np.clip(
            angular_velocity_body,
            -self._max_angular_velocity,
            self._max_angular_velocity,
        )

        self._smoothed_force_cmd_body = self._first_order_filter(
            self._smoothed_force_cmd_body,
            np.array(desired_force_cmd_body, dtype=np.float64),
            self._linear_command_tau,
            step,
        )
        self._smoothed_torque_cmd_body = self._first_order_filter(
            self._smoothed_torque_cmd_body,
            np.array(desired_torque_cmd_body, dtype=np.float64),
            self._angular_command_tau,
            step,
        )
        self._smoothed_torque_cmd_body = np.clip(
            self._smoothed_torque_cmd_body,
            -self._max_smoothed_torque_cmd,
            self._max_smoothed_torque_cmd,
        )

        linear_drag_body = (
            -self._linear_drag * linear_velocity_body
            - self._quadratic_linear_drag * np.abs(linear_velocity_body) * linear_velocity_body
        )
        angular_drag_body = (
            -self._angular_drag * angular_velocity_body
            - self._quadratic_angular_drag * np.abs(angular_velocity_body) * angular_velocity_body
        )

        roll, pitch, _ = matrix_to_euler_angles(rot_body_to_world, degrees=False, extrinsic=True)
        attitude_error = np.array([roll, pitch, 0.0], dtype=np.float64)
        attitude_error = np.clip(
            attitude_error,
            -self._max_attitude_error,
            self._max_attitude_error,
        )
        restoring_torque_body = (
            -self._attitude_stiffness * attitude_error
            - self._attitude_damping * angular_velocity_body
        )

        total_force_body = self._smoothed_force_cmd_body + linear_drag_body
        total_torque_body = self._smoothed_torque_cmd_body + angular_drag_body + restoring_torque_body
        total_torque_body = np.clip(
            total_torque_body,
            -self._max_total_torque,
            self._max_total_torque,
        )

        total_force_world = rot_body_to_world @ total_force_body
        total_torque_world = rot_body_to_world @ total_torque_body
        return total_force_world, total_torque_world
