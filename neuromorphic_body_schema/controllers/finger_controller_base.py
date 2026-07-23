import numpy as np
import threading
from dataclasses import dataclass

@dataclass
class FingerControllerParams:
    def __init__(self, finger_force_min, finger_force_max, finger_torque_max):
        self.finger_force_min = finger_force_min
        self.finger_force_max = finger_force_max
        self.finger_torque_max = finger_torque_max
        return

class FingerController:
    """Force/torque controller for the index finger.

    Manual commands come from ``FingerControlPanel`` (IJKL keys).
    To implement sensor-driven control override ``update()``, or inject
    commands directly via ``set_joint_torques()`` / ``press()``.
    """

    _ACTION_VECTORS: dict = {
        "left":    np.array([-1.0,  0.0]),
        "right":   np.array([ 1.0,  0.0]),
        "forward": np.array([ 0.0,  1.0]),
        "back":    np.array([ 0.0, -1.0]),
    }
    _KEY_TO_ACTION: dict = {
        "i": "forward", "up":   "forward",
        "k": "back",    "down": "back",
        "j": "left",    "left": "left",
        "l": "right",   "right":"right",
    }

    def __init__(self, controller_params : FingerControllerParams) -> None:
        self._lock = threading.Lock()
        self._pressed_actions: set = set()
        self._joint_torques = np.zeros(2, dtype=float)
        self._cartesian_force = np.zeros(2, dtype=float)
        self._force_magnitude = 0.5 * controller_params.finger_force_max
        self._controller_params = controller_params

    def press(self, action: str) -> None:
        with self._lock:
            self._pressed_actions.add(action)

    def release(self, action: str) -> None:
        with self._lock:
            self._pressed_actions.discard(action)

    def stop(self) -> None:
        with self._lock:
            self._pressed_actions.clear()
            self._joint_torques[:] = 0.0
            self._cartesian_force[:] = 0.0

    def set_force_magnitude(self, force_magnitude: float) -> None:
        """Set the magnitude used for I/J/K/L motion commands."""
        with self._lock:
            self._force_magnitude = float(np.clip(force_magnitude, 0.0, self._controller_params.finger_force_max))

    def set_cartesian_force(self, fx: float, fy: float) -> None:
        """Set the commanded XY force directly in world frame."""
        with self._lock:
            self._cartesian_force[0] = np.clip(fx, self._controller_params.finger_force_min, self._controller_params.finger_force_max)
            self._cartesian_force[1] = np.clip(fy, self._controller_params.finger_force_min, self._controller_params.finger_force_max)

    def set_joint_torques(self, proximal: float, distal: float) -> None:
        """Set finger joint torques directly (N·m).  For controller use."""
        with self._lock:
            self._joint_torques[:] = [
                np.clip(proximal, -self._controller_params.finger_torque_max, self._controller_params.finger_force_max),
                np.clip(distal,   -self._controller_params.finger_force_max, self._controller_params.finger_torque_max),
            ]

    def get_cartesian_force(self) -> np.ndarray:
        """Return commanded XY force vector [N] in world frame."""
        with self._lock:
            if np.any(self._cartesian_force):
                return self._cartesian_force.copy()
            if not self._pressed_actions:
                return np.zeros(2)
            direction = np.zeros(2, dtype=float)
            for action in self._pressed_actions:
                direction += self._ACTION_VECTORS[action]
            norm = np.linalg.norm(direction)
            if norm > 0.0:
                direction /= norm
            return np.clip(direction * self._force_magnitude, self._controller_params.finger_force_min, self._controller_params.finger_torque_max)

    def get_joint_torques(self) -> np.ndarray:
        """Return commanded joint torques [N·m] as [proximal, distal]."""
        with self._lock:
            return self._joint_torques.copy()

    @classmethod
    def key_to_action(cls, key: str):
        return cls._KEY_TO_ACTION.get(key.lower())

    def update(
        self,
        taxel_readings: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
    ) -> None:
        """Override to implement a closed-loop sensor-driven controller.

        Called once per simulation step *after* ``mj_step``.

        Args:
            taxel_readings: shape (12,) raw contact pressure per taxel [N]
            joint_pos:      shape (2,)  [proximal, index_2] positions [rad]
            joint_vel:      shape (2,)  [proximal, index_2] velocities [rad/s]

        Use ``set_joint_torques()`` and/or ``press()`` / ``release()``
        inside this method to command the finger.
        """
        return