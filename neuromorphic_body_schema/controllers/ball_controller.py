import numpy as np
import threading

class BallController:
    """Tracks currently held movement directions for the floor ball."""

    _ACTION_VECTORS = {
        "left": np.array([-1.0, 0.0]),
        "right": np.array([1.0, 0.0]),
        "forward": np.array([0.0, 1.0]),
        "back": np.array([0.0, -1.0]),
    }

    _KEY_TO_ACTION = {
        "a": "left",
        "left": "left",
        "d": "right",
        "right": "right",
        "w": "forward",
        "up": "forward",
        "s": "back",
        "down": "back",
    }

    def __init__(self, ball_speed_default, pattern_distance_default) -> None:
        self._lock = threading.Lock()
        self._pressed_actions: set[str] = set()
        self._speed = ball_speed_default
        self._pattern_distance = pattern_distance_default
        self._pattern_axis: str | None = None
        self._pattern_direction = 1.0
        self._pattern_center = np.zeros(2, dtype=float)
        self._reset_requested = False

    def press(self, action: str) -> None:
        with self._lock:
            self._pattern_axis = None
            self._pressed_actions.add(action)

    def release(self, action: str) -> None:
        with self._lock:
            self._pressed_actions.discard(action)

    def stop(self) -> None:
        with self._lock:
            self._pressed_actions.clear()
            self._pattern_axis = None

    def request_reset(self) -> None:
        with self._lock:
            self._pressed_actions.clear()
            self._pattern_axis = None
            self._reset_requested = True

    def consume_reset_request(self) -> bool:
        with self._lock:
            requested = self._reset_requested
            self._reset_requested = False
            return requested

    def set_speed(self, speed: float) -> None:
        clamped_speed = min(max(speed, self._pattern_distance), self._speed)
        with self._lock:
            self._speed = clamped_speed

    def speed(self) -> float:
        with self._lock:
            return self._speed

    def pattern_speed(self) -> float:
        return self._speed

    def set_pattern_distance(self, distance: float) -> None:
        with self._lock:
            self._pattern_distance = max(0.0, distance)

    def pattern_distance(self) -> float:
        with self._lock:
            return self._pattern_distance

    def set_pattern_center(self, position: np.ndarray) -> None:
        with self._lock:
            self._pattern_center = np.asarray(position, dtype=float).copy()

    def start_pattern(self, axis: str) -> None:
        with self._lock:
            self._pressed_actions.clear()
            self._pattern_axis = axis
            self._pattern_direction = 1.0

    def stop_pattern(self) -> None:
        with self._lock:
            self._pattern_axis = None

    def step(self, dt: float, position: np.ndarray) -> np.ndarray:
        with self._lock:
            if self._pattern_axis is not None:
                direction = np.array([self._pattern_direction, 0.0], dtype=float)
                axis_index = 0
                if self._pattern_axis == "y":
                    direction = np.array([0.0, self._pattern_direction], dtype=float)
                    axis_index = 1

                displacement = position[axis_index] - self._pattern_center[axis_index]
                if displacement >= self._pattern_distance:
                    self._pattern_direction = -1.0
                    direction *= -1.0
                elif displacement <= -self._pattern_distance:
                    self._pattern_direction *= -1.0
                    direction *= -1.0

                return direction

            if not self._pressed_actions:
                return np.zeros(2, dtype=float)

            direction = np.zeros(2, dtype=float)
            for action in self._pressed_actions:
                direction += self._ACTION_VECTORS[action]

            norm = np.linalg.norm(direction)
            if norm == 0.0:
                return direction
            return direction / norm

    @classmethod
    def key_to_action(cls, key: str) -> str | None:
        return cls._KEY_TO_ACTION.get(key.lower())