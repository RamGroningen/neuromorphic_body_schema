import tkinter as tk
from neuromorphic_body_schema.controllers.ball_controller import BallController
from neuromorphic_body_schema.controllers.finger_controller_base import FingerController

BALL_SPEED_MIN = 0.0
BALL_SPEED_MAX = 0.125
PATTERN_DISTANCE_DEFAULT = 0.25

class SimulationControlPanel:
    """Single GUI for both ball and finger control."""

    def __init__(
        self,
        ball_controller: BallController,
        finger_controller: FingerController,
        reset_callback,
    ) -> None:
        self.ball_controller = ball_controller
        self.finger_controller = finger_controller
        self._reset_callback = reset_callback
        self._open = True

        self._root = tk.Toplevel()
        self._root.title("Simulation Control")
        self._root.resizable(False, False)
        self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._root.attributes("-topmost", True)

        container = tk.Frame(self._root, padx=12, pady=12)
        container.grid(row=0, column=0, sticky="nsew")

        ball_frame = tk.LabelFrame(container, text="Ball", padx=10, pady=10)
        ball_frame.grid(row=0, column=0, padx=4, pady=4, sticky="nsew")
        finger_frame = tk.LabelFrame(container, text="Finger", padx=10, pady=10)
        finger_frame.grid(row=1, column=0, padx=4, pady=4, sticky="nsew")

        tk.Label(
            ball_frame,
            text="Ball: WASD or arrow keys. Drag/press buttons for direction.",
            justify="left",
            wraplength=320,
        ).grid(row=0, column=0, columnspan=3, pady=(0, 8), sticky="w")

        tk.Label(ball_frame, text="Speed").grid(row=1, column=0, sticky="w")
        self._ball_speed_var = tk.DoubleVar(value=self.ball_controller._speed)
        self._ball_speed_label = tk.Label(ball_frame, text=f"{self.ball_controller._speed:.3f} m/s")
        self._ball_speed_label.grid(row=1, column=2, sticky="e")
        self._ball_speed_scale = tk.Scale(
            ball_frame,
            from_=BALL_SPEED_MIN,
            to=BALL_SPEED_MAX,
            resolution=0.005,
            orient=tk.HORIZONTAL,
            length=180,
            variable=self._ball_speed_var,
            command=self._on_ball_speed_change,
        )
        self._ball_speed_scale.grid(row=1, column=1, sticky="ew", padx=4)

        tk.Label(ball_frame, text="Pattern Distance").grid(row=2, column=0, sticky="w")
        self._ball_pattern_distance_var = tk.DoubleVar(value=PATTERN_DISTANCE_DEFAULT)
        self._ball_pattern_distance_entry = tk.Entry(
            ball_frame,
            textvariable=self._ball_pattern_distance_var,
            width=10,
        )
        self._ball_pattern_distance_entry.grid(row=2, column=1, sticky="w", padx=4)
        tk.Button(
            ball_frame,
            text="Apply",
            command=self._apply_ball_pattern_distance,
            width=8,
        ).grid(row=2, column=2, sticky="e", padx=4)

        self._make_ball_button(ball_frame, "Forward", "forward", 3, 1)
        self._make_ball_button(ball_frame, "Left", "left", 4, 0)
        self._make_ball_button(ball_frame, "Stop", None, 4, 1, stop_button=True)
        self._make_ball_button(ball_frame, "Right", "right", 4, 2)
        self._make_ball_button(ball_frame, "Back", "back", 5, 1)

        tk.Label(
            ball_frame,
            text="Patterns",
            font=("TkDefaultFont", 9, "bold"),
        ).grid(row=6, column=0, columnspan=3, pady=(8, 2), sticky="w")
        tk.Button(
            ball_frame,
            text="Forward <-> Back",
            command=lambda: self.ball_controller.start_pattern("y"),
            width=16,
        ).grid(row=7, column=0, columnspan=2, padx=4, pady=2, sticky="ew")
        tk.Button(
            ball_frame,
            text="Left <-> Right",
            command=lambda: self.ball_controller.start_pattern("x"),
            width=16,
        ).grid(row=7, column=2, padx=4, pady=2, sticky="ew")
        tk.Button(
            ball_frame,
            text="Stop Pattern",
            command=self.ball_controller.stop_pattern,
            width=16,
        ).grid(row=8, column=0, columnspan=3, padx=4, pady=2, sticky="ew")

        tk.Label(
            finger_frame,
            text="Finger: I/J/K/L keys. Slider sets the force magnitude used for those moves.",
            justify="left",
            wraplength=320,
        ).grid(row=0, column=0, columnspan=3, pady=(0, 8), sticky="w")

        tk.Label(finger_frame, text="Force").grid(row=1, column=0, sticky="w")
        self._finger_force_var = tk.DoubleVar(value=0.5 * self.finger_controller._controller_params.finger_force_max)
        self._finger_force_label = tk.Label(finger_frame, text=f"{0.5 * self.finger_controller._controller_params.finger_force_max:.2f} N")
        self._finger_force_label.grid(row=1, column=2, sticky="e")
        self._finger_force_scale = tk.Scale(
            finger_frame,
            from_=0.0,
            to=self.finger_controller._controller_params.finger_force_max,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            length=180,
            variable=self._finger_force_var,
            command=self._on_finger_force_change,
        )
        self._finger_force_scale.grid(row=1, column=1, sticky="ew", padx=4)

        self._make_finger_button(finger_frame, "Forward (I)", "forward", 2, 1)
        self._make_finger_button(finger_frame, "Left (J)", "left", 3, 0)
        self._make_finger_button(finger_frame, "Stop Finger", None, 3, 1, stop_button=True)
        self._make_finger_button(finger_frame, "Right (L)", "right", 3, 2)
        self._make_finger_button(finger_frame, "Back (K)", "back", 4, 1)

        tk.Button(
            container,
            text="Reset Ball + Finger",
            command=self._reset_callback,
            width=24,
        ).grid(row=2, column=0, padx=4, pady=(10, 0), sticky="ew")

        self._root.bind("<KeyPress>", self._on_key_press)
        self._root.bind("<KeyRelease>", self._on_key_release)

        self._on_ball_speed_change(f"{self.ball_controller._speed}")
        self._apply_ball_pattern_distance()
        self._on_finger_force_change(None)
        self._root.update_idletasks()
        self._root.focus_force()

    def _on_ball_speed_change(self, value: str) -> None:
        speed = float(value)
        self.ball_controller.set_speed(speed)
        self._ball_speed_label.configure(text=f"{speed:.3f} m/s")

    def _apply_ball_pattern_distance(self) -> None:
        try:
            distance = float(self._ball_pattern_distance_var.get())
        except (tk.TclError, ValueError):
            distance = PATTERN_DISTANCE_DEFAULT
            self._ball_pattern_distance_var.set(distance)

        self.ball_controller.set_pattern_distance(distance)

    def _on_finger_force_change(self, _value) -> None:
        force = float(self._finger_force_var.get())
        self.finger_controller.set_force_magnitude(force)
        self._finger_force_label.configure(text=f"{force:.2f} N")

    def _make_ball_button(
        self,
        parent: tk.Widget,
        label: str,
        action: str | None,
        row: int,
        column: int,
        stop_button: bool = False,
    ) -> None:
        button = tk.Button(parent, text=label, width=10 if not stop_button else 8)
        button.grid(row=row, column=column, padx=4, pady=4, sticky="nsew")
        if stop_button:
            button.configure(command=self.ball_controller.stop)
            return
        assert action is not None
        button.bind("<ButtonPress-1>", lambda _event, a=action: self.ball_controller.press(a))
        button.bind("<ButtonRelease-1>", lambda _event, a=action: self.ball_controller.release(a))

    def _make_finger_button(
        self,
        parent: tk.Widget,
        label: str,
        action: str | None,
        row: int,
        column: int,
        stop_button: bool = False,
    ) -> None:
        button = tk.Button(parent, text=label, width=12 if not stop_button else 10)
        button.grid(row=row, column=column, padx=4, pady=4, sticky="nsew")
        if stop_button:
            button.configure(command=self.finger_controller.stop)
            return
        assert action is not None
        button.bind("<ButtonPress-1>", lambda _event, a=action: self.finger_controller.press(a))
        button.bind("<ButtonRelease-1>", lambda _event, a=action: self.finger_controller.release(a))

    def _on_key_press(self, event: tk.Event) -> None:
        ball_action = BallController.key_to_action(str(event.keysym))
        if ball_action is not None:
            self.ball_controller.press(ball_action)

        finger_action = FingerController.key_to_action(str(event.keysym))
        if finger_action is not None:
            self.finger_controller.press(finger_action)

    def _on_key_release(self, event: tk.Event) -> None:
        ball_action = BallController.key_to_action(str(event.keysym))
        if ball_action is not None:
            self.ball_controller.release(ball_action)

        finger_action = FingerController.key_to_action(str(event.keysym))
        if finger_action is not None:
            self.finger_controller.release(finger_action)

    def pump(self) -> None:
        if not self._open:
            return
        try:
            self._root.update_idletasks()
            self._root.update()
        except tk.TclError:
            self._open = False

    def close(self) -> None:
        self.ball_controller.stop()
        self.finger_controller.stop()
        if self._open:
            self._open = False
            try:
                self._root.destroy()
            except tk.TclError:
                pass

    @property
    def is_open(self) -> bool:
        return self._open