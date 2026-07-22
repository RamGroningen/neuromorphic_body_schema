"""
main.py

Author: Simon F. Muller-Cleve
Affiliation: Istituto Italiano di Tecnologia (IIT)
Department: Event-Driven Perception for Robotics (EDPR)
Date: 29.04.2025

Description:
This script initializes and runs a neuromorphic simulation of the iCub robot using MuJoCo.
It integrates event-based camera, proprioception, and skin sensors, and provides visualization
options for each sensory modality.

Modules:
- ICubEyes: Simulates event-based camera functionality.
- ICubProprioception: Simulates proprioceptive spiking events.
- ICubSkin: Simulates tactile skin events.
- DynamicGroupedSensors: Provides dynamic access to grouped sensor data.
- update_joint_positions: Updates joint positions in the MuJoCo model.
- init_POV: Configures the viewer's point of view.

Usage:
Run this script to start the simulation and visualize the sensory data.

"""

# Add parent directory to path for direct script execution
import sys
from pathlib import Path

# Get the project root directory (parent of neuromorphic_body_schema)
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import copy
import logging
import math
import re
import threading
from collections import defaultdict

import mujoco
import numpy as np
import tkinter as tk
from neuromorphic_body_schema.helpers.ed_cam import ICubEyes
from neuromorphic_body_schema.helpers.ed_prop import ICubProprioception
from neuromorphic_body_schema.helpers.ed_skin import ICubSkin
from neuromorphic_body_schema.helpers.helpers import (
    MODEL_PATH,
    DynamicGroupedSensors,
    init_POV,
)
from neuromorphic_body_schema.helpers.ik_solver import Ik_solver
from neuromorphic_body_schema.helpers.robot_controller import (
    check_joints,
    ik_calculation,
    reset_simulation,
    update_joint_positions,
)
from mujoco import viewer

# from helpers.ik_solver_fede import qpos_from_site_pose

DEBUG = False  # use to visualize the triangles
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

# MuJoCo is a C-extension module and some symbols are not visible to static analyzers.
# Resolve once via getattr so runtime behavior stays identical while Pylance can type-check calls.
MjModel = getattr(mujoco, "MjModel")
MjData = getattr(mujoco, "MjData")
mj_name2id = getattr(mujoco, "mj_name2id")
mjtObj = getattr(mujoco, "mjtObj")
mj_forward = getattr(mujoco, "mj_forward")
mj_step = getattr(mujoco, "mj_step")

CAMERA_MODE = "frame_based"  # "event_driven" or "frame_based"
CAM_TO_USE = "all"  # "left", "right", or "all"
VISUALIZE_CAMERA_FEED = False
# setting this to true also activates the event-based camera if not already activated, since we need it to show the feed
if CAMERA_MODE == "event_driven":
    VISUALIZE_ED_CAMERA_FEED = True
else:
    VISUALIZE_ED_CAMERA_FEED = False

# NOTE: Skin can be visualized when toggling Site group 1-5 in the MuJoCo viewer
SKIN_MODE = "frame_based"  # "event_driven" or "frame_based"
SKIN_PART = "all"  # see helpers SKIN_PARTS for the list of possible skin parts
# ["r_hand", "r_forearm", "r_upper_arm", "torso", "l_hand", "l_forearm", "l_upper_arm", "r_upper_leg", "r_lower_leg", "l_upper_leg", "l_lower_leg"]
VISUALIZE_SKIN_FEED = False
VISUALIZE_ED_SKIN_FEED = False

PROPRIOCEPTION_MODE = "event_driven"  # "event_driven" or "frame_based"
VISUALIZE_PROPRIOCEPTION_FEED = False

BALL_BODY_NAME = "ground_sphere_body"
BALL_SPEED = 0.05  # meters per second while a control input is held


class BallMotionController:
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

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pressed_actions: set[str] = set()

    def press(self, action: str) -> None:
        with self._lock:
            self._pressed_actions.add(action)

    def release(self, action: str) -> None:
        with self._lock:
            self._pressed_actions.discard(action)

    def stop(self) -> None:
        with self._lock:
            self._pressed_actions.clear()

    def vector(self) -> np.ndarray:
        with self._lock:
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


class BallControlPanel:
    """Small Tk panel for pressing/releasing ball movement commands."""

    def __init__(self, controller: BallMotionController) -> None:
        self.controller = controller
        self._open = True
        self._root = tk.Tk()
        self._root.title("Ball Control")
        self._root.resizable(False, False)
        self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._root.attributes("-topmost", True)

        container = tk.Frame(self._root, padx=12, pady=12)
        container.grid(row=0, column=0)

        tk.Label(
            container,
            text="Hold buttons or W/A/S/D, or arrow keys. Release stops the ball.",
            justify="left",
            wraplength=280,
        ).grid(row=0, column=0, columnspan=3, pady=(0, 10), sticky="w")

        self._make_button(container, "Forward", "forward", 1, 1)
        self._make_button(container, "Left", "left", 2, 0)
        self._make_button(container, "Stop", None, 2, 1, stop_button=True)
        self._make_button(container, "Right", "right", 2, 2)
        self._make_button(container, "Back", "back", 3, 1)

        tk.Label(
            container,
            text="Tip: keep this window focused for keyboard input.",
            fg="#555555",
        ).grid(row=4, column=0, columnspan=3, pady=(10, 0), sticky="w")

        self._root.bind("<KeyPress>", self._on_key_press)
        self._root.bind("<KeyRelease>", self._on_key_release)
        self._root.update_idletasks()
        self._root.focus_force()

    def _make_button(
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
            button.configure(command=self.controller.stop)
            return

        assert action is not None
        button.bind("<ButtonPress-1>", lambda _event, a=action: self.controller.press(a))
        button.bind(
            "<ButtonRelease-1>",
            lambda _event, a=action: self.controller.release(a),
        )

    def _on_key_press(self, event: tk.Event) -> None:
        action = BallMotionController.key_to_action(str(event.keysym))
        if action is not None:
            self.controller.press(action)

    def _on_key_release(self, event: tk.Event) -> None:
        action = BallMotionController.key_to_action(str(event.keysym))
        if action is not None:
            self.controller.release(action)

    def pump(self) -> None:
        if not self._open:
            return

        try:
            self._root.update_idletasks()
            self._root.update()
        except tk.TclError:
            self._open = False

    def close(self) -> None:
        self.controller.stop()
        if self._open:
            self._open = False
            try:
                self._root.destroy()
            except tk.TclError:
                pass

    @property
    def is_open(self) -> bool:
        return self._open


if __name__ == "__main__":
    #############################
    ### setting everything up ###
    #############################

    viewer_closed_event = threading.Event()

    # Load the MuJoCo model and create a simulation
    model = MjModel.from_xml_path(MODEL_PATH)
    '''body_id = mj_name2id(model, mjtObj.mjOBJ_BODY, "icub_r_hand_welding")
    mocap_id = model.body_mocapid[body_id]
    z_fixed = 1.5'''
    data = MjData(model)
    # set model to 0.0 start position
    #data.qpos.fill(0.0)
    # define a start position for the model
    '''
    joint_init_pos = {
        "r_shoulder_roll": 0.6,
        "r_shoulder_pitch": -0.5,
        "r_shoulder_yaw": 0.0,
        "r_elbow": 1.1,
        "l_shoulder_roll": 0.6,
        "l_shoulder_pitch": -0.5,
        "l_shoulder_yaw": 0.0,
        "l_elbow": 1.1,
    }
    # let's set the initial joint positions and actuator controls
    for joint_name, position in joint_init_pos.items():
        try:
            joint_id = mj_name2id(model, mjtObj.mjOBJ_JOINT, joint_name)
            data.joint(joint_id).qpos[0] = position
            data.actuator(joint_name).ctrl[0] = position
        except ValueError:
            logging.warning(f"Joint {joint_name} not found in the model.")'''
    print("Model loaded")

    # Set the time step duration to 0.001 seconds (1 milliseconds)
    model.opt.timestep = 0.001  # sec

    # prepare the mapping from skin to body parts
    names_list = model.names.decode("utf-8").split("\x00")
    sensor_info = [x for x in names_list if "taxel" in x]

    # Extract base names and group sensor addresses by base names
    grouped_sensors = defaultdict(list)
    for adr, name in enumerate(sensor_info):
        base_name = re.sub(r"_\d+$", "", name)
        grouped_sensors[base_name].append(adr)

    if DEBUG:
        for key, value in grouped_sensors.items():
            print(key, len(value))

    dynamic_grouped_sensors = DynamicGroupedSensors(data, grouped_sensors)

    ball_body_id = mj_name2id(model, mjtObj.mjOBJ_BODY, BALL_BODY_NAME)
    ball_mocap_id = model.body_mocapid[ball_body_id]
    ball_rest_position = data.mocap_pos[ball_mocap_id].copy()
    ball_controller = BallMotionController()
    control_panel = BallControlPanel(ball_controller)

    joint_dict_prop = {
        "r_shoulder_roll": {
            "position_max_freq": 1000,  # Hz
            "velocity_max_freq": 1000,
            "load_max_freq": 1000,
            "limits_max_freq": 1000,
        },
        "l_shoulder_roll": {
            "position_max_freq": 1000,
            "velocity_max_freq": 1000,
            "load_max_freq": 1000,
            "limits_max_freq": 1000,
        },
    }

    ############################
    ### Start the simulation ###
    ############################

    try:
        with viewer.launch_passive(model, data) as sim_viewer:
            init_POV(sim_viewer)

            sim_time = data.time

            skin_object = ICubSkin(
                sim_time,
                dynamic_grouped_sensors,
                skin=SKIN_PART,
                skin_mode=SKIN_MODE,
                show_raw_feed=VISUALIZE_SKIN_FEED,
                show_ed_feed=VISUALIZE_ED_SKIN_FEED,
                DEBUG=DEBUG
            )

            # count = 0
            while sim_viewer.is_running():
                control_panel.pump()

                direction = ball_controller.vector()
                if np.any(direction):
                    ball_position = data.mocap_pos[ball_mocap_id]
                    ball_position[:2] += direction * BALL_SPEED * model.opt.timestep
                    ball_position[2] = ball_rest_position[2]
                    mj_forward(model, data)

                mj_step(model, data)  # Step the simulation
                sim_viewer.sync()

                skin_events = skin_object.update_skin(
                    data.time * 1e9
                )  # expects ns

                '''current_pos = data.mocap_pos[mocap_id]
                x_cmd = current_pos[0] + 0.0001
                y_cmd = current_pos[1] + 0.0001'''
                #data.mocap_pos[mocap_id] = np.array([x_cmd, y_cmd, z_fixed])

                pass
    finally:
        control_panel.close()