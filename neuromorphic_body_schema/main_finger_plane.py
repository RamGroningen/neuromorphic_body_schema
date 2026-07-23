"""
main.py

Author: Ram Eshwar Kaundinya
Affiliation: University of Groningen (UG)
Department: Bernoulli Institute and Cognigron
Date: 29.04.2025

Description:
This script initializes and runs a neuromorphic simulation of the iCub robot using MuJoCo.
A single finger's movement is simulated along with a ball in the scene. Both can be moved.

Modules:
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

# Get required roots for direct script execution:
# - package_root exposes `neuromorphic_body_schema`
# - workspace_root exposes top-level `robot`
if __name__ == "__main__":
    file_path = Path(__file__).resolve()
    package_root = file_path.parent.parent
    workspace_root = file_path.parents[3]

    for _path in (package_root, workspace_root):
        path_str = str(_path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

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
from neuromorphic_body_schema.ui.tkinter_guis import SimulationControlPanel
from neuromorphic_body_schema.controllers.ball_controller import BallController
from neuromorphic_body_schema.controllers.finger_controller_base import FingerController, FingerControllerParams

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
BALL_SPEED = 0.05
BALL_SPEED_DEFAULT = BALL_SPEED
BALL_SPEED_MIN = 0.0
BALL_SPEED_MAX = 0.125
PATTERN_DISTANCE_DEFAULT = 0.25

# --- Finger force controller constants ---
FINGER_DAMPING_TRANS = 8.0   # N·s/m  translational velocity damping
FINGER_DAMPING_ROT   = 0.5   # N·m·s/rad rotational velocity damping
FINGER_FORCE_MIN     = -2.0  # N     minimum commanded Cartesian force per axis
FINGER_FORCE_MAX     = 2.0   # N     maximum commanded Cartesian force per axis
FINGER_TORQUE_MAX    = 2.0   # N·m   maximum joint torque


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
    ball_controller = BallController(BALL_SPEED_DEFAULT, PATTERN_DISTANCE_DEFAULT)
    ball_controller.set_pattern_center(ball_rest_position[:2])

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

    # --- Finger force controller setup ---
    hand_body_id = mj_name2id(model, mjtObj.mjOBJ_BODY, "icub_r_hand")
    jid_free = mj_name2id(model, mjtObj.mjOBJ_JOINT, "r_hand_freejoint")
    hand_free_dofadr = model.jnt_dofadr[jid_free]  # index into data.qvel for the 6 free DOFs
    hand_free_qposadr = model.jnt_qposadr[jid_free]  # index into data.qpos for the freejoint pose
    finger_root_z_target = data.qpos[hand_free_qposadr + 2]
    finger_root_xy_target = data.qpos[hand_free_qposadr : hand_free_qposadr + 2].copy()

    # Enable gravity compensation for the entire finger chain so it floats
    for _bname in ["icub_r_hand", "r_hand_index_0", "r_hand_index_1",
                   "r_hand_index_2", "r_hand_index_3"]:
        _bid = mj_name2id(model, mjtObj.mjOBJ_BODY, _bname)
        model.body_gravcomp[_bid] = 1.0

    finger_controller = FingerController(FingerControllerParams(FINGER_FORCE_MIN, FINGER_FORCE_MAX, FINGER_TORQUE_MAX))
    finger_control_panel = None

    def reset_ball_and_finger() -> None:
        data.mocap_pos[ball_mocap_id] = ball_rest_position.copy()
        data.qpos[hand_free_qposadr : hand_free_qposadr + 2] = finger_root_xy_target
        data.qpos[hand_free_qposadr + 2] = finger_root_z_target
        data.qvel[hand_free_dofadr : hand_free_dofadr + 6] = 0.0
        finger_controller.stop()
        mj_forward(model, data)

    control_panel = SimulationControlPanel(
        ball_controller=ball_controller,
        finger_controller=finger_controller,
        reset_callback=reset_ball_and_finger,
    )

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

                if ball_controller.consume_reset_request():
                    reset_ball_and_finger()

                direction = ball_controller.step(
                    model.opt.timestep,
                    data.mocap_pos[ball_mocap_id][:2],
                )
                if np.any(direction):
                    ball_position = data.mocap_pos[ball_mocap_id]
                    ball_position[:2] += direction * ball_controller.speed() * model.opt.timestep
                    ball_position[2] = ball_rest_position[2]
                    mj_forward(model, data)

                # --- Apply finger directional motion as force, only while I/J/K/L is pressed ---
                trans_vel = data.qvel[hand_free_dofadr     : hand_free_dofadr + 3]
                rot_vel   = data.qvel[hand_free_dofadr + 3 : hand_free_dofadr + 6]
                cart_force = finger_controller.get_cartesian_force()
                finger_xy_force = np.zeros(2, dtype=float)
                finger_xy_force[0] = cart_force[0] - FINGER_DAMPING_TRANS * trans_vel[0]
                finger_xy_force[1] = cart_force[1] - FINGER_DAMPING_TRANS * trans_vel[1]
                wrench = np.zeros(6)
                wrench[0] = finger_xy_force[0]
                wrench[1] = finger_xy_force[1]
                wrench[2] = 0.0
                wrench[3] =              - FINGER_DAMPING_ROT   * rot_vel[0]
                wrench[4] =              - FINGER_DAMPING_ROT   * rot_vel[1]
                wrench[5] =              - FINGER_DAMPING_ROT   * rot_vel[2]
                data.xfrc_applied[hand_body_id] = wrench

                mj_step(model, data)  # Step the simulation

                # Hard-lock the finger height at the initial z position.
                data.qpos[hand_free_qposadr + 2] = finger_root_z_target
                data.qvel[hand_free_dofadr + 2] = 0.0
                mj_forward(model, data)

                sim_viewer.sync()

                skin_events = skin_object.update_skin(
                    data.time * 1e9
                )  # expects ns

                # --- Feed sensor readings to the finger controller hook ---
                finger_taxels = np.array(dynamic_grouped_sensors["r_hand_index_3_taxel"])
                j_pos = np.array([
                    data.joint("r_hand_index_1_joint").qpos[0],
                    data.joint("r_hand_index_2_joint").qpos[0],
                ])
                j_vel = np.array([
                    data.joint("r_hand_index_1_joint").qvel[0],
                    data.joint("r_hand_index_2_joint").qvel[0],
                ])
                finger_controller.update(finger_taxels, j_pos, j_vel)

                '''current_pos = data.mocap_pos[mocap_id]
                x_cmd = current_pos[0] + 0.0001
                y_cmd = current_pos[1] + 0.0001'''
                #data.mocap_pos[mocap_id] = np.array([x_cmd, y_cmd, z_fixed])

                pass
    finally:
        control_panel.close()