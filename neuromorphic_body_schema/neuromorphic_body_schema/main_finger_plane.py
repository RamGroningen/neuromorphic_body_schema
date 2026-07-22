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