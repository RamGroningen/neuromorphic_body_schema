import numpy as np
import mujoco


def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero-length vector")
    return v / n


def quat_wxyz_from_R(R):
    """Convert a 3x3 rotation matrix to MuJoCo quaternion order [w, x, y, z]."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def build_root_rotation(f_local, down_local, finger_pos, sphere_pos, keep_parallel_to_floor=True):
    """
    Build root rotation so:
    - local fingertip-forward points toward the sphere
    - local sensor-facing axis points downward
    - optional: keep forward parallel to the floor (z=0)
    """
    f_world_raw = sphere_pos - finger_pos
    if keep_parallel_to_floor:
        f_world_raw = np.array([f_world_raw[0], f_world_raw[1], 0.0])
        if np.linalg.norm(f_world_raw) < 1e-8:
            f_world_raw = np.array([-1.0, 0.0, 0.0])
    f_world = normalize(f_world_raw)

    down_world = np.array([0.0, 0.0, -1.0])
    v_world = normalize(np.cross(down_world, f_world))
    down_world = normalize(np.cross(f_world, v_world))

    f_local = normalize(f_local)
    down_local = down_local - np.dot(down_local, f_local) * f_local
    down_local = normalize(down_local)
    v_local = normalize(np.cross(down_local, f_local))
    down_local = normalize(np.cross(f_local, v_local))

    B_local = np.column_stack([f_local, v_local, down_local])
    B_world = np.column_stack([f_world, v_world, down_world])

    R_root = B_world @ B_local.T
    q_root = quat_wxyz_from_R(R_root)
    return R_root, q_root


model = mujoco.MjModel.from_xml_path("neuromorphic_body_schema/models/finger_plane_scene.xml")
data = mujoco.MjData(model)

# Set free-joint root to identity at origin so local offsets are easy to read.
jid_free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_hand_freejoint")
qadr = model.jnt_qposadr[jid_free]
data.qpos[qadr : qadr + 3] = [0.0, 0.0, 0.0]
data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]  # w x y z
mujoco.mj_forward(model, data)

bid_hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "icub_r_hand")
bid_i0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_hand_index_0")
bid_distal = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_hand_index_3")
bid_sph = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ground_sphere_body")
gid_sph = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground_sphere")

sid_sensor_names = [f"r_hand_index_3_taxel_{i}" for i in range(12)]
sid_sensors = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in sid_sensor_names]
sensor_points = np.array([data.site_xpos[sid].copy() for sid in sid_sensors])

# Use distal-most sensor cluster as fingertip contact anchor.
distal_ids = np.array([8, 9, 10, 11])
mid_ids = np.array([4, 5, 6, 7])
p_distal = np.mean(sensor_points[distal_ids], axis=0)
p_mid = np.mean(sensor_points[mid_ids], axis=0)

# Hand-root offset vector and distance to distal cluster center.
p_hand = data.xpos[bid_hand].copy()
R_hand = data.xmat[bid_hand].reshape(3, 3).copy()
o_tip_hand = R_hand.T @ (p_distal - p_hand)
d_tip_hand = np.linalg.norm(p_distal - p_hand)

# Index-0 offset vector and distance (diagnostic only).
p_i0 = data.xpos[bid_i0].copy()
R_i0 = data.xmat[bid_i0].reshape(3, 3).copy()
o_tip_i0 = R_i0.T @ (p_distal - p_i0)
d_tip_i0 = np.linalg.norm(p_distal - p_i0)

print("distance hand->distal cluster:", d_tip_hand)
print("distance index0->distal cluster:", d_tip_i0)

sphere_radius = model.geom_size[gid_sph, 0]
sphere_pos = data.xpos[bid_sph].copy()

# Root guess used only to define a direction toward sphere.
finger_pos_guess = sphere_pos + np.array([d_tip_hand, 0.0, sphere_radius])

# Local forward from middle to distal patch.
f_local = normalize(p_distal - p_mid)

# Local down from distal patch plane normal.
v1 = sensor_points[10] - sensor_points[8]
v2 = sensor_points[11] - sensor_points[9]
down_local = normalize(np.cross(v1, v2))

# Choose normal direction that points away from the finger base.
to_base = normalize(p_hand - p_distal)
if np.dot(down_local, to_base) > 0.0:
    down_local = -down_local

R_root, q_root = build_root_rotation(
    f_local=f_local,
    down_local=down_local,
    finger_pos=finger_pos_guess,
    sphere_pos=sphere_pos,
    keep_parallel_to_floor=True,
)

# Target point: sphere top with tiny inward bias for robust initial contact.
eps = 5e-4
p_target = sphere_pos + np.array([0.0, 0.0, sphere_radius - eps])

# Solve root position so distal-cluster center lands on target.
p_root = p_target - R_root @ o_tip_hand

# Validate solved pose.
data.qpos[qadr : qadr + 3] = p_root
data.qpos[qadr + 3 : qadr + 7] = q_root
mujoco.mj_forward(model, data)

distal_world = np.mean(np.array([data.site_xpos[sid].copy() for sid in sid_sensors])[distal_ids], axis=0)
distal_err = np.linalg.norm(distal_world - p_target)

R_distal_new = data.xmat[bid_distal].reshape(3, 3).copy()
forward_world = normalize(R_distal_new[:, 0])
down_world = normalize(R_distal_new[:, 2])

print("sphere center:", sphere_pos)
print("sphere top target:", p_target)
print("root guess:", finger_pos_guess)
print("solved root position:", p_root)
print("solved root quat [w x y z]:", q_root)
print("distal-cluster->target error [m]:", distal_err)
print("finger parallel to floor check abs(forward.z):", abs(forward_world[2]))
print("sensor axis downward dot([0,0,-1]):", np.dot(down_world, np.array([0.0, 0.0, -1.0])))

# Write p_root and q_root to both XML bodies:
# - icub_r_hand_welding
# - icub_r_hand