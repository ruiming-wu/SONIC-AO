from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

ASSET_DIR = "/home/ruiming.wu/codes/H4"

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

AGILE_ONE_MUJOCO_BODIES = [
    "pelvis_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "left_ankle_pitch_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "right_ankle_pitch_link",
    "torso_link",
    "head_yaw_link",
    "head_pitch_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_roll_link",
    "left_wrist_yaw_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_roll_link",
    "right_wrist_yaw_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
]

# IsaacLab imports URDF articulations in a breadth-first articulation order. This
# order is validated in the Agile One smoke test and converted back to the MJCF
# order used by the AO motionlib.
AGILE_ONE_ISAACLAB_BODIES = [
    "pelvis_link",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "torso_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "head_yaw_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "head_pitch_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_roll_link",
    "right_elbow_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
]

AGILE_ONE_MUJOCO_DOFS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_roll_joint",
    "left_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_roll_joint",
    "right_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
]

AGILE_ONE_ISAACLAB_DOFS = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "head_yaw_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "head_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_roll_joint",
    "right_elbow_roll_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
]

AGILE_ONE_ISAACLAB_TO_MUJOCO_DOF = [
    AGILE_ONE_ISAACLAB_DOFS.index(name) for name in AGILE_ONE_MUJOCO_DOFS
]
AGILE_ONE_MUJOCO_TO_ISAACLAB_DOF = [
    AGILE_ONE_MUJOCO_DOFS.index(name) for name in AGILE_ONE_ISAACLAB_DOFS
]
AGILE_ONE_ISAACLAB_TO_MUJOCO_BODY = [
    AGILE_ONE_ISAACLAB_BODIES.index(name) for name in AGILE_ONE_MUJOCO_BODIES
]
AGILE_ONE_MUJOCO_TO_ISAACLAB_BODY = [
    AGILE_ONE_MUJOCO_BODIES.index(name) for name in AGILE_ONE_ISAACLAB_BODIES
]

AGILE_ONE_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": AGILE_ONE_ISAACLAB_BODIES,
    "mujoco_dofs": AGILE_ONE_MUJOCO_DOFS,
    "mujoco_bodies": AGILE_ONE_MUJOCO_BODIES,
    "isaaclab_to_mujoco_dof": AGILE_ONE_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": AGILE_ONE_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": AGILE_ONE_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": AGILE_ONE_MUJOCO_TO_ISAACLAB_BODY,
}

AGILE_ONE_NO_HANDS_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/urdf/agile_one_no_hands.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            ".*_hip_pitch_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_knee_joint": 0.0,
            ".*_ankle_pitch_joint": 0.0,
            ".*_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "head_yaw_joint": 0.0,
            "head_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": -1.57079627,
            "left_shoulder_roll_joint": -1.3962634,
            "left_shoulder_yaw_joint": -1.57079627,
            "left_elbow_roll_joint": 1.57079627,
            "left_wrist_yaw_joint": 1.3962634,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "right_shoulder_pitch_joint": 1.57079627,
            "right_shoulder_roll_joint": 1.3962634,
            "right_shoulder_yaw_joint": 1.57079627,
            "right_elbow_roll_joint": -1.57079627,
            "right_wrist_yaw_joint": -1.3962634,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    # Retargeted Agile One references often sit near the MJCF hard limits.
    # Using a 0.9 soft range clips the reset pose and desynchronizes joint
    # state from the FK/body targets stored in the motion library.
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 139.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 20.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_7520_22,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_knee_joint": STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_7520_22,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_knee_joint": DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_7520_22,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        "head": ImplicitActuatorCfg(
            effort_limit_sim=25.0,
            velocity_limit_sim=22.0,
            joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
            stiffness=STIFFNESS_4010,
            damping=DAMPING_4010,
            armature=ARMATURE_4010,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow.*joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow.*joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow.*joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_5020,
                ".*_shoulder_roll_joint": STIFFNESS_5020,
                ".*_shoulder_yaw_joint": STIFFNESS_5020,
                ".*_elbow.*joint": STIFFNESS_5020,
                ".*_wrist_roll_joint": STIFFNESS_5020,
                ".*_wrist_pitch_joint": STIFFNESS_4010,
                ".*_wrist_yaw_joint": STIFFNESS_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_5020,
                ".*_shoulder_roll_joint": DAMPING_5020,
                ".*_shoulder_yaw_joint": DAMPING_5020,
                ".*_elbow.*joint": DAMPING_5020,
                ".*_wrist_roll_joint": DAMPING_5020,
                ".*_wrist_pitch_joint": DAMPING_4010,
                ".*_wrist_yaw_joint": DAMPING_4010,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_5020,
                ".*_shoulder_roll_joint": ARMATURE_5020,
                ".*_shoulder_yaw_joint": ARMATURE_5020,
                ".*_elbow.*joint": ARMATURE_5020,
                ".*_wrist_roll_joint": ARMATURE_5020,
                ".*_wrist_pitch_joint": ARMATURE_4010,
                ".*_wrist_yaw_joint": ARMATURE_4010,
            },
        ),
    },
)

AGILE_ONE_ACTION_SCALE = {}
for actuator in AGILE_ONE_NO_HANDS_CFG.actuators.values():
    effort = actuator.effort_limit_sim
    stiffness = actuator.stiffness
    names = actuator.joint_names_expr
    if not isinstance(effort, dict):
        effort = dict.fromkeys(names, effort)
    if not isinstance(stiffness, dict):
        stiffness = dict.fromkeys(names, stiffness)
    for name in names:
        if name in effort and name in stiffness and stiffness[name]:
            AGILE_ONE_ACTION_SCALE[name] = 0.25 * effort[name] / stiffness[name]
