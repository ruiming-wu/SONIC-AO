from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

ASSET_DIR = "gear_sonic/data/assets"

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

VELOCITY_SAFETY_MARGIN = 0.0
TORQUE_SAFETY_MARGIN = 0.1

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
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "head_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_elbow_roll_link",
    "right_elbow_roll_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
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
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_elbow_roll_joint",
    "right_elbow_roll_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
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
    "isaaclab_to_mujoco_dof": AGILE_ONE_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": AGILE_ONE_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": AGILE_ONE_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": AGILE_ONE_MUJOCO_TO_ISAACLAB_BODY,
}

AGILE_ONE_NO_HANDS_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/robot_description/urdf/agile_one/agile_one_no_hands.urdf",
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
        pos=(0.0, 0.0, 0.981532),
        joint_pos={
            ".*_hip_pitch_joint": -0.2713691,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": -1.340424e-13,
            ".*_knee_joint": 0.5430939,
            ".*_ankle_pitch_joint": -0.2717248,
            ".*_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "head_yaw_joint": 0.0,
            "head_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": -1.3963,
            "left_shoulder_roll_joint": -1.30,
            "left_shoulder_yaw_joint": -1.57079627,
            "left_elbow_roll_joint": 0.34907,
            "left_wrist_yaw_joint": 1.57079627,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "right_shoulder_pitch_joint": 1.3963,
            "right_shoulder_roll_joint": 1.30,
            "right_shoulder_yaw_joint": 1.57079627,
            "right_elbow_roll_joint": -0.34907,
            "right_wrist_yaw_joint": -1.57079627,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "hip": DCMotorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
            ],
            effort_limit=704.0 * (1 - TORQUE_SAFETY_MARGIN),
            velocity_limit=5.0 * (1 - VELOCITY_SAFETY_MARGIN),
            saturation_effort=704.0,
            stiffness=227.0,
            damping=12.0,
            armature=0.16,
        ),
        "knee": DCMotorCfg(
            joint_names_expr=[
                ".*_knee_joint",
            ],
            effort_limit=704.0 * (1 - TORQUE_SAFETY_MARGIN),
            velocity_limit=5.0 * (1 - VELOCITY_SAFETY_MARGIN),
            saturation_effort=704.0,
            stiffness=227.0,
            damping=12.0,
            armature=0.16,
        ),
        "hip_rot": DCMotorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
            ],
            effort_limit=350.0 * (1 - TORQUE_SAFETY_MARGIN),
            velocity_limit=8.0 * (1 - VELOCITY_SAFETY_MARGIN),
            saturation_effort=350.0,
            stiffness=227.0,
            damping=8.0,
            armature=0.16,
        ),
        "feet": DCMotorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit=1150.0,
            velocity_limit=2.7,
            saturation_effort=1200.0,
            stiffness=227.0,
            damping=8.0,
            armature=0.16,
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=300.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=250.0,
            damping=7.0,
            armature=0.3,
        ),
        "head": ImplicitActuatorCfg(
            effort_limit_sim=300.0,
            joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
            stiffness=40.0,
            damping=10.0,
            armature=1.53e-4,
        ),
        "shoulder_pitch": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
            ],
            effort_limit_sim=300.0,
            stiffness=40.0,
            damping=10.0,
            armature=1.53e-4,
        ),
        "shoulder_roll": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_roll_joint",
            ],
            effort_limit_sim=300.0,
            stiffness=40.0,
            damping=10.0,
            armature=4.8e-5,
        ),
        "shoulder_yaw_elbow": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_yaw_joint",
                ".*_elbow.*joint",
            ],
            effort_limit_sim=300.0,
            stiffness=40.0,
            damping=10.0,
            armature=1.7e-5,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_.*_joint",
            ],
            effort_limit_sim=300.0,
            stiffness=40.0,
            damping=10.0,
            armature=8.51e-6,
        ),
    },
)

AGILE_ONE_ACTION_SCALE = {}
for actuator in AGILE_ONE_NO_HANDS_CFG.actuators.values():
    for name in actuator.joint_names_expr:
        AGILE_ONE_ACTION_SCALE[name] = 0.5
