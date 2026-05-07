# Agile One SONIC Training / Finetuning Plan

This note describes the minimum viable plan for adapting the G1-trained SONIC
policy to Agile One for pure PICO4Ultra whole-body teleoperation. The target
input at deployment is full-body SMPLX/SMPL-style human pose, not language,
planner, sparse 3-point VR, autonomous motion, or upper/lower-body split
control.

## Target Scope

Goal:

- Use the selected five SEED categories, about 58k motion clips.
- Use the retargeted Agile One CSVs as robot motion imitation targets.
- Use PICO4Ultra full-body SMPLX output as the deployment input modality.
- Finetune or retrain the SONIC universal-token policy for Agile One.
- Keep only the G1-style robot motion branch and the SMPL/SMPLX branch.

Non-goals for the first training version:

- Natural-language control.
- Planner / autonomous locomotion.
- Sparse VR 3-point control.
- Upper/lower-body split control.
- SOMA encoder training, unless later used as an ablation.
- Hand/finger policy learning beyond whatever Agile One no-hands body model
  currently exposes.

## Relevant Code Facts

SONIC release config uses three encoders:

- `g1`: robot motion / joint trajectory encoder.
- `teleop`: VR 3-point encoder.
- `smpl`: SMPL human motion encoder.

They share a finite-scalar-quantized token space and a dynamic decoder. The
deployment decoder is `g1_dyn`; `g1_kin` is only for auxiliary reconstruction
loss.

Relevant files:

- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml`
- `gear_sonic/config/actor_critic/universal_token/all_mlp_v1.yaml`
- `gear_sonic/config/actor_critic/encoders/g1_mf_mlp.yaml`
- `gear_sonic/config/actor_critic/encoders/smpl_mlp.yaml`
- `gear_sonic/config/actor_critic/decoders/g1_dyn_mlp.yaml`
- `gear_sonic/trl/modules/universal_token_modules.py`
- `gear_sonic/envs/manager_env/mdp/commands.py`

The official new-embodiment guide says a non-G1 robot needs:

- URDF/USD for Isaac Lab simulation.
- MJCF/XML for motion-library forward kinematics.
- Robot config with joint/body order, actuator parameters, action scale, and
  IsaacLab <-> MuJoCo mappings.
- Order converter.
- Experiment config.
- Retargeted robot motion data in SONIC `motion_lib` PKL format.

Reference file:

- `docs/source/user_guide/new_embodiments.md`

## Data Assets We Have

Current expected data sources:

- `soma_uniform` BVH for the selected five categories.
- `soma_proportional` BVH for the selected five categories.
- Retargeted Agile One CSVs for the selected five categories.

The aggregated retargeting route is:

- `/home/ruiming.wu/data/seed-retargeted/soma_agile_one/<category>/motions/*.csv`

The selected categories are:

- `Basic Locomotion Neutral`
- `Object Manipulation`
- `Household`
- `Object Interaction`
- `Environments`

Previous Agile One baseline, using the older retargeting config:

| Route | Total | Success | Warn | Fail | Success % | Warn % | Fail % |
|---|---:|---:|---:|---:|---:|---:|
| `soma_agile_one` | 57799 | 50320 | 7383 | 96 | 87.06 | 12.77 | 0.17 |

The first training dataset should use only success clips. Warning clips can be
added later as hard cases. Failed clips should stay quarantined.

## Minimum Required Training Data

The minimum viable SONIC training set needs two matched streams:

1. Agile One robot motion library.
2. SMPL/SMPLX-compatible human motion library.

SOMA BVH data is useful for traceability and optional SOMA encoder experiments,
but it is not enough by itself for the released `sonic_release` config.

Recommended layout:

```text
data/agile_one_seed_wbc/
├── robot_filtered/
│   ├── basic_locomotion_neutral/*.pkl
│   ├── object_manipulation/*.pkl
│   ├── household/*.pkl
│   ├── object_interaction/*.pkl
│   └── environments/*.pkl
├── smpl_filtered/
│   ├── basic_locomotion_neutral/*.pkl
│   ├── object_manipulation/*.pkl
│   ├── household/*.pkl
│   ├── object_interaction/*.pkl
│   └── environments/*.pkl
└── manifests/
    ├── train_success_keys.txt
    ├── warn_keys.txt
    └── fail_keys.txt
```

The robot and SMPL files must use matching motion keys. The motion library
loads SMPL files by matching the robot motion basename against SMPL PKL
basenames.

## Agile One Robot Motion PKL Format

Each Agile One robot motion PKL should contain:

```python
{
    "motion_name": {
        "root_trans_offset": np.ndarray,  # (T, 3), meters
        "pose_aa": np.ndarray,            # (T, num_bodies, 3), axis-angle
        "dof": np.ndarray,                # (T, 29), radians, Agile One MuJoCo order
        "root_rot": np.ndarray,           # (T, 4), xyzw
        "fps": 30,
        "smpl_joints": np.zeros((T, 24, 3), dtype=np.float32),
    }
}
```

Important:

- `dof` must be in Agile One MuJoCo joint order.
- `pose_aa` must be in Agile One MuJoCo body order.
- `root_trans_offset` is meters.
- `root_rot` convention must be consistent with the SONIC motion library path.
- The current G1 converter cannot be reused as-is.

The existing converter:

- `gear_sonic/data_process/convert_soma_csv_to_motion_lib.py`

is G1-specific because it hardcodes:

- G1 29-DoF joint names.
- G1 MuJoCo / IsaacLab reorder mapping.
- G1 `DOF_AXIS`.
- G1 body count assumptions.

For Agile One, write a new converter that parses:

- `/home/ruiming.wu/codes/H4/mjcf/agile_one_no_hands.xml`

or reuses the same FK path as SONIC `Humanoid_Batch`.

## SMPL / SMPLX Human Motion PKL Format

Each matched SMPL motion PKL should contain:

```python
{
    "motion_name": {
        "pose_aa": np.ndarray,      # (T, 72), SMPL root + 23 joints axis-angle
        "smpl_joints": np.ndarray,  # (T, 24, 3)
        "transl": np.ndarray,       # (T, 3), recommended
        "fps": 30,
    }
}
```

Deployment with PICO4Ultra should use the SONIC SMPL path. The documented ZMQ
protocol v3 uses:

- `joint_pos`: `[N, 29]`
- `joint_vel`: `[N, 29]`
- `smpl_joints`: `[N, 24, 3]`
- `smpl_pose`: `[N, 21, 3]`

For v3, only the six wrist joint values need to be meaningful in `joint_pos`;
the SMPL fields carry the main full-body motion. This matters because PICO4Ultra
SMPLX may not provide robot joint angles directly. The deployment bridge should
map PICO SMPLX output into SONIC-compatible:

- 24 SMPL-style joint positions.
- 21 body joint axis-angle rotations.
- Optional wrist joint hints.

If PICO gives full SMPLX, reduce it to the SONIC SMPL-compatible representation
instead of sending the full SMPLX parameter vector.

## What To Do With SOMA Uniform / Proportional

Minimum route:

- Keep `soma_uniform` and `soma_proportional` as source/reference data.
- Do not train a SOMA encoder in the first pass.
- Use them only to generate or validate the matched SMPL/SMPLX features.

Optional route:

- Use `sonic_bones_seed` and add the SOMA encoder later.
- Convert SOMA BVH to SOMA PKL using:

```bash
python gear_sonic/data_process/extract_soma_joints_from_bvh.py \
    --input /path/to/soma_uniform/bvh \
    --output data/agile_one_seed_wbc/soma_filtered \
    --fps 30 \
    --num_workers 16 \
    --skip_existing
```

This is optional because the target deployment input is PICO SMPLX, not SOMA BVH.

## Required SONIC Engineering Work

### 1. Add Agile One Embodiment

Add or modify:

- `gear_sonic/data/assets/robot_description/urdf/agile_one/`
- `gear_sonic/data/assets/robot_description/mjcf/agile_one_no_hands.xml`
- `gear_sonic/envs/manager_env/robots/agile_one.py`
- `gear_sonic/envs/manager_env/robots/__init__.py`
- `gear_sonic/envs/manager_env/modular_tracking_env_cfg.py`
- `gear_sonic/trl/utils/order_converter.py`
- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_agile_one_smplx.yaml`

Critical checks:

- IsaacLab joint names and MuJoCo joint names must be mapped correctly.
- IsaacLab body names and MuJoCo body names must be mapped correctly.
- Action dimension must match Agile One no-hands body DoF.
- Action scales must be based on Agile One actuator limits/stiffness.
- Spawn height and default joint pose must be stable.
- All reward/termination body names must exist on Agile One.

### 2. Add Agile One CSV -> MotionLib Converter

Input:

- Flat Agile One CSV from soma-retargeter.

Current CSV format is defined in:

- `/home/ruiming.wu/codes/general-soma-retargeter/soma_retargeter/assets/csv.py`

Agile One CSV fields:

- `Frame`
- root translation in centimeters
- root Euler rotation in degrees
- 29 joint angles in degrees

Output:

- Per-motion joblib PKL in SONIC motion_lib format.

Do not use the G1 hardcoded `DOF_AXIS`. Parse axes from Agile One MJCF or use
the robot FK parser.

### 3. Prepare Success-Only Manifest

Use previous benchmark files:

- `clips.csv`
- `warnings.csv`
- `failures.csv`
- `summary.json`

For the first training set:

- Include `status == success`.
- Exclude `warning`.
- Exclude `failed`.

Later:

- Add warnings as hard cases only after success-only training is stable.
- Keep failures out unless manually inspected.

### 4. Prepare SMPLX -> SONIC SMPL Data

The training and deployment representations must match. If deployment receives
PICO SMPLX, the training SMPL data should be generated using the same reduction
logic:

- SMPLX body pose -> SONIC `smpl_pose` compatible 21 joints.
- SMPLX joints -> SONIC 24-joint `smpl_joints`.
- Root orientation convention must match SONIC `smpl_y_up` / Z-up conversion.
- Frame rate should be 30 fps for training data; runtime can stream a future
  window but must match exported encoder expectations.

Do not train on one human representation and deploy another without a conversion
layer.

### 5. Add Agile One Training Config

Start from:

- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_h2.yaml`

or:

- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml`

Recommended first config changes:

```yaml
manager_env:
  config:
    robot:
      type: agile_one
  commands:
    motion:
      teleop_sample_prob_when_smpl: 0.0
      encoder_sample_probs:
        g1: 1.0
        teleop: 0.0
        smpl: 1.0
      motion_lib_cfg:
        motion_file: null
        smpl_motion_file: null
        smpl_y_up: true
        asset:
          assetFileName: "agile_one_no_hands.xml"
```

The name `g1` in the universal-token config should be understood as the
robot-joint/motion encoder branch. It does not have to mean Unitree G1 after
the embodiment is changed, but all body/joint dimensions and mappings must be
updated correctly.

## Training Stages

### Stage 0: Replay And Sanity Check

Purpose:

- Verify data conversion.
- Verify body/joint mapping.
- Verify Agile One loads in Isaac Lab.
- Verify root height, feet, wrists, and head are not scrambled.

Run with very small settings:

```bash
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_agile_one_smplx \
    ++replay=True \
    num_envs=4 \
    headless=False \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/agile_one_seed_wbc/robot_filtered \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/agile_one_seed_wbc/smpl_filtered
```

Do not train until replay is visually correct.

### Stage 1: Robot-Only Tracking Warm Start

Purpose:

- Teach or adapt the dynamic decoder to Agile One physics and action semantics.
- Avoid corrupting SMPL alignment before robot tracking is stable.

Suggested behavior:

- Sample only robot/G1-style branch.
- Disable teleop.
- Disable SMPL sampling or keep SMPL frozen.
- Use success-only motion clips.

Possible overrides:

```bash
++manager_env.commands.motion.encoder_sample_probs.g1=1.0 \
++manager_env.commands.motion.encoder_sample_probs.teleop=0.0 \
++manager_env.commands.motion.encoder_sample_probs.smpl=0.0 \
++manager_env.commands.motion.teleop_sample_prob_when_smpl=0.0
```

### Stage 2: Add SMPL / SMPLX Branch

Purpose:

- Align SMPL/SMPLX human input with Agile One robot-motion latent space.
- Make PICO SMPLX deployment meaningful.

Suggested behavior:

- Sample robot and SMPL branches.
- Keep teleop disabled.
- Keep quantizer frozen at first if loading from released SONIC.
- Train SMPL encoder and Agile One dynamic decoder.
- Consider freezing robot encoder initially if the robot motion branch is stable.

Possible overrides:

```bash
++manager_env.commands.motion.encoder_sample_probs.g1=1.0 \
++manager_env.commands.motion.encoder_sample_probs.teleop=0.0 \
++manager_env.commands.motion.encoder_sample_probs.smpl=1.0 \
++manager_env.commands.motion.teleop_sample_prob_when_smpl=0.0
```

### Stage 3: Low-LR Full Finetune

Purpose:

- Improve closed-loop robustness.
- Reduce residual tracking errors.

Suggested behavior:

- Unfreeze most actor modules except teleop branch.
- Use a lower learning rate than full training.
- Add warning clips only if success-only converges.
- Continue to exclude failed clips unless manually inspected.

## Checkpoint Loading Strategy

The released G1 checkpoint is useful, but Agile One is not G1.

Safe loading:

- Load SMPL encoder weights.
- Load G1/robot encoder weights if input dimensions match.
- Load FSQ quantizer.
- Load decoder hidden layers where shapes match.
- Reinitialize or carefully verify the final action layer if action semantics
  differ.

Risky loading:

- Blind full-state loading into Agile One if joint order/body order/action meaning
  changed.
- Keeping G1 reward body names or termination body names.
- Assuming matching 29 DoF means matching semantics.

## First Full Training Command Shape

After the Agile One embodiment and converters are ready:

```bash
cd /home/ruiming.wu/codes/GR00T-WholeBodyControl

accelerate launch --num_processes=8 gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_agile_one_smplx \
    +checkpoint=sonic_release/last.pt \
    num_envs=4096 \
    headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/agile_one_seed_wbc/robot_filtered \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/agile_one_seed_wbc/smpl_filtered \
    ++manager_env.commands.motion.encoder_sample_probs.g1=1.0 \
    ++manager_env.commands.motion.encoder_sample_probs.teleop=0.0 \
    ++manager_env.commands.motion.encoder_sample_probs.smpl=1.0 \
    ++manager_env.commands.motion.teleop_sample_prob_when_smpl=0.0
```

Start smaller before this:

```bash
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_agile_one_smplx \
    num_envs=16 \
    headless=False \
    ++algo.config.num_learning_iterations=100 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/agile_one_seed_wbc/robot_filtered \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/agile_one_seed_wbc/smpl_filtered
```

## Deployment Implications

The existing deployment path is G1-centric. For Agile One, the following must be
adapted:

- Robot communication/output backend.
- Observation config.
- Joint order and action order.
- Encoder mode semantics.
- SMPLX-to-SMPL conversion.
- Wrist joint hint indices.
- Reference motion format and visualizer.

For PICO full-body teleop, the relevant SONIC mode is SMPL encode mode:

- ZMQ protocol v3 sets encode mode 2.
- `smpl_joints` and `smpl_pose` carry the main human motion.
- `joint_pos` should at least carry meaningful wrist joint hints.

Do not depend on the official `teleop` encoder for this task. That branch is for
VR 3-point sparse tracking.

## Main Risks

- Agile One CSVs converted with the G1 converter will silently produce wrong
  motion libraries.
- SMPLX training representation and PICO runtime representation may differ.
- Agile One and G1 both having 29 DoF does not imply action semantic equality.
- Reward, termination, and body-name configs can still reference G1 bodies.
- Head joints in Agile One change the body/action semantics relative to G1.
- Warning clips can dominate hard cases and reduce stability if added too early.
- Root height / foot thickness differences can create persistent ground-contact
  artifacts.
- Full checkpoint loading can cause misleading early performance if the action
  head is semantically wrong.

## Minimum Viable Definition

The shortest meaningful path is:

1. Success-only Agile One CSVs from the five selected categories.
2. Agile One-specific CSV -> SONIC motion_lib converter.
3. Matched SMPLX/SMPL PKLs with the same motion keys.
4. Agile One embodiment registered in SONIC.
5. New `sonic_agile_one_smplx` config.
6. Teleop encoder disabled.
7. Two-stage finetune: robot-only tracking, then robot + SMPL.

Only after this works should we add warning clips, SOMA encoder experiments, or
more aggressive full-model finetuning.
