#!/usr/bin/env python3  # noqa: EXE001
# ruff: noqa: T201, DOC
"""Convert Agile One retargeted flat CSVs to SONIC motion_lib PKLs.

This is the Agile One counterpart of ``convert_soma_csv_to_motion_lib.py``.
It is intended for successful CSVs produced by our SOMA-retargeter Agile One
pipeline, e.g.:

    /home/ruiming.wu/data/seed-retargeted/ao_motionlib/raw_csv/<category>/*.csv

Input CSV format:
    Frame, root_translate{X,Y,Z}, root_rotate{X,Y,Z}, 29 Agile One joint DOFs

Units follow the G1/Bones-SEED flat CSV convention:
    - root translation: centimeters
    - root rotation: Euler XYZ degrees
    - joint DOFs: degrees, in Agile One MJCF actuator order

Output PKL format matches SONIC motion_lib expectations:
    {
        "motion_name": {
            "root_trans_offset": (T, 3) float32 meters,
            "pose_aa": (T, 30, 3) float32 axis-angle in MJCF body order,
            "dof": (T, 29) float32 radians in MJCF actuator order,
            "root_rot": (T, 4) float32 quaternion in scipy xyzw order,
            "smpl_joints": (T, 24, 3) float32 zeros placeholder,
            "fps": int,
        }
    }

The Agile One definition here is tied to:
    /home/ruiming.wu/codes/H4/mjcf/agile_one_no_hands.xml

That XML has 30 MuJoCo bodies in SkeletonTree preorder:
    pelvis_link + one body per actuated joint.

Examples:
    # Convert all category subdirectories, one PKL per CSV.
    python gear_sonic/data_process/convert_agile_one_csv_to_motion_lib.py \
        --input /home/ruiming.wu/data/seed-retargeted/ao_motionlib/raw_csv \
        --output /home/ruiming.wu/data/seed-retargeted/ao_motionlib/robot_motionlib \
        --fps 30 --fps_source 120 --individual --num_workers 16

    # Slow 2x for Agile One: keep 60 Hz samples but mark as 30 Hz.
    python gear_sonic/data_process/convert_agile_one_csv_to_motion_lib.py \
        --input /path/to/raw_csv \
        --output /path/to/robot_motionlib_slow2x \
        --fps 30 --fps_source 60 --individual --num_workers 16
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import joblib
import numpy as np
from scipy.spatial import transform


NUM_DOF = 29
NUM_BODIES = 30  # pelvis_link + 29 actuated links in agile_one_no_hands.xml

# Joint names in Agile One flat CSV / MJCF actuator order.
AGILE_ONE_CSV_JOINT_NAMES = [
    "left_hip_pitch_joint_dof",
    "left_hip_roll_joint_dof",
    "left_hip_yaw_joint_dof",
    "left_knee_joint_dof",
    "left_ankle_roll_joint_dof",
    "left_ankle_pitch_joint_dof",
    "right_hip_pitch_joint_dof",
    "right_hip_roll_joint_dof",
    "right_hip_yaw_joint_dof",
    "right_knee_joint_dof",
    "right_ankle_roll_joint_dof",
    "right_ankle_pitch_joint_dof",
    "waist_yaw_joint_dof",
    "head_yaw_joint_dof",
    "head_pitch_joint_dof",
    "left_shoulder_pitch_joint_dof",
    "left_shoulder_roll_joint_dof",
    "left_shoulder_yaw_joint_dof",
    "left_elbow_roll_joint_dof",
    "left_wrist_yaw_joint_dof",
    "left_wrist_roll_joint_dof",
    "left_wrist_pitch_joint_dof",
    "right_shoulder_pitch_joint_dof",
    "right_shoulder_roll_joint_dof",
    "right_shoulder_yaw_joint_dof",
    "right_elbow_roll_joint_dof",
    "right_wrist_yaw_joint_dof",
    "right_wrist_roll_joint_dof",
    "right_wrist_pitch_joint_dof",
]

# Single-axis joint definitions from agile_one_no_hands.xml, in the same order
# as AGILE_ONE_CSV_JOINT_NAMES and SkeletonTree body order excluding pelvis.
DOF_AXIS = np.array(
    [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],  # left leg
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],  # right leg
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],  # waist + head
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],  # left arm
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],  # right arm
    ],
    dtype=np.float32,
)

ROOT_COLUMNS = [
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
]


def _load_table_columns(csv_path: str) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load a CSV into column arrays.

    Prefer pandas when present for speed; fall back to numpy so the converter is
    still usable in lean motion-processing environments.
    """
    try:
        import pandas as pd

        frame = pd.read_csv(csv_path)
        return {c: frame[c].to_numpy() for c in frame.columns}, list(frame.columns)
    except ImportError:
        arr = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.float64, encoding=None)
        if arr.shape == ():
            arr = arr.reshape(1)
        columns = list(arr.dtype.names or [])
        return {c: arr[c] for c in columns}, columns


def load_agile_one_csv(csv_path: str) -> dict[str, np.ndarray]:
    """Load one Agile One flat CSV motion."""
    data, columns = _load_table_columns(csv_path)
    missing = [c for c in ["Frame", *ROOT_COLUMNS, *AGILE_ONE_CSV_JOINT_NAMES] if c not in data]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    joint_cols = [c for c in columns if c.endswith("_dof")]
    if joint_cols != AGILE_ONE_CSV_JOINT_NAMES:
        raise ValueError(
            "Unexpected Agile One joint column order in "
            f"{csv_path}. Expected {AGILE_ONE_CSV_JOINT_NAMES}, got {joint_cols}"
        )

    root_pos = (
        np.stack(
            [
                data["root_translateX"],
                data["root_translateY"],
                data["root_translateZ"],
            ],
            axis=1,
        ).astype(np.float32)
        / 100.0
    )

    euler_deg = np.stack(
        [
            data["root_rotateX"],
            data["root_rotateY"],
            data["root_rotateZ"],
        ],
        axis=1,
    ).astype(np.float64)
    root_quat_xyzw = transform.Rotation.from_euler("xyz", euler_deg, degrees=True).as_quat()
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]].astype(np.float32)

    joint_pos_mj = np.deg2rad(np.stack([data[c] for c in joint_cols], axis=1)).astype(np.float32)
    num_frames = joint_pos_mj.shape[0]

    body_pos_w = np.zeros((num_frames, 1, 3), dtype=np.float32)
    body_pos_w[:, 0, :] = root_pos
    body_quat_w = np.zeros((num_frames, 1, 4), dtype=np.float32)
    body_quat_w[:, 0, :] = root_quat_wxyz

    return {
        "joint_pos": joint_pos_mj,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
    }


def convert_sequence(seq_data: dict[str, Any], fps: int) -> dict[str, Any]:
    """Convert one Agile One sequence to SONIC motion_lib entry."""
    joint_pos = seq_data["joint_pos"]
    body_pos_w = seq_data["body_pos_w"]
    body_quat_w = seq_data["body_quat_w"]

    num_frames = joint_pos.shape[0]
    root_trans_offset = body_pos_w[:, 0, :].copy()
    root_quat_xyzw = body_quat_w[:, 0, :][:, [1, 2, 3, 0]]

    dof = joint_pos[:, :NUM_DOF]
    pose_aa = np.zeros((num_frames, NUM_BODIES, 3), dtype=np.float32)
    pose_aa[:, 1:NUM_BODIES, :] = DOF_AXIS[None, :, :] * dof[:, :, None]
    pose_aa[:, 0, :] = transform.Rotation.from_quat(root_quat_xyzw).as_rotvec()

    return {
        "root_trans_offset": root_trans_offset.astype(np.float32),
        "pose_aa": pose_aa.astype(np.float32),
        "dof": dof.astype(np.float32),
        "root_rot": root_quat_xyzw.astype(np.float32),
        "smpl_joints": np.zeros((num_frames, 24, 3), dtype=np.float32),
        "fps": fps,
    }


def downsample_sequence(entry: dict[str, Any], fps_source: int, fps_target: int) -> dict[str, Any]:
    """Stride-downsample an entry and store it at fps_target."""
    if fps_source == fps_target:
        return entry
    jump = int(fps_source / fps_target)
    if jump <= 1:
        return entry
    return {
        "root_trans_offset": entry["root_trans_offset"][::jump],
        "pose_aa": entry["pose_aa"][::jump],
        "dof": entry["dof"][::jump],
        "root_rot": entry["root_rot"][::jump],
        "smpl_joints": entry["smpl_joints"][::jump],
        "fps": fps_target,
    }


def process_session_csvs(args_tuple):
    """Process all flat CSVs in one category/session directory."""
    session_dir, session_name, out_dir, fps, fps_source = args_tuple
    csv_files = sorted(f for f in os.listdir(session_dir) if f.endswith(".csv"))

    session_out = os.path.join(out_dir, session_name)
    os.makedirs(session_out, exist_ok=True)

    converted = 0
    failed = 0
    for csv_f in csv_files:
        name = os.path.splitext(csv_f)[0]
        out_path = os.path.join(session_out, name + ".pkl")
        if os.path.exists(out_path):
            converted += 1
            continue
        try:
            seq = load_agile_one_csv(os.path.join(session_dir, csv_f))
            fps_for_convert = fps_source if fps_source else fps
            entry = convert_sequence(seq, fps_for_convert)
            if fps_source and fps_source != fps:
                entry = downsample_sequence(entry, fps_source, fps)
            joblib.dump({name: entry}, out_path, compress=True)
            converted += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAILED {session_name}/{csv_f}: {exc}", file=sys.stderr)
    return session_name, converted, failed, len(csv_files)


def _discover_sessions(input_dir: str, output_dir: str, fps: int, fps_source: int | None):
    has_csvs = any(f.endswith(".csv") for f in os.listdir(input_dir))
    subdirs = sorted(d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d)))
    has_session_subdirs = (
        any(any(f.endswith(".csv") for f in os.listdir(os.path.join(input_dir, d))) for d in subdirs)
        if subdirs
        else False
    )

    if has_session_subdirs:
        return [
            (os.path.join(input_dir, d), d, output_dir, fps, fps_source)
            for d in subdirs
            if any(f.endswith(".csv") for f in os.listdir(os.path.join(input_dir, d)))
        ]
    if has_csvs:
        return [(input_dir, os.path.basename(input_dir.rstrip("/")), output_dir, fps, fps_source)]
    return []


def main():
    parser = argparse.ArgumentParser(description="Convert Agile One flat CSVs to motion_lib PKLs")
    parser.add_argument("--input", required=True, help="CSV dir or parent dir of CSV dirs")
    parser.add_argument("--output", required=True, help="Output PKL file or directory")
    parser.add_argument("--fps", type=int, default=30, help="Target output FPS")
    parser.add_argument(
        "--fps_source",
        type=int,
        default=None,
        help="Source CSV FPS. Use 120->30 for normal speed, 60->30 for 2x slow motion.",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Write one PKL per CSV, preserving category/session subdirectories.",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Workers for --individual mode")
    args = parser.parse_args()

    print(f"Agile One {NUM_DOF} DOFs, {NUM_BODIES} bodies (agile_one_no_hands.xml)")

    if args.individual:
        if not os.path.isdir(args.input):
            print("ERROR: --individual requires a directory input")
            sys.exit(1)
        sessions = _discover_sessions(args.input, args.output, args.fps, args.fps_source)
        if not sessions:
            print(f"ERROR: no CSV sessions found under {args.input}")
            sys.exit(1)

        print(f"\nBatch converting {len(sessions)} sessions with {args.num_workers} workers")
        print(f"Output: {args.output}")
        os.makedirs(args.output, exist_ok=True)

        import multiprocessing

        total_converted = 0
        total_failed = 0
        total_csvs = 0
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            for session_name, converted, failed, n_csvs in pool.imap_unordered(
                process_session_csvs, sessions
            ):
                total_converted += converted
                total_failed += failed
                total_csvs += n_csvs
                print(
                    f"  {session_name}: {converted}/{n_csvs} converted"
                    + (f" ({failed} failed)" if failed else "")
                )

        print(
            f"\nDone: {total_converted} motions converted, {total_failed} failed, {total_csvs} total CSVs"
        )
        return

    if not os.path.isdir(args.input):
        print("ERROR: combined mode requires a directory of flat CSVs")
        sys.exit(1)

    csv_files = sorted(f for f in os.listdir(args.input) if f.endswith(".csv"))
    if not csv_files:
        print(f"ERROR: no CSV files found directly under {args.input}")
        sys.exit(1)

    motion_lib_dict = {}
    for csv_f in csv_files:
        name = os.path.splitext(csv_f)[0]
        seq = load_agile_one_csv(os.path.join(args.input, csv_f))
        fps_for_convert = args.fps_source if args.fps_source else args.fps
        entry = convert_sequence(seq, fps_for_convert)
        if args.fps_source and args.fps_source != args.fps:
            entry = downsample_sequence(entry, args.fps_source, args.fps)
        motion_lib_dict[name] = entry
        print(f"  Converted {name}: {entry['dof'].shape[0]} frames @ {entry['fps']} fps")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    print(f"\nSaving motion_lib PKL: {args.output}")
    joblib.dump(motion_lib_dict, args.output, compress=True)
    print(f"Done: {len(motion_lib_dict)} sequences saved")


if __name__ == "__main__":
    main()
