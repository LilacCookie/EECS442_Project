"""
yoga_scoring.py
================================
3D NTU-25 Skeleton-based Yoga Pose Evaluation Module
================================

Main Features:
- Process 3D skeleton sequences of NTU 25 joints:
  - Spatial normalization (translation, scale, orientation)
  - Temporal resampling alignment
  - Pose position error
  - Joint angle error
  - Range of motion (maximum angle) error
  - Stability (jitter) error
- Map multiple errors to a total score and sub-scores ranging from 0 to 100

Usage example can be found at the bottom in `if __name__ == "__main__":`.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np

from mmcv import load

# ============================================================
# 1. NTU 25 Joint Index Definitions (0-based)
# ============================================================

NTU25_JOINTS: Dict[str, int] = {
    "spine_base": 0,
    "spine_mid": 1,
    "neck": 2,
    "head": 3,
    "left_shoulder": 4,
    "left_elbow": 5,
    "left_wrist": 6,
    "left_hand": 7,
    "right_shoulder": 8,
    "right_elbow": 9,
    "right_wrist": 10,
    "right_hand": 11,
    "left_hip": 12,
    "left_knee": 13,
    "left_ankle": 14,
    "left_foot": 15,
    "right_hip": 16,
    "right_knee": 17,
    "right_ankle": 18,
    "right_foot": 19,
    "spine_shoulder": 20,
    "left_hand_tip": 21,
    "left_thumb": 22,
    "right_hand_tip": 23,
    "right_thumb": 24,
}

# ============================================================
# 2. Configuration Class: Control Scoring Parameters
# ============================================================


@dataclass
class YogaScoringConfig:
    """
    Yoga Scoring Configuration

    L: Number of frames after resampling
    root_idx: Root joint (used for translation and stability analysis)
    left_hip_idx, right_hip_idx: Used for orientation alignment
    joint_weights: Weights for each joint in position error (length 25), None for automatic generation
    angle_triplets: List of joint angle triplets (p1, p2, p3) with p2 as the vertex
    E_pos_max: Maximum tolerance for position error, beyond which the score is 0
    E_ang_max_deg: Maximum tolerance for angle error (degrees)
    E_rom_max_deg: Maximum tolerance for range of motion (maximum angle) error (degrees)
    E_stab_max: Maximum tolerance for stability (jitter) error
    w_*: Weights for each sub-score (sum to 1.0)
    """

    L: int = 64

    root_idx: int = NTU25_JOINTS["spine_base"]
    left_hip_idx: int = NTU25_JOINTS["left_hip"]
    right_hip_idx: int = NTU25_JOINTS["right_hip"]

    joint_weights: Optional[np.ndarray] = None
    angle_triplets: List[Tuple[int, int, int]] = field(default_factory=list)

    E_pos_max: float = 0.5          # Average position error ~ 0.25 (normalized height)
    E_ang_max_deg: float = 25.0      # Average angle error 25°
    E_rom_max_deg: float = 30.0      # Maximum angle error 30°
    E_stab_max: float = 0.05         # Jitter error (adjustable based on experience)

    w_pos: float = 0.4
    w_ang: float = 0.3
    w_rom: float = 0.2
    w_stab: float = 0.1

    def __post_init__(self):
        # If joint weights are not provided, default: higher weights for torso/limbs, lower for fingers/thumbs
        if self.joint_weights is None:
            w = np.ones(25, dtype=np.float32)

            # Torso (spine / neck / head) weights slightly higher
            for name in ["spine_base", "spine_mid", "spine_shoulder", "neck", "head"]:
                w[NTU25_JOINTS[name]] = 2.0

            # 大关节（肩、肘、髋、膝、踝）
            for name in [
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
            ]:
                w[NTU25_JOINTS[name]] = 1.0

            # Hand details weights slightly lower
            for name in [
                "left_hand",
                "right_hand",
                "left_hand_tip",
                "right_hand_tip",
                "left_thumb",
                "right_thumb",
            ]:
                w[NTU25_JOINTS[name]] = 0.0

            self.joint_weights = w

        # If angle triplets are not provided, give a common set for elbows, knees, hips, and shoulders
        if not self.angle_triplets:
            self.angle_triplets = [
                # Left and right elbows: shoulder - elbow - wrist
                (NTU25_JOINTS["left_shoulder"], NTU25_JOINTS["left_elbow"], NTU25_JOINTS["left_wrist"]),
                (NTU25_JOINTS["right_shoulder"], NTU25_JOINTS["right_elbow"], NTU25_JOINTS["right_wrist"]),
                # Left and right knees: hip - knee - ankle
                (NTU25_JOINTS["left_hip"], NTU25_JOINTS["left_knee"], NTU25_JOINTS["left_ankle"]),
                (NTU25_JOINTS["right_hip"], NTU25_JOINTS["right_knee"], NTU25_JOINTS["right_ankle"]),
                # Left and right hips: spine_mid - hip - knee (rough representation of hip angle)
                (NTU25_JOINTS["spine_mid"], NTU25_JOINTS["left_hip"], NTU25_JOINTS["left_knee"]),
                (NTU25_JOINTS["spine_mid"], NTU25_JOINTS["right_hip"], NTU25_JOINTS["right_knee"]),
                # Left and right shoulders: spine_shoulder - shoulder - elbow
                (NTU25_JOINTS["spine_shoulder"], NTU25_JOINTS["left_shoulder"], NTU25_JOINTS["left_elbow"]),
                (NTU25_JOINTS["spine_shoulder"], NTU25_JOINTS["right_shoulder"], NTU25_JOINTS["right_elbow"]),
            ]


# ============================================================
# 3. Basic utility functions: normalization & resampling
# ============================================================


def normalize_skeleton_sequence(
    seq: np.ndarray,
    root_idx: int,
    left_hip_idx: int,
    right_hip_idx: int,
    vertical_axis: int = 1,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Normalize a single skeleton sequence spatially:
    1) Translation: root joint to origin
    2) Scaling: normalize by height (vertical coordinate range) to 1
    3) Orientation: use left and right hips to define horizontal direction, rotate around vertical_axis to align with +x axis

    Parameters
    ----------
    seq : [T, 25, 3]
    root_idx, left_hip_idx, right_hip_idx : Joint indices
    vertical_axis : Axis representing the vertical direction in the coordinate system, common values:
        - If y is vertical: vertical_axis = 1
        - If z is vertical: vertical_axis = 2

    Returns
    -------
    norm_seq : [T, 25, 3] Normalized sequence
    """
    seq = np.asarray(seq, dtype=np.float32).copy()
    T, J, C = seq.shape
    assert J == 25 and C == 3, "Expected input shape [T, 25, 3]"
    # 1) Translation: root joint to origin
    root = seq[:, root_idx:root_idx + 1, :]  # [T,1,3]
    seq = seq - root

    # 2) Scaling: normalize by height (vertical coordinate range)
    vertical_vals = seq[..., vertical_axis]
    body_height = float(vertical_vals.max() - vertical_vals.min())
    scale = body_height if body_height > eps else 1.0
    seq = seq / scale

    # 3) Orientation alignment: project left and right hips onto the horizontal plane, rotate to x-axis direction
    # Here we assume the horizontal plane is (x, z), i.e., axis 0 and 2
    # If your data coordinates are different, you can modify accordingly
    l_hip = seq[0, left_hip_idx]
    r_hip = seq[0, right_hip_idx]
    hip_vec = r_hip - l_hip  # [3]
    # Project onto (x, z) plane
    hip_vec[vertical_axis] = 0.0
    hip_norm = np.linalg.norm(hip_vec) + eps
    hip_dir = hip_vec / hip_norm

    # Target direction: along +x axis
    # Rotate only around vertical_axis
    # Here we assume vertical_axis == 1 (y axis)
    hx = hip_dir[0]
    hz = hip_dir[2]
    yaw = np.arctan2(hz, hx)  # Current angle relative to x axis
    # We want to rotate hip_dir to (1,0,0), so the rotation angle is -yaw

    cos_y = np.cos(-yaw)
    sin_y = np.sin(-yaw)

    # Rotation matrix around y axis (adjust if vertical_axis != 1)
    R = np.array(
        [
            [cos_y, 0.0, sin_y],
            [0.0, 1.0, 0.0],
            [-sin_y, 0.0, cos_y],
        ],
        dtype=np.float32,
    )

    # Apply rotation
    seq = np.einsum("ij,tkj->tki", R, seq)

    return seq


def resample_sequence(seq: np.ndarray, L: int) -> np.ndarray:
    """
    Resample a skeleton sequence of length T to a fixed length L.
    
    Parameters
    ----
    seq : [T, 25, 3]
    L   : New length

    Returns
    ----
    seq_new : [L, 25, 3]
    """
    seq = np.asarray(seq, dtype=np.float32)
    T, J, C = seq.shape
    if T == L:
        return seq.copy()

    t_old = np.linspace(0.0, 1.0, T)
    t_new = np.linspace(0.0, 1.0, L)

    seq_new = np.zeros((L, J, C), dtype=np.float32)
    for j in range(J):
        for d in range(C):
            seq_new[:, j, d] = np.interp(t_new, t_old, seq[:, j, d])

    return seq_new


# ============================================================
# 4. Error and Scoring Utility Functions
# ============================================================

def positional_error(std_seq: np.ndarray, user_seq: np.ndarray, joint_weights: np.ndarray) -> float:
    """
    Positional error: average Euclidean distance between two aligned skeleton sequences.

    Parameters
    ----
    std_seq, user_seq : [L, 25, 3]
    joint_weights     : [25], weights for each joint

    Returns
    ----
    E_pos : float, average positional error
    """
    std_seq = np.asarray(std_seq, dtype=np.float32)
    user_seq = np.asarray(user_seq, dtype=np.float32)
    assert std_seq.shape == user_seq.shape
    L, J, C = std_seq.shape
    assert J == 25 and C == 3
    assert joint_weights.shape[0] == 25

    diff = user_seq - std_seq          # [L,25,3]
    dist = np.linalg.norm(diff, axis=-1)  # [L,25]

    # Weighted average: first weight by joint weights, then average over time
    weighted = dist * joint_weights[None, :]
    E = weighted.sum() / joint_weights.sum() / L
    return float(E)


def joint_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute the joint angle at p2 (p1 - p2 - p3), return in radians.
    """
    v1 = p1 - p2
    v2 = p3 - p2
    n1 = np.linalg.norm(v1) + eps
    n2 = np.linalg.norm(v2) + eps
    cos_t = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.arccos(cos_t))


def angle_error(
    std_seq: np.ndarray,
    user_seq: np.ndarray,
    angle_triplets: List[Tuple[int, int, int]],
) -> float:
    """
    Angle error: For specified triplets (p1, p2, p3), compute joint angles at each frame,
    then compare the differences between std_seq and user_seq, returning the average angle difference (radians).

    Parameters
    ----
    std_seq, user_seq : [L, 25, 3]
    angle_triplets    : List[(p1_idx, p2_idx, p3_idx)]

    Returns
    ----
    E_ang : float, average angle error (radians)
    """
    std_seq = np.asarray(std_seq, dtype=np.float32)
    user_seq = np.asarray(user_seq, dtype=np.float32)
    assert std_seq.shape == user_seq.shape
    L, J, C = std_seq.shape
    assert J == 25 and C == 3

    errors = []
    for t in range(L):
        for (i1, i2, i3) in angle_triplets:
            th_std = joint_angle(std_seq[t, i1], std_seq[t, i2], std_seq[t, i3])
            th_usr = joint_angle(user_seq[t, i1], user_seq[t, i2], user_seq[t, i3])
            errors.append(abs(th_std - th_usr))
    if not errors:
        return 0.0
    return float(np.mean(errors))


def rom_error(
    std_seq: np.ndarray,
    user_seq: np.ndarray,
    angle_triplets: List[Tuple[int, int, int]],
) -> float:
    """
    ROM (Range of Motion) amplitude error:
    For each angle triplet, compute the "maximum angle" over the entire motion,
    then compare the maximum angle difference between the standard and user,
    and finally average over all triplets.

    Returns
    """
    std_seq = np.asarray(std_seq, dtype=np.float32)
    user_seq = np.asarray(user_seq, dtype=np.float32)
    assert std_seq.shape == user_seq.shape
    L, J, C = std_seq.shape
    assert J == 25 and C == 3

    std_max_list = []
    usr_max_list = []

    for (i1, i2, i3) in angle_triplets:
        std_angles = []
        usr_angles = []
        for t in range(L):
            th_std = joint_angle(std_seq[t, i1], std_seq[t, i2], std_seq[t, i3])
            th_usr = joint_angle(user_seq[t, i1], user_seq[t, i2], user_seq[t, i3])
            std_angles.append(th_std)
            usr_angles.append(th_usr)
        std_max_list.append(max(std_angles))
        usr_max_list.append(max(usr_angles))

    if not std_max_list:
        return 0.0

    std_max = np.array(std_max_list)
    usr_max = np.array(usr_max_list)
    diff = np.abs(std_max - usr_max)
    return float(diff.mean())


def stability_error(
    seq: np.ndarray,
    root_idx: int,
    vertical_axis: int = 1,
) -> float:
    """
    Stability error: measure "jitter" by the velocity variation of the root joint on the horizontal plane.

    Simple approach:
    - Take the trajectory of the root joint on the (x,z) plane (or (x,y), depending on the vertical axis)
    - Calculate the displacement between adjacent frames v_t = ||c_{t+1} - c_t||
    - Return the standard deviation or variance of v_t as the stability error.

    Larger return value => more jitter => worse stability.
    """
    seq = np.asarray(seq, dtype=np.float32)
    T, J, C = seq.shape
    assert J == 25 and C == 3

    root_traj = seq[:, root_idx]  # [T,3]

    # Horizontal plane: remove the vertical axis
    if vertical_axis == 1:  # y axis vertical => use (x,z)
        horiz = root_traj[:, [0, 2]]
    elif vertical_axis == 2:  # z axis vertical => use (x,y)
        horiz = root_traj[:, [0, 1]]
    else:  # generally not used, fallback to (x,z)
        horiz = root_traj[:, [0, 2]]

    if T < 2:
        return 0.0

    # Velocity (displacement)
    v = np.linalg.norm(horiz[1:] - horiz[:-1], axis=-1)  # [T-1]
    # Use standard deviation as jitter metric
    return float(np.std(v))


def score_from_error(E: float, E_max: float) -> float:
    # """
    # Error -> Score (0~100)
    
    # E=0 => 100 points
    # E>=E_max => 0 points
    # Linear interpolation in between
    # """
    # if E_max <= 0:
    #     return 0.0
    # s = 1.0 - E / E_max
    # s = float(np.clip(s, 0.0, 1.0))
    # return s * 100.0
    """
    Piecewise deduction function based on error percentage:

    - 0   ~ p1   : 100 points       (no deduction)
    - p1  ~ p2   : from 100 → 90   (light deduction)
    - p2  ~ p3   : from 90 → 60    (medium deduction)
    - p3  ~ 1.0  : from 60 → 0     (heavy deduction)
    - >= 1.0     : 0 points

    Parameters
    ----
    E: Actual error
    E_max: Maximum error
    p1, p2, p3: Three segment boundary points (percentages)

    Returns a score between 0 and 100
    """
    p1: float = 0.20
    p2: float = 0.50
    p3: float = 0.75
    if E_max <= 0:
        return 0.0  # Avoid division by zero

    r = np.abs(E / E_max)  # Percentage
    
    if r <= 0.1:
        return 100.0
    
    return max(0.0, 100.0 * (1.0 - 0.5 * r))
    # # Above 0~100%
    # if r <= 0:
    #     return 100.0
    # if r >= 1.0:
    #     return 0.0

    # # Segment 1: 0 ~ p1 (100 points)
    # if r <= p1:
    #     return 100.0

    # # Segment 2: p1 ~ p2 (100 → 90)
    # if r <= p2:
    #     t = (r - p1) / (p2 - p1)
    #     return 100.0 - t * 10.0  # 100 → 90

    # # Segment 3: p2 ~ p3 (90 → 60)
    # if r <= p3:
    #     t = (r - p2) / (p3 - p2)
    #     return 90.0 - t * 30.0   # 90 → 60

    # # Segment 4: p3 ~ 1.0 (60 → 0)
    # t = (r - p3) / (1.0 - p3)
    # return 60.0 - t * 60.0        # 60 → 0


# ============================================================
# 5. Template Construction & Main Scoring Function    
# ============================================================


def build_template_from_list(
    std_seqs: List[np.ndarray],
    cfg: YogaScoringConfig,
    vertical_axis: int = 1,
) -> np.ndarray:
    """
    Build a template from multiple "standard demonstration" skeleton sequences:
    - Normalize each sequence spatially
    - Resample to cfg.L frames
    - Take the pointwise average of all demonstrations to form the template

    Parameters
    ----
    std_seqs : List[[T_i, 25, 3]]
    cfg      : YogaScoringConfig
    vertical_axis : Vertical axis (same as normalize_skeleton_sequence)

    Returns
    ----
    template_seq : [L, 25, 3]
    """
    processed = []
    for seq in std_seqs:
        seq_norm = normalize_skeleton_sequence(
            seq,
            root_idx=cfg.root_idx,
            left_hip_idx=cfg.left_hip_idx,
            right_hip_idx=cfg.right_hip_idx,
            vertical_axis=vertical_axis,
        )
        seq_rs = resample_sequence(seq_norm, cfg.L)
        processed.append(seq_rs)

    if not processed:
        raise ValueError("std_seqs is empty, cannot build template.")

    stacked = np.stack(processed, axis=0)  # [K,L,25,3]
    template = stacked.mean(axis=0)        # [L,25,3]
    return template.astype(np.float32)


def score_yoga_sequence(
    std_template_seq: np.ndarray,
    user_seq: np.ndarray,
    cfg: YogaScoringConfig,
    vertical_axis: int = 1,
) -> Tuple[float, Dict[str, float]]:
    """
    Main scoring function: Given a "standard template sequence" and a "user sequence" for a certain action category,
    calculate the total score and scores for each sub-item.

    Parameters
    ----
    std_template_seq : [L0, 25, 3]
        Standard template sequence for the pose (can be an average of multiple demonstrations or a single standard demonstration)
    user_seq         : [T_user, 25, 3]
        User action sequence
    cfg              : YogaScoringConfig
    vertical_axis    : Vertical axis index (1: y-axis vertical; 2: z-axis vertical)

    Returns
    ----
    total_score : float  0~100
    detail      : dict   Contains scores and errors for each sub-item
    """
    # 1) Normalize and resample the standard template and user sequences
    std_norm = normalize_skeleton_sequence(
        std_template_seq,
        root_idx=cfg.root_idx,
        left_hip_idx=cfg.left_hip_idx,
        right_hip_idx=cfg.right_hip_idx,
        vertical_axis=vertical_axis,
    )
    user_norm = normalize_skeleton_sequence(
        user_seq,
        root_idx=cfg.root_idx,
        left_hip_idx=cfg.left_hip_idx,
        right_hip_idx=cfg.right_hip_idx,
        vertical_axis=vertical_axis,
    )

    std_rs = resample_sequence(std_norm, cfg.L)
    user_rs = resample_sequence(user_norm, cfg.L)

    # 2) Positional error & positional score
    E_pos = positional_error(std_rs, user_rs, cfg.joint_weights)
    S_pos = score_from_error(E_pos, cfg.E_pos_max)

    # 3) Angular error (average angle difference) & angular score
    E_ang_rad = angle_error(std_rs, user_rs, cfg.angle_triplets)
    E_ang_deg = E_ang_rad * 180.0 / np.pi
    S_ang = score_from_error(E_ang_deg, cfg.E_ang_max_deg)

    # 4) ROM error (maximum angle difference) & ROM score
    E_rom_rad = rom_error(std_rs, user_rs, cfg.angle_triplets)
    E_rom_deg = E_rom_rad * 180.0 / np.pi
    S_rom = score_from_error(E_rom_deg, cfg.E_rom_max_deg)

    # 5) Stability error (jitter) & stability score
    E_stab = stability_error(user_rs, cfg.root_idx, vertical_axis=vertical_axis)
    S_stab = score_from_error(E_stab, cfg.E_stab_max)

    # 6) Overall score (weighted sum)
    total = (
        cfg.w_pos * S_pos
        + cfg.w_ang * S_ang
        + cfg.w_rom * S_rom
        + cfg.w_stab * S_stab
    )

    detail = {
        "S_pos": S_pos,
        "S_ang": S_ang,
        "S_rom": S_rom,
        "S_stab": S_stab,
        "E_pos": E_pos,
        "E_ang_deg": E_ang_deg,
        "E_rom_deg": E_rom_deg,
        "E_stab": E_stab,
    }

    return float(total), detail


# ============================================================
# 6. Optional: Example pkl loading function (modify according to your actual format)
# ============================================================


def load_ntu_pkl_skeleton(pkl_obj) -> np.ndarray:
    """
    Given an object parsed from a pkl file, return the skeleton sequence as a numpy array [T, 25, 3].

    This is highly dependent on your own data structure, please modify according to your pkl format.

    For example:
    - Some are dicts with a 'keypoint' field, shape (T, 25, 3)
    - Some have (C, T, V, M) structure, need to be unpacked

    Here is a common example:
    """
    # 假设 pkl_obj 是 dict，pkl_obj['keypoint'] shape: [T, 25, 3]
    if isinstance(pkl_obj, dict) and "keypoint" in pkl_obj:
        kp = pkl_obj["keypoint"]
        arr = np.asarray(kp, dtype=np.float32)
        assert arr.shape[2] == 25 and arr.shape[3] == 3, "Expected keypoint to be [T,25,3]"
        return arr[0]

    raise ValueError("Please implement load_ntu_pkl_skeleton() according to your pkl format")


# ============================================================
# 7. example usage
# ============================================================
yoga_poses = {
    0: "chair",
    1: "goddess",
    4: "chair",
    9: "downward-dog",
}


if __name__ == "__main__":
    data = load('/root/autodl-tmp/EECS442 project/pyskl/tools/data/3dyoga/3dyoga_annotations2.pkl') # standard data
    # test_data = load('/root/autodl-tmp/test.pkl') # test data
    # results = load('/root/autodl-tmp/EECS442 project/pyskl/result_test.pkl')
    
    user_seq_example = load_ntu_pkl_skeleton(data['annotations'][1005])
    
    std_seqs_example = []
    for i in range(1020, 1030):
        item = data['annotations'][i]
        if item['label'] == 9:
            std_seqs_example.append(load_ntu_pkl_skeleton(item))
    
    cfg = YogaScoringConfig(
        L=64,
        E_pos_max = 0.8,
        E_ang_max_deg=30.0,
        E_rom_max_deg=30.0,
        E_stab_max=0.05,
        w_pos=0.3,
        w_ang=0.4,
        w_rom=0.2,
        w_stab=0.1,
    )
    
    template = build_template_from_list(std_seqs_example, cfg, vertical_axis=1)
        
    total_score, detail_scores = score_yoga_sequence(
        template,
        user_seq_example,
        cfg,
        vertical_axis=1,
    )
    
    print(f"Total score: {total_score:.2f}")
    for k, v in detail_scores.items():
        print(f"  {k}: {v:.4f}")
    