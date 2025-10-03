from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Dict

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import math
from dataclasses import MISSING

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv

def hand_positions(env: ManagerBasedEnv) -> torch.Tensor:
    """Get positions of robot hands.
    
    Returns:
        torch.Tensor: Flattened tensor of shape (N, 6) containing [left_x, left_y, left_z, right_x, right_y, right_z]
    """
    robot = env.scene["robot"]
    hand_cfg = SceneEntityCfg("robot", body_names=["left_palm_link", "right_palm_link"])
    hand_pos = robot.data.body_state_w[:, hand_cfg.body_ids, :3]  # Should be shape (64, 2, 3)
    # print(f"Original hand_pos shape: {hand_pos.shape}")  # Debug print
    
    flattened = hand_pos.reshape(hand_pos.shape[0], -1)  # Should become (64, 6)
    # print(f"Flattened shape: {flattened.shape}")  # Debug print
    
    return flattened

def object_held(env: ManagerBasedEnv) -> torch.Tensor:
    """Binary indicator if object is being held."""
    target_object = env.scene["target_object"]
    robot = env.scene["robot"]
    
    object_pos = target_object.data.root_pos_w
    hand_cfg = SceneEntityCfg("robot", body_names=["left_palm_link", "right_palm_link"])
    hand_pos = robot.data.body_state_w[:, hand_cfg.body_ids, :3]
    
    distances = torch.norm(object_pos.unsqueeze(1) - hand_pos, dim=-1)
    holding = torch.any(distances < 0.1, dim=-1)
    return holding.float().unsqueeze(-1)

def get_scene_object_positions(env: ManagerBasedEnv) -> Dict[str, torch.Tensor]:
    """Get the positions of objects in the scene.

    Args:
        env: The environment instance.

    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping object names (Object1-10) to position tensors 
                                with shape (num_envs, 3) containing (x, y, z) coordinates.
    """
    positions_dict = {}
    
    # Iterate through objects 1-10
    for i in range(1, 11):
        object_name = f"Object{i}"
        # Get the object directly from the scene
        obj = getattr(env.scene, object_name)
        # Get position through the data attribute
        positions_dict[object_name] = obj.data.root_pos_w
    
    return positions_dict

def get_object_features(env: ManagerBasedEnv) -> torch.Tensor:
    """Get object features including type (one-hot), size parameters, and positions.
    Orders objects based on attention mechanism, vectorized for better performance.
    """
    num_envs = env.num_envs
    device = env.device
    
    # Get robot head position and forward direction (unchanged)
    robot = env.scene["robot"]
    head_pos = robot.data.body_state_w[:, robot.body_names.index("head_link"), :3]
    head_rot = robot.data.body_state_w[:, robot.body_names.index("head_link"), 3:7]
    forward_dir = quat_apply_inverse(head_rot, torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32).expand(num_envs, 3))

    # Pre-allocate tensors for all objects
    all_features = torch.zeros((num_envs, 5, 11), device=device)  # Shape: (num_envs, num_objects, feature_dim)
    all_scores = torch.zeros((num_envs, 5), device=device)
    all_relevant = torch.zeros((num_envs, 5), dtype=torch.bool, device=device)
    
    # Process all objects at once
    for i in range(1, 6):
        object_name = f"Object{i}"
        obj = env.scene[object_name]
        obj_cfg = obj.cfg.spawn
        positions = obj.data.root_pos_w
        
        # Calculate relative positions and directions
        rel_pos = positions - head_pos
        # Transform relative position to robot's local frame
        local_pos = quat_apply_inverse(head_rot, rel_pos)
        
        # Calculate attention metrics vectorized
        distance = torch.norm(local_pos, dim=-1)
        direction = local_pos / (distance.unsqueeze(-1) + 1e-8)
        angle = torch.acos(torch.clamp(torch.sum(direction * forward_dir, dim=-1), -1.0, 1.0)).abs()
        
        # Set features for current object
        if isinstance(obj_cfg, sim_utils.SphereCfg):
            all_features[:, i-1, 0] = 1  # one-hot
            all_features[:, i-1, 5] = obj_cfg.radius  # size param
            all_features[:, i-1, 6:8] = 1e-10  # remaining size params
        elif isinstance(obj_cfg, sim_utils.CuboidCfg):
            all_features[:, i-1, 1] = 1
            all_features[:, i-1, 5:8] = torch.tensor(obj_cfg.size, device=device)
        elif isinstance(obj_cfg, sim_utils.CylinderCfg):
            all_features[:, i-1, 2] = 1
            all_features[:, i-1, 5] = obj_cfg.radius
            all_features[:, i-1, 6] = obj_cfg.height
            all_features[:, i-1, 7] = 1e-10
        elif isinstance(obj_cfg, sim_utils.ConeCfg):
            all_features[:, i-1, 3] = 1
            all_features[:, i-1, 5] = obj_cfg.radius
            all_features[:, i-1, 6] = obj_cfg.height
            all_features[:, i-1, 7] = 1e-10
        elif isinstance(obj_cfg, sim_utils.CapsuleCfg):
            all_features[:, i-1, 4] = 1
            all_features[:, i-1, 5] = obj_cfg.radius
            all_features[:, i-1, 6] = obj_cfg.height
            all_features[:, i-1, 7] = 1e-10
        
        # Set positions (now using local coordinates instead of global)
        all_features[:, i-1, 8:] = local_pos
        
        # Set scores and relevance
        all_scores[:, i-1] = angle + 0.5 * distance
        all_relevant[:, i-1] = torch.any(all_features[:, i-1, 5:8] > 0.005, dim=1)
    
    # Vectorized sorting
    # Create indices tensor for sorting
    indices = torch.arange(5, device=device).expand(num_envs, -1)
    
    # Create sorting keys: relevant objects first, then by score
    sorting_keys = (~all_relevant).float() * 1000 + all_scores
    
    # Sort indices based on keys
    sorted_indices = torch.argsort(sorting_keys, dim=1)
    
    # Use sorted indices to reorder features
    batch_indices = torch.arange(num_envs, device=device).unsqueeze(1).expand(-1, 5)
    sorted_features = all_features[batch_indices, sorted_indices]
    
    return sorted_features.reshape(num_envs, -1)

def get_robot_body_positions(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Get the positions of key robot body parts.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot, specifying the body names.

    Returns:
        torch.Tensor: Flattened position tensor with shape (num_envs, 30) containing x,y,z coordinates
                     for hands, feet, pelvis, knees, elbows, and head concatenated into a single vector.
    """
    # Define the key body parts we want to track
    key_body_parts = [
        'left_palm_link',      # Left hand
        'right_palm_link',     # Right hand
        'left_ankle_roll_link', # Left foot
        'right_ankle_roll_link',# Right foot
        'pelvis',              # Pelvis
        'left_knee_link',      # Left knee
        'right_knee_link',     # Right knee
        'left_elbow_pitch_link',# Left elbow
        'right_elbow_pitch_link',# Right elbow
        'head_link'            # Head
    ]
    
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get the body names from the articulation if not already set
    if asset_cfg.body_names is None:
        asset_cfg.body_names = asset.body_names
    
    # Get indices of the key body parts
    key_body_indices = [asset_cfg.body_names.index(name) for name in key_body_parts]
    
    # Get positions and flatten the last two dimensions
    positions = asset.data.body_pos_w[:, key_body_indices]  # Shape: (num_envs, 10, 3)
    return positions.reshape(positions.shape[0], -1)  # Shape: (num_envs, 30)