# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from typing import TYPE_CHECKING
import math
from dataclasses import MISSING
import numpy as np

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
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, TiledCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
from isaaclab.managers.manager_base import ManagerTermBase

import torch
import os
import json
from pathlib import Path
import random
import math
import warnings

# Suppress specific IsaacLab deprecation warnings
warnings.filterwarnings("ignore", message=".*quat_rotate.*deprecated.*", category=UserWarning)
import sys

# Add the workspace root to Python path to ensure genhrl can be imported
workspace_root = Path(__file__).parent.parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

# Import base MDP functions from IsaacLab
from isaaclab.envs import mdp
# Import custom MDP functions from genhrl
import genhrl.generation.mdp.rewards as custom_rewards
import genhrl.generation.mdp.terminations as custom_terminations  
import genhrl.generation.mdp.observations as custom_observations
import genhrl.generation.mdp.events as custom_events
import genhrl.generation.mdp.curriculums as custom_curriculums

from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from genhrl.generation.mdp.events import preload_success_states
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.envs.common import ViewerCfg
from isaaclab.utils.assets import retrieve_file_path

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from .TaskRewardsCfg import TaskRewardsCfg
from .SuccessTerminationCfg import SuccessTerminationCfg
# from .CollapseTerminationCfg import CollapseTerminationCfg
from genhrl.generation.objects import generate_object_configs, load_object_config

# Import robot configurations directly from IsaacLab Assets
ROBOT_NAME = os.environ.get('ROBOT_NAME', 'G1')  # Default to G1 if not set

# Import standard robot configurations from isaaclab_assets
if ROBOT_NAME == 'G1':
    from isaaclab_assets import G1_MINIMAL_CFG, G1_CFG
    ROBOT_CFG = G1_CFG
elif ROBOT_NAME == 'H1':
    from isaaclab_assets import UNITREE_H1_CFG
    ROBOT_CFG = UNITREE_H1_CFG
elif ROBOT_NAME == 'ANYmal-D':
    from isaaclab_assets import ANYMAL_D_CFG
    ROBOT_CFG = ANYMAL_D_CFG
elif ROBOT_NAME == 'ANYmal-C':
    from isaaclab_assets import ANYMAL_C_CFG
    ROBOT_CFG = ANYMAL_C_CFG
elif ROBOT_NAME == 'A1':
    from isaaclab_assets import UNITREE_A1_CFG
    ROBOT_CFG = UNITREE_A1_CFG
elif ROBOT_NAME == 'Go1':
    from isaaclab_assets import UNITREE_GO1_CFG
    ROBOT_CFG = UNITREE_GO1_CFG
elif ROBOT_NAME == 'Go2':
    from isaaclab_assets import UNITREE_GO2_CFG
    ROBOT_CFG = UNITREE_GO2_CFG
else:
    # Default to G1 if robot not found
    from isaaclab_assets import G1_MINIMAL_CFG, G1_CFG
    ROBOT_CFG = G1_CFG
    print(f"⚠️ Robot '{ROBOT_NAME}' not supported, defaulting to G1")

print(f"🤖 Using robot configuration: {ROBOT_NAME}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
object_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "object_config.json"))
print(f"Using object_config_path: {object_config_path}")


##
# Reward weight constants: Everything is now normalised except Success and Termination. 
# This means we can compare these weights directly with those which are generated.
# This is incredibly helpful.
##
LIN_VEL_Z_L2_WEIGHT = 0 #-0.2
ANG_VEL_XY_L2_WEIGHT = 0 #-0.005
DOF_TORQUES_L2_WEIGHT = -0.2
DOF_ACC_L2_WEIGHT = -0.2
ACTION_RATE_L2_WEIGHT = -0.3 
FEET_AIR_TIME_WEIGHT = 0 # 0.125
UNDESIRED_CONTACTS_WEIGHT = -1.0
FLAT_ORIENTATION_L2_WEIGHT = -0.4 # -0.1
DOF_POS_LIMITS_WEIGHT = -0.1
TERMINATION_WEIGHT = -100 # NOT normalised and averaged across the episode length.
HEAD_BEHIND_WEIGHT = 0 #-0.2
HIPS_OVEREXTENDED_WEIGHT = 0 # -0.2 
FOOT_CYCLING_WEIGHT = 0.1 # 0.2
SUCCESS_WEIGHT = 1000 # NOT normalised and averaged across the episode length. 
COLLAPSE_WEIGHT = -100
MOVEMENT_DIRECTION_ALIGNMENT_WEIGHT = -0.1
FEET_SLIDE_WEIGHT = -0.1
FORWARD_WORKSPACE_WEIGHT = 0

FINAL_LIN_VEL_Z_L2_WEIGHT = 0.0
FINAL_ANG_VEL_XY_L2_WEIGHT = 0.0
FINAL_DOF_TORQUES_L2_WEIGHT = DOF_TORQUES_L2_WEIGHT*2
FINAL_DOF_ACC_L2_WEIGHT = DOF_ACC_L2_WEIGHT*2
FINAL_ACTION_RATE_L2_WEIGHT = ACTION_RATE_L2_WEIGHT*2
FINAL_FEET_AIR_TIME_WEIGHT = 0.0
FINAL_UNDESIRED_CONTACTS_WEIGHT = -1.0
FINAL_FLAT_ORIENTATION_L2_WEIGHT = 0.0
FINAL_DOF_POS_LIMITS_WEIGHT = -0.1
FINAL_TERMINATION_WEIGHT = -1000
FINAL_HEAD_BEHIND_WEIGHT = 0.0
FINAL_HIPS_OVEREXTENDED_WEIGHT = 0.0
FINAL_FOOT_CYCLING_WEIGHT = 0.0
FINAL_SUCCESS_WEIGHT = 1000
FINAL_MOVEMENT_DIRECTION_ALIGNMENT_WEIGHT = 0.0
FINAL_FEET_SLIDE_WEIGHT = 0.0
FINAL_FORWARD_WORKSPACE_WEIGHT = 0.0
FINAL_COLLAPSE_WEIGHT = 0.0
FINAL_MOVEMENT_DIRECTION_ALIGNMENT_WEIGHT = -0.3
FINAL_FEET_SLIDE_WEIGHT = 0.0  

# Additional joint-specific weights
ANKLE_JOINT_LIMITS_WEIGHT = -0.05
JOINT_DEVIATION_HIP_WEIGHT = -0.05 # -0.1
JOINT_DEVIATION_ARMS_WEIGHT = -0.05
JOINT_DEVIATION_FINGERS_WEIGHT = -0.5 # we almost never want these changing.
JOINT_DEVIATION_TORSO_WEIGHT = -0.05 # -0.1

FINAL_ANKLE_JOINT_LIMITS_WEIGHT = ANKLE_JOINT_LIMITS_WEIGHT * 3
FINAL_JOINT_DEVIATION_HIP_WEIGHT = JOINT_DEVIATION_HIP_WEIGHT * 3
FINAL_JOINT_DEVIATION_ARMS_WEIGHT = JOINT_DEVIATION_ARMS_WEIGHT * 3
FINAL_JOINT_DEVIATION_FINGERS_WEIGHT = JOINT_DEVIATION_FINGERS_WEIGHT * 3
FINAL_JOINT_DEVIATION_TORSO_WEIGHT = JOINT_DEVIATION_TORSO_WEIGHT * 3

TOTAL_STEPS = 50_000

##
# Custom Debug Termination Functions
##

def debug_bad_orientation(
    env: ManagerBasedRLEnv, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Debug version of bad_orientation termination with detailed logging."""
    from isaaclab.assets import RigidObject
    
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Debug logging
    gravity_z = -asset.data.projected_gravity_b[:, 2]
    angles = torch.acos(torch.clamp(gravity_z, -1.0, 1.0)).abs()  # Clamp to avoid NaN
    terminations = angles > limit_angle
    
    # Log debug info for early terminations (episode length < 5)
    if torch.any(terminations) and hasattr(env, 'episode_length_buf'):
        terminating_envs = torch.nonzero(terminations).flatten()
        episode_lengths = env.episode_length_buf[terminating_envs]
        
        # Only log if any terminating environment has episode length < 5
        early_terminations = episode_lengths < 5
        if torch.any(early_terminations):
            early_envs = terminating_envs[early_terminations]
            early_lengths = episode_lengths[early_terminations]
            
            print(f"\n=== EARLY BAD_ORIENTATION DEBUG ===")
            print(f"Episode lengths: {early_lengths.tolist()}")
            print(f"Limit angle: {limit_angle:.4f} rad ({limit_angle * 180/math.pi:.1f} deg)")
            print(f"Early environments with bad orientation: {early_envs.tolist()}")
            print(f"Gravity Z values: {gravity_z[early_envs]}")
            angles_deg = (angles[early_envs] * 180/math.pi).tolist()
            print(f"Computed angles: {angles[early_envs]} rad ({angles_deg} deg)")
            print(f"Root orientation (quat): {asset.data.root_quat_w[early_envs]}")
            print(f"Projected gravity full: {asset.data.projected_gravity_b[early_envs]}")
            print("=====================================\n")
    
    return terminations


def debug_root_height_below_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Debug version of root_height_below_minimum termination with detailed logging."""
    from isaaclab.assets import RigidObject
    
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Debug logging
    heights = asset.data.root_pos_w[:, 2]
    terminations = heights < minimum_height
    
    # Log debug info for early terminations (episode length < 5)
    if torch.any(terminations) and hasattr(env, 'episode_length_buf'):
        terminating_envs = torch.nonzero(terminations).flatten()
        episode_lengths = env.episode_length_buf[terminating_envs]
        
        # Only log if any terminating environment has episode length < 5
        early_terminations = episode_lengths < 5
        if torch.any(early_terminations):
            early_envs = terminating_envs[early_terminations]
            early_lengths = episode_lengths[early_terminations]
            
            print(f"\n=== EARLY HEIGHT_LIMIT DEBUG ===")
            print(f"Episode lengths: {early_lengths.tolist()}")
            print(f"Minimum height threshold: {minimum_height:.4f}")
            print(f"Early environments below height limit: {early_envs.tolist()}")
            print(f"Actual heights: {heights[early_envs]}")
            print(f"Root positions (full): {asset.data.root_pos_w[early_envs]}")
            print(f"Root velocities: {asset.data.root_vel_w[early_envs]}")
            print("=====================================\n")
    
    return terminations


def debug_time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Debug version of time_out with logging."""
    terminations = env.episode_length_buf >= env.max_episode_length
    
    # Log debug info for early terminations (episode length < 5)
    if torch.any(terminations) and hasattr(env, 'episode_length_buf'):
        terminating_envs = torch.nonzero(terminations).flatten()
        episode_lengths = env.episode_length_buf[terminating_envs]
        
        # Only log if any terminating environment has episode length < 5
        early_terminations = episode_lengths < 5
        if torch.any(early_terminations):
            early_envs = terminating_envs[early_terminations]
            early_lengths = episode_lengths[early_terminations]
            
            print(f"\n=== EARLY TIMEOUT DEBUG ===")
            print(f"Episode lengths: {early_lengths.tolist()}")
            print(f"Max episode length: {env.max_episode_length}")
            print(f"Early environments timing out: {early_envs.tolist()}")
            print("=====================================\n")
    
    return terminations


def debug_general_terminations(env: ManagerBasedRLEnv) -> None:
    """General debug function to track all early terminations."""
    if hasattr(env, 'episode_length_buf'):
        early_envs = env.episode_length_buf < 5
        if torch.any(early_envs):
            early_env_ids = torch.nonzero(early_envs).flatten()
            early_lengths = env.episode_length_buf[early_envs]
            
            print(f"\n=== GENERAL EARLY TERMINATION CHECK ===")
            print(f"Environments with episode length < 5: {early_env_ids.tolist()}")
            print(f"Their episode lengths: {early_lengths.tolist()}")
            print(f"Total environments: {len(env.episode_length_buf)}")
            print(f"Min episode length: {env.episode_length_buf.min().item()}")
            print(f"Max episode length: {env.episode_length_buf.max().item()}")
            print(f"Mean episode length: {env.episode_length_buf.float().mean().item():.2f}")
            print("=====================================\n")


def debug_episode_length_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Debug reward function that calls the general termination debug and returns zero reward."""
    debug_general_terminations(env)
    return torch.zeros(env.num_envs, device=env.device)

##
# Scene definition
##
    

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",  # Will be overridden in __post_init__
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        history_length=3, 
        track_air_time=True
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # THIS NEEDS TO BE CALLED WHEN RUNNING OUR SCRIPT FOR as:
    # This is set in the example.py script

    ## objects here:
    Object1, Object2, Object3, Object4, Object5 = generate_object_configs(
        **load_object_config(json_path=object_config_path)
    )

    

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # base_velocity = mdp.UniformVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 10.0),
    #     rel_standing_envs=0.02,
    #     rel_heading_envs=1.0,
    #     heading_command=True,
    #     heading_control_stiffness=0.5,
    #     debug_vis=True,
    #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
    #     ),
    # )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.1, n_max=0.1))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        object_features = ObsTerm(func=custom_observations.get_object_features)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()



@configclass
class EventCfg:
    """Configuration for events."""

    # interval

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.1, 0.1),
            "num_buckets": 1,
        },
    )

    # Add actuator gain randomization
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (1.0, 1.0),
            "damping_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "mass_distribution_params": (-5.0, 5.0),
    #         "operation": "add",
    #     },
    # )

    # reset
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "force_range": (0, 0),
    #         "torque_range": (0, 0),
    #     },
    # )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0, 0), "y": (0, 0), "yaw": (0, 0)},
            "velocity_range": {
                "x": (0, 0),
                "y": (0, 0),
                "z": (0, 0),
                "roll": (0, 0),
                "pitch": (0, 0),
                "yaw": (0, 0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # # interval - Enable periodic perturbations with low forces
    # push_robot = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="interval",
    #     interval_range_s=(3.0, 6.0),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #         "force_range": (-1, 1),
    #         "torque_range": (-0.5, 0.5),
    #     },
    # )

    # add skill loading from dependent skills
    add_dependent_skill_loading = EventTerm(
        func=custom_events.load_success_state_from_dependent_skill,
        mode="reset",
    )

    # def __post_init__(self):    
    #     ### OBJECT CODE ###
    #     for object in ["Object1", "Object2", "Object3", "Object4", "Object5"]:
    #         # Add new event for shape randomization
    #         setattr(self, f"randomize_shapes_{object}", EventTerm(
    #             func=mdp.randomize_object_shapes,
    #             mode="reset",
    #             params={
    #                 "asset_cfg": SceneEntityCfg(object),
    #                 "scale_range": (1, 1),  # Scale dimensions by ±10%
    #             },
    #         ))

    #         # Add new event for object randomization - adjusted scale range to ensure positive masses
    #         setattr(self, f"randomize_mass_{object}", EventTerm(
    #             func=mdp.randomize_rigid_body_mass,
    #             mode="reset",
    #             params={
    #                 "asset_cfg": SceneEntityCfg(object),
    #                 "mass_distribution_params": (1, 1),  # Scale mass by 0.9x to 1.1x
    #                 "operation": "scale",  # Scales the mass by a factor
    #             },
    #         ))

    #         # Add new event for randomizing object positions
    #         setattr(self, f"randomize_pos_{object}", EventTerm(
    #             func=mdp.reset_root_state_uniform,
    #             mode="reset",
    #             params={
    #                 "asset_cfg": SceneEntityCfg(object),
    #                 "pose_range": {
    #                     "x": (-0.3, 0.3),     # Add noise to x position
    #                     "y": (-0.3, 0.3),     # Add noise to y position
    #                     "yaw": (-0.3, 0.3),   # Add small rotation noise
    #                 },
    #                 "velocity_range": {
    #                     "x": (0.0, 0.0),      # No initial velocity
    #                     "y": (0.0, 0.0),
    #                     "z": (0.0, 0.0),
    #                     "roll": (0.0, 0.0),
    #                     "pitch": (0.0, 0.0),
    #                     "yaw": (0.0, 0.0),
    #                 },
    #             },
    #         ))

    #     # Add success state loading from dependent skills
    #     # self._add_dependent_skill_loading()
    


@configclass
class RewardsCfg(TaskRewardsCfg):
    """Reward terms for the MDP."""

    # -- task
    # track_lin_vel_xy_exp = RewTerm(
    #     func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )
    # track_ang_vel_z_exp = RewTerm(
    #     func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )
    
    termination_penalty = RewTerm(func=mdp.is_terminated_term, params = {"term_keys": ["height_limit", "bad_orientation"]},
                                                                            weight=TERMINATION_WEIGHT)


    success_reward = RewTerm(func=mdp.is_terminated_term, params = {"term_keys": ["success"]},
                                                                            weight=SUCCESS_WEIGHT)
    
    # Add a debug reward that will let us track episode lengths
    # episode_length_debug = RewTerm(func=debug_episode_length_reward, weight=0.0)
    
    # uncomment if want to use and change the inheritance to CollapseTerminationCfg of the TerminationsCfg class
    # collapse_reward = RewTerm(func=mdp.is_terminated_term, params = {"normalise": True,
    #                                                                       "normaliser_name": "collapse_reward",
    #                                                                       "term_keys": ["collapse"]},
    #                                                                         weight=COLLAPSE_WEIGHT)



    # -- penalties
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, params = {"normalise": True, "normaliser_name": "lin_vel_z_l2"}, weight=LIN_VEL_Z_L2_WEIGHT)
    # ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, params = {"normalise": True, "normaliser_name": "ang_vel_xy_l2"}, weight=ANG_VEL_XY_L2_WEIGHT)
    dof_torques_l2 = RewTerm(func=custom_rewards.joint_torques_l2, params = {"normalise": True, "normaliser_name": "joint_torques_l2", "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])}, weight=DOF_TORQUES_L2_WEIGHT)
    dof_acc_l2 = RewTerm(func=custom_rewards.joint_acc_l2, params = {"normalise": True, "normaliser_name": "joint_acc_l2", "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])}, weight=DOF_ACC_L2_WEIGHT)
    action_rate_l2 = RewTerm(func=custom_rewards.action_rate_l2, params = {"normalise": True, "normaliser_name": "action_rate_l2"}, weight=ACTION_RATE_L2_WEIGHT)
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time,
    #     weight=FEET_AIR_TIME_WEIGHT,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
    #         "command_name": "base_velocity",
    #         "threshold": 0.5,
    #     },
    # )
    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=UNDESIRED_CONTACTS_WEIGHT,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    # )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=custom_rewards.flat_orientation_l2, params = {"normalise": True, "normaliser_name": "flat_orientation_l2"}, weight=FLAT_ORIENTATION_L2_WEIGHT)
    dof_pos_limits = RewTerm(func=custom_rewards.joint_pos_limits, params = {"normalise": True, "normaliser_name": "dof_pos_limits"},  weight=DOF_POS_LIMITS_WEIGHT)

    # Add the new penalty rewards with strong negative weights
    # head_behind = RewTerm(
    #     func=mdp.head_behind_torso_penalty,
    #     weight=HEAD_BEHIND_WEIGHT,  # Adjust this weight as needed
    #     params={"normalise": True, "normaliser_name": "head_behind"}
    # )
    
    # hips_overextended = RewTerm(
    #     func=mdp.hips_overextended_penalty,
    #     weight=HIPS_OVEREXTENDED_WEIGHT,  # Adjust this weight as needed
    #     params={"normalise": True, "normaliser_name": "hips_overextended"}
    # )

    # Add the foot cycling reward
    foot_cycling = RewTerm(
        func=custom_rewards.foot_contact_cycling_upgraded,
        weight=FOOT_CYCLING_WEIGHT,
        params={
            "normalise": True,
            "normaliser_name": "foot_cycling",
        }
    )

    movement_direction_alignment = RewTerm(
        func=custom_rewards.movement_direction_alignment,
        weight=MOVEMENT_DIRECTION_ALIGNMENT_WEIGHT,
        params={"normalise": True, "normaliser_name": "movement_direction_alignment"}
    )

    feet_slide = RewTerm(
        func=custom_rewards.feet_slide,
        weight=FEET_SLIDE_WEIGHT,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "normalise": True,
            "normaliser_name": "feet_slide"
        },
    )

    # forward_workspace = RewTerm(
    #     func=mdp.forward_workspace_reward,
    #     weight=FORWARD_WORKSPACE_WEIGHT,
    #     params={"normalise": True, "normaliser_name": "forward_workspace"}
    # )


@configclass
class TerminationsCfg(SuccessTerminationCfg): # change to CollapseTerminationCfg is you want to use it 
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    # )

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": np.pi/4},
    )
    
    height_limit = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.45, "asset_cfg": SceneEntityCfg("robot")})


def reward_weights_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    initial_weights: dict,
    final_weights: dict,
    total_steps: int,
) -> float:
    """Curriculum that linearly anneals reward weights from initial to final values.
    
    Args:
        env: The environment instance
        env_ids: The environment IDs (not used but required for curriculum interface)
        initial_weights: Dictionary of initial weights for each reward term
        final_weights: Dictionary of final weights for each reward term
        total_steps: Total number of training steps
        
    Returns:
        Mean of current weights (for tracking purposes)
    """
    # Calculate progress (clamped between 0 and 1)
    progress = min(env.common_step_counter / total_steps, 1.0)
    
    # Add debug logging
    if env.common_step_counter % 1000 == 0:  # Log every 1000 steps
        print(f"Step counter: {env.common_step_counter}, Total steps: {total_steps}, Progress: {progress}")
    
    current_weights = []
    # For each reward term, interpolate between initial and final weights
    for term_name in initial_weights:
        if term_name in final_weights:
            initial_weight = initial_weights[term_name]
            final_weight = final_weights[term_name]
            # Linear interpolation
            current_weight = initial_weight + (final_weight - initial_weight) * progress
            current_weights.append(current_weight)
            
            # Get the term configuration
            term_cfg = env.reward_manager.get_term_cfg(term_name)
            # Update the weight
            term_cfg.weight = current_weight
            # Set the updated configuration
            env.reward_manager.set_term_cfg(term_name, term_cfg)
            
            # Add debug logging for weight changes
            if env.common_step_counter % 1000 == 0:  # Log every 1000 steps
                print(f"Term: {term_name}, Initial: {initial_weight}, Current: {current_weight}, Final: {final_weight}")
    
    return sum(current_weights) / len(current_weights) if current_weights else 0.0


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    
    # Enable reward weight curriculum focused on termination and success consistency
    # reward_weights = CurrTerm(
    #     func=reward_weights_curriculum,
    #     params={
    #                      "initial_weights": {
    #              "termination_penalty": TERMINATION_WEIGHT,  
    #              "success_reward": SUCCESS_WEIGHT,           
    #              "dof_torques_l2": DOF_TORQUES_L2_WEIGHT,
    #              "dof_acc_l2": DOF_ACC_L2_WEIGHT,
    #              "action_rate_l2": ACTION_RATE_L2_WEIGHT,
    #              "dof_pos_limits": DOF_POS_LIMITS_WEIGHT,
    #              "joint_deviation_hip": JOINT_DEVIATION_HIP_WEIGHT,
    #              "joint_deviation_arms": JOINT_DEVIATION_ARMS_WEIGHT,
    #              "joint_deviation_fingers": JOINT_DEVIATION_FINGERS_WEIGHT,
    #              "joint_deviation_torso": JOINT_DEVIATION_TORSO_WEIGHT,
    #          },
    #          "final_weights": {
    #              "termination_penalty": FINAL_TERMINATION_WEIGHT,   # End with much harsher penalty
    #              "success_reward": SUCCESS_WEIGHT,         # Increase success reward
    #              "dof_torques_l2": FINAL_DOF_TORQUES_L2_WEIGHT,
    #              "dof_acc_l2": FINAL_DOF_ACC_L2_WEIGHT,
    #              "action_rate_l2": FINAL_ACTION_RATE_L2_WEIGHT,
    #              "dof_pos_limits": FINAL_DOF_POS_LIMITS_WEIGHT,
    #              "joint_deviation_hip": FINAL_JOINT_DEVIATION_HIP_WEIGHT,
    #              "joint_deviation_arms": FINAL_JOINT_DEVIATION_ARMS_WEIGHT,
    #              "joint_deviation_fingers": FINAL_JOINT_DEVIATION_FINGERS_WEIGHT,
    #              "joint_deviation_torso": FINAL_JOINT_DEVIATION_TORSO_WEIGHT,
    #          },
    #         "total_steps": TOTAL_STEPS,
    #     }
    # )

##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=10)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # ------------------------------------------------------------------
        # Ensure each new environment starts with a fresh reward normalizer.
        # This prevents the running mean / variance from the previous skill
        # leaking into the next one (which caused all-zero `info/rewards`).
        # ------------------------------------------------------------------
        try:
            import genhrl.generation.reward_normalizer as _rn
            _rn._NORMALIZER = None  # type: ignore[attr-defined]
        except Exception as _reset_exc:
            print(f"[WARNING] Failed to reset RewardNormalizer: {_reset_exc}")

        # general settings
        self.decimation = 2
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = False
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


# Robot configuration is already imported above based on ROBOT_NAME


@configclass
class RobotSpecificRewards(RewardsCfg):
    """Robot-specific reward terms for the MDP."""

    def __post_init__(self):
        super().__post_init__()
        
        # Add standard joint deviation rewards for humanoid robots
        self.joint_deviation_hip = RewTerm(
            func=custom_rewards.joint_deviation_l1,
            weight=JOINT_DEVIATION_HIP_WEIGHT,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*"]),
                "normalise": True,
                "normaliser_name": "joint_deviation_hip",
            },
        )
        
        self.joint_deviation_arms = RewTerm(
            func=custom_rewards.joint_deviation_l1,
            weight=JOINT_DEVIATION_ARMS_WEIGHT,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*"]),
                "normalise": True,
                "normaliser_name": "joint_deviation_arms",
            },
        )
        
        self.joint_deviation_torso = RewTerm(
            func=custom_rewards.joint_deviation_l1,
            weight=JOINT_DEVIATION_TORSO_WEIGHT,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["torso_joint"]),
                "normalise": True,
                "normaliser_name": "joint_deviation_torso",
            },
        )



@configclass
class RobotRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: RobotSpecificRewards = RobotSpecificRewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # Add action and observation noise models for better generalization
        self.action_noise_model = NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
        )
        
        self.observation_noise_model = NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
        )
        
        # Scene - Use the imported robot configuration directly
        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # Update height scanner to use the correct robot base link for G1
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Add hook to preload success states when environment is created
        self.preload_success_states = True
        # Get the skill name from the directory path
        skill_name = Path(__file__).parent.name
        custom_events.preload_success_states(device=device, skill_name=skill_name)

        # Randomization
        # self.events.push_robot = None
        # self.events.add_base_mass = None
        # self.events.reset_robot_joints.params["position_range"] = (0.7, 1.3)
        # self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        # self.events.reset_base.params = {
        #     "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        #     "velocity_range": {
        #         "x": (-0.2, 0.2),
        #         "y": (-0.2, 0.2),
        #         "z": (-0.2, 0.2),
        #         "roll": (-0.2, 0.2),
        #         "pitch": (-0.2, 0.2),
        #         "yaw": (-0.2, 0.2),
        #     },
        # }

        # terminations
        # self.terminations.base_contact.params["sensor_cfg"].body_names = ["head_link"]


@configclass
class RobotRoughEnvCfg_PLAY(RobotRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 10
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None


@configclass
class RobotFlatEnvCfg(RobotRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Add action and observation noise models for better generalization
        self.action_noise_model = NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
        )
        
        self.observation_noise_model = NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
        )

        # Configure camera for video recording
        self.tiled_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(-20.0, 0.0, 15.0),  # Position camera further back and higher
                rot=(0.9945, 0.0, 0.1045, 0.0),  # Look down at the scene
                convention="world"
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 20.0)
            ),
            width=1280,
            height=720,
        )
        
        # Configure viewport camera for display
        self.viewer = ViewerCfg(
            origin_type="env",  # Track environment 0
            env_index=0,  # Specifically track env 0
            eye=(0.0, -20.0, 15.0),  # Position camera further back and higher
            lookat=(0.0, 0.0, 0.0),  # Look at center of env 0
            cam_prim_path="/OmniverseKit_Persp",  # Default camera in viewport
            resolution=(1280, 720)  # Standard resolution
        )

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        # self.scene.height_scanner = None
        # self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        self.preload_success_states = True

        # self.terminations.base_contact.params["sensor_cfg"].body_names = ["head_link"]



class RobotFlatEnvCfg_PLAY(RobotFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 10
        
        
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# Add this function at the module level
def load_success_state_from_dependent_skill(env: ManagerBasedRLEnv, env_ids: torch.Tensor, dependent_skill: str, probability: float = 1.0) -> None:
    """Load success states from a dependent skill for specific environments that are resetting.
    
    Args:
        env: The environment instance
        env_ids: The environment IDs that are being reset
        dependent_skill: The name of the dependent skill to load states from
        probability: Probability of loading a success state (vs. random initialization)
    """
    # Convert env_ids to a list if it's a tensor
    if isinstance(env_ids, torch.Tensor):
        env_ids = env_ids.cpu().tolist()
    
    # Determine which environments will load success states (based on probability)
    use_success_state = []
    for env_id in env_ids:
        if random.random() < probability:
            use_success_state.append(env_id)
    
    if not use_success_state:
        return  # No environments selected for success state loading
    
    # Get the path to the dependent skill's success states
    current_dir = Path(__file__).parent
    task_dir = current_dir.parent
    dependent_skill_dir = task_dir / dependent_skill / "success_states"
    
    if not dependent_skill_dir.exists() or not any(dependent_skill_dir.iterdir()):
        print(f"Warning: No success states found for dependent skill {dependent_skill} at {dependent_skill_dir}")
        return
    
    # Get all state files for the dependent skill
    state_files = list(dependent_skill_dir.glob(f"success_states_{dependent_skill}_*.pt"))
    if not state_files:
        print(f"Warning: No success state files found for {dependent_skill}")
        return
    
    # Randomly select a file and load states
    selected_file = random.choice(state_files)
    try:
        states = torch.load(selected_file)
        
        # If no states in the file, return
        if not states:
            print(f"Warning: No states in file {selected_file}")
            return
        
        # For each environment that needs a success state
        for env_id in use_success_state:
            # Randomly select a state
            if states:
                state = random.choice(states)
                
                # Use reset_to to apply the state to the environment
                # The state is already in the correct format from base_success.py
                env.reset_to(state=state, env_ids=[env_id], is_relative=True)
                
    except Exception as e:
        print(f"Error loading success states: {e}")


