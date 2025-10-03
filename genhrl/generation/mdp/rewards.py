# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable reward functions.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to include
the reward introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Optional

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat, quat_apply
from ..reward_normalizer import RewardNormalizer, RewardStats
import wandb

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

""" init the reward_normalizer"""

RewNormalizer = RewardNormalizer(device=torch.device("cuda"))


"""
General.
"""




def is_alive(env: ManagerBasedRLEnv, normalise: bool = False, normaliser_name: str = "is_alive") -> torch.Tensor:
    """Reward for being alive."""

    reward = (~env.termination_manager.terminated).float()

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward

    return reward


def is_terminated(env: ManagerBasedRLEnv, normalise: bool = False, normaliser_name: str = "is_terminated") -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""

    reward = env.termination_manager.terminated.float()

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward

    return reward


def overall_raw_reward_temp_terminated(env: ManagerBasedRLEnv, normalise: bool = False, normaliser_name: str = "overall_raw_reward") -> torch.Tensor:
    '''Overall raw reward for the obstacle course.
    '''

    # This should be a combination of +1 for being past the low wall.
    # +1 for pushing the large sphere towards the high wall.
    # +1 for kicking the small sphere towards the block.
    # +1 for jumping onto the block.
    reward = torch.zeros(env.num_envs, device=env.device)
    
    try:
        robot = env.scene["robot"]
        low_wall = env.scene['Object3']
        large_sphere = env.scene['Object1']
        high_wall = env.scene['Object4']
        small_sphere = env.scene['Object2']
        block = env.scene['Object5']
        
        # +1 for being past the low wall
        pelvis_idx = robot.body_names.index('pelvis')
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
        pelvis_pos_x = pelvis_pos[:, 0]
        low_wall_pos_x = low_wall.data.root_pos_w[:, 0]
        past_low_wall = pelvis_pos_x > low_wall_pos_x
        reward = torch.where(past_low_wall, reward + 1.0, reward)
        
        # +1 for pushing the large sphere towards the high wall (only if past low wall)
        large_sphere_pos_x = large_sphere.data.root_pos_w[:, 0]
        high_wall_pos_x = high_wall.data.root_pos_w[:, 0]
        high_wall_pos_z = high_wall.data.root_pos_w[:, 2]
        large_sphere_near_wall = torch.abs(high_wall_pos_x - large_sphere_pos_x) < 2.0
        wall_pushed = high_wall_pos_z < 0.4  # Wall has fallen
        sphere_pushed_wall = large_sphere_near_wall
        # Only award if past low wall
        reward = torch.where(past_low_wall & wall_pushed, reward + 1.0, reward)
        
        # +1 for kicking the small sphere away from the block (only if wall pushed)
        small_sphere_pos_xy = small_sphere.data.root_pos_w[:, :2]
        block_pos_xy = block.data.root_pos_w[:, :2]
        small_sphere_away_from_block = torch.norm(small_sphere_pos_xy - block_pos_xy, dim=1) > 4.0
        # Only award if past low wall and wall pushed
        reward = torch.where(past_low_wall & wall_pushed & small_sphere_away_from_block, reward + 1.0, reward)
        
        # +1 for jumping onto the block (only if sphere kicked away)
        pelvis_pos_z = pelvis_pos[:, 2]
        block_pos_x = block.data.root_pos_w[:, 0]
        block_pos_z_top = block.data.root_pos_w[:, 2] + 0.5  # top of block, 0.5m height
        on_block = (pelvis_pos_x > block_pos_x - 0.5) & (pelvis_pos_x < block_pos_x + 0.5) & (pelvis_pos_z > block_pos_z_top + 0.3)
        # Only award if all previous milestones completed
        reward = torch.where(past_low_wall & wall_pushed & small_sphere_away_from_block & on_block, reward + 1.0, reward)

        # reward now masked by the termination
        is_terminated = env.termination_manager.terminated.float()
        sensible_steps = env._sim_step_counter > 50

        is_terminated_and_sensible = is_terminated * sensible_steps

        reward = reward * is_terminated_and_sensible
        # count number of non terminated episodes
        num_terminated = torch.sum(is_terminated_and_sensible)
        # sum of rewards of non terminated episodes
        sum_rewards = torch.sum(reward)

        # Count environments with different reward values (0, 1, 2, 3, 4)
        # Only count terminated environments
        terminated_mask = is_terminated_and_sensible.bool()
        if torch.any(terminated_mask):
            # Initialize counters for each reward value
            reward_0_count = torch.sum((reward >= 0.0) & terminated_mask)
            reward_1_count = torch.sum((reward >= 1.0) & terminated_mask)
            reward_2_count = torch.sum((reward >= 2.0) & terminated_mask)
            reward_3_count = torch.sum((reward >= 3.0) & terminated_mask)
            reward_4_count = torch.sum((reward >= 4.0) & terminated_mask)
            
            # Log the counts to wandb
            wandb.log({
                "rewards/terminated_with_reward_0": reward_0_count.item()/num_terminated*100,
                "rewards/terminated_with_reward_1": reward_1_count.item()/num_terminated*100,
                "rewards/terminated_with_reward_2": reward_2_count.item()/num_terminated*100,
                "rewards/terminated_with_reward_3": reward_3_count.item()/num_terminated*100,
                "rewards/terminated_with_reward_4": reward_4_count.item()/num_terminated*100,
            })

        reward = sum_rewards / 2000

        if num_terminated > 0:
            wandb.log({f"infos/obstacles_completed": sum_rewards.item() / num_terminated.item(),
                    })

        if normalise:
            reward = RewNormalizer.normalize(normaliser_name, reward)
            RewNormalizer.update_stats(normaliser_name, reward)
            return reward
        else:
            reward = reward
    
    except KeyError:
        pass  # Keep reward at zeros if objects not found
    
    return reward


def record_temp(env: ManagerBasedRLEnv, normalise: bool = False, normaliser_name: str = "record_temp") -> torch.Tensor:
    """Record the amount of obstacles completed on termination."""

    reward = overall_raw_reward_temp_terminated(env, normalise, normaliser_name)



    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward

    return reward


class is_terminated_term(ManagerTermBase):
    """Penalize termination for specific terms that don't correspond to episodic timeouts.

    The parameters are as follows:

    * attr:`term_keys`: The termination terms to penalize. This can be a string, a list of strings
      or regular expressions. Default is ".*" which penalizes all terminations.

    The reward is computed as the sum of the termination terms that are not episodic timeouts.
    This means that the reward is 0 if the episode is terminated due to an episodic timeout. Otherwise,
    if two termination terms are active, the reward is 2.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        term_keys = cfg.params.get("term_keys", ".*")
        self._term_names = env.termination_manager.find_terms(term_keys)

    def __call__(self, env: ManagerBasedRLEnv, term_keys: str | list[str] = ".*", normalise: bool = False, normaliser_name: str = "termination_penalty") -> torch.Tensor:
        # Return the unweighted reward for the termination terms
        reset_buf = torch.zeros(env.num_envs, device=env.device)
        for term in self._term_names:
            # Sums over terminations term values to account for multiple terminations in the same step
            reset_buf += env.termination_manager.get_term(term)

        reward = (reset_buf * (~env.termination_manager.time_outs)).float()

        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()

        if normalise:
            if normaliser_name not in RewNormalizer.stats:
                RewNormalizer.stats[normaliser_name] = RewardStats()
            scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
            RewNormalizer.update_stats(normaliser_name, reward)
            return scaled_reward

        return reward


"""
Root penalties.
"""


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), normalise: bool = False, normaliser_name: str = "lin_vel_z_l2") -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward

    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), normalise: bool = False, normaliser_name: str = "ang_vel_xy_l2") -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    
    return reward

def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), normalise: bool = False, normaliser_name: str = "flat_orientation_l2") -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    
    return reward

def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    normalise: bool = False,
    normaliser_name: str = "base_height_l2"
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    
    return reward

def body_lin_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), normalise: bool = False, normaliser_name: str = "body_lin_acc_l2") -> torch.Tensor:
    """Penalize the linear acceleration of bodies using L2-kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1)

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    
    return reward

"""
Joint penalties.
"""


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), normalise: bool = False, normaliser_name: str = "joint_torques_l2") -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    
    return reward

def joint_vel_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), normalise: bool = False, normaliser_name: str = "joint_vel_l1") -> torch.Tensor:
    """Penalize joint velocities on the articulation using an L1-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    
    return reward

def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), normalise: bool = False, normaliser_name: str = "joint_vel_l2") -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    
    return reward

def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), normalise: bool = False, normaliser_name: str = "joint_acc_l2") -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    
    return reward

def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), normalise: bool = False, normaliser_name: str = "joint_deviation_l1") -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(angle), dim=1)

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    
    return reward

def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), normalise: bool = False, normaliser_name: str = "joint_pos_limits") -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    reward = torch.sum(out_of_limits, dim=1)

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    
    return reward

def joint_vel_limits(
    env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), normalise: bool = False, normaliser_name: str = "joint_vel_limits"
) -> torch.Tensor:
    """Penalize joint velocities if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint velocity and the soft limits.

    Args:
        soft_ratio: The ratio of the soft limits to be used.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = (
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
        - asset.data.soft_joint_vel_limits[:, asset_cfg.joint_ids] * soft_ratio
    )
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
    reward = torch.sum(out_of_limits, dim=1)

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    
    return reward

"""
Action penalties.
"""


def applied_torque_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), normalise: bool = False, normaliser_name: str = "applied_torque_limits") -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.

    .. caution::
        Currently, this only works for explicit actuators since we manually compute the applied torques.
        For implicit actuators, we currently cannot retrieve the applied torques from the physics engine.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    # TODO: We need to fix this to support implicit joints.
    out_of_limits = torch.abs(
        asset.data.applied_torque[:, asset_cfg.joint_ids] - asset.data.computed_torque[:, asset_cfg.joint_ids]
    )
    reward = torch.sum(out_of_limits, dim=1)

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    
    return reward

def action_rate_l2(env: ManagerBasedRLEnv, normalise: bool = False, normaliser_name: str = "action_rate_l2") -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    reward = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    
    return reward   

def action_l2(env: ManagerBasedRLEnv, normalise: bool = False, normaliser_name: str = "action_l2") -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    reward = torch.sum(torch.square(env.action_manager.action), dim=1)

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    
    return reward

"""
Contact sensor.
"""


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg, normalise: bool = False, normaliser_name: str = "undesired_contacts") -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1)

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    
    return reward

def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg, normalise: bool = False, normaliser_name: str = "contact_forces") -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    # compute the penalty
    reward = torch.sum(violation.clip(min=0.0), dim=1)

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    
    return reward

"""
Velocity-tracking rewards.
"""


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), normalise: bool = False, normaliser_name: str = "track_lin_vel_xy_exp"
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / (std**2 + 1e-8))

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    
    return reward

def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), normalise: bool = False, normaliser_name: str = "track_ang_vel_z_exp"
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / (std**2 + 1e-8))

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    
    return reward

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float, normalise: bool = False, normaliser_name: str = "feet_air_time"
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    
    return reward   

def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg, normalise: bool = False, normaliser_name: str = "feet_air_time_positive_biped") -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    
    return reward   

def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), normalise: bool = False, normaliser_name: str = "feet_slide") -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    
    return reward

def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), normalise: bool = False, normaliser_name: str = "track_lin_vel_xy_yaw_frame_exp"
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / (std**2 + 1e-8))

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    
    return reward

def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), normalise: bool = False, normaliser_name: str = "track_ang_vel_z_world_exp"
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / (std**2 + 1e-8))

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    
    return reward   

def lin_vel_xy_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), normalise: bool = False, normaliser_name: str = "lin_vel_xy_l2") -> torch.Tensor:
    """Return the L2 norm of the linear velocity in the xy plane.
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset.
        
    Returns:
        torch.Tensor: The magnitude of linear velocity in the xy plane for each environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # compute the L2 norm of xy velocities
    reward = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=1)

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    
    return reward

# def lin_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), normalise: bool = False, normaliser_name: str = "lin_vel_xy_l2_penalty") -> torch.Tensor:
#     """Penalize xy-axis base linear velocity using L2 squared kernel."""
#     # extract the used quantities (to enable type-hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
#     reward = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)

#     if normalise:
#         if normaliser_name not in RewNormalizer.stats:
#             RewNormalizer.stats[normaliser_name] = RewardStats()
#         scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
#         RewNormalizer.update_stats(normaliser_name, reward)
#         return scaled_reward

#     return reward

# def forward_workspace_reward(env: ManagerBasedRLEnv, 
#                            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
#                            body_names: list = ["left_palm_link", "right_palm_link"], 
#                            reference_body: str = "torso_link",
#                            normalise: bool = False, 
#                            normaliser_name: str = "forward_workspace") -> torch.Tensor:
#     """Reward for keeping end effectors in front of the reference body (usually torso)."""
#     # Get the robot asset
#     asset: Articulation = env.scene[asset_cfg.name]
    
#     # Get reference body index
#     ref_idx = asset.body_names.index(reference_body)
#     ref_pos = asset.data.body_pos_w[:, ref_idx]  # [num_envs, 3]
#     ref_quat = asset.data.body_quat_w[:, ref_idx]  # [num_envs, 4]
    
#     # Initialize reward
#     total_forward_reward = torch.zeros(env.num_envs, device=env.device)
    
#     # Get positions of tracked bodies
#     tracked_indices = [asset.body_names.index(name) for name in body_names]
#     tracked_pos = asset.data.body_pos_w[:, tracked_indices]  # [num_envs, num_bodies, 3]
    
#     # Convert to reference body's frame
#     relative_pos = tracked_pos - ref_pos.unsqueeze(1)  # Position relative to reference
#     local_pos = quat_apply_inverse(ref_quat.unsqueeze(1), relative_pos)
    
#     # Reward based on forward position (x in local frame)
#     forward_reward = torch.nn.functional.softplus(local_pos[..., 0])
#     total_forward_reward = torch.sum(forward_reward, dim=1)
    
#     # Normalize reward
#     if normalise:
#         if normaliser_name not in RewNormalizer.stats:
#             RewNormalizer.stats[normaliser_name] = RewardStats()
#         total_forward_reward = RewNormalizer.normalize(normaliser_name, total_forward_reward)
#         RewNormalizer.update_stats(normaliser_name, total_forward_reward)
    
#     return total_forward_reward

def forward_facing_reward(env: ManagerBasedRLEnv,
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                         reference_body: str = "torso_link",
                         target_body: str = "head_link",
                         normalise: bool = False, 
                         normaliser_name: str = "forward_facing") -> torch.Tensor:
    """Reward for keeping the head/face pointing in the same direction as the torso."""
    # Get the robot asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get body indices
    ref_idx = asset.body_names.index(reference_body)
    target_idx = asset.body_names.index(target_body)
    
    # Get orientations
    ref_quat = asset.data.body_quat_w[:, ref_idx]  # [num_envs, 4]
    target_quat = asset.data.body_quat_w[:, target_idx]  # [num_envs, 4]
    
    # Create forward vector with correct batch dimension
    forward_vec = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3)
    
    # Convert to forward vectors (local +x axis)
    ref_forward = quat_apply(ref_quat, forward_vec)
    target_forward = quat_apply(target_quat, forward_vec)
    
    # Compute alignment (dot product of forward vectors)
    alignment = torch.sum(ref_forward * target_forward, dim=1)
    
    # Convert to reward (shift to 0-1 range and square to emphasize alignment)
    reward = ((alignment + 1) / 2) ** 2
    
    # Normalize reward
    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
    
    return reward

def head_behind_torso_penalty(env: ManagerBasedRLEnv,
                            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                            reference_body: str = "torso_link",
                            head_body: str = "head_link",
                            normalise: bool = False,
                            normaliser_name: str = "head_behind_torso") -> torch.Tensor:
    """Penalize when the head is behind the torso in the robot's local frame.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        reference_body: Name of the reference body (torso)
        head_body: Name of the head body
        normalise: Whether to normalize the reward
        normaliser_name: Name of the normalizer stats
    """
    # Get the robot asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get body indices
    ref_idx = asset.body_names.index(reference_body)
    head_idx = asset.body_names.index(head_body)
    
    # Get positions
    ref_pos = asset.data.body_pos_w[:, ref_idx]  # [num_envs, 3]
    ref_quat = asset.data.body_quat_w[:, ref_idx]  # [num_envs, 4]
    head_pos = asset.data.body_pos_w[:, head_idx]  # [num_envs, 3]
    
    # Get head position relative to torso in torso's local frame
    relative_pos = head_pos - ref_pos  # Position relative to torso
    local_pos = quat_apply_inverse(ref_quat, relative_pos)  # Transform to torso's frame
    
    # Penalize when head is behind torso (negative x in local frame)
    penalty = torch.nn.functional.relu(-local_pos[..., 0])  # Only penalize when x < 0
    
    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        penalty = RewNormalizer.normalize(normaliser_name, penalty)
        RewNormalizer.update_stats(normaliser_name, penalty)
    
    return penalty


def forward_workspace_reward(env: ManagerBasedRLEnv, 
                           asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
                           body_names: list = ["left_palm_link", "right_palm_link"], 
                           reference_body: str = "torso_link",
                           normalise: bool = False, 
                           normaliser_name: str = "forward_workspace") -> torch.Tensor:
    """Reward for keeping end effectors in front of the reference body (usually torso)."""
    # Get the robot asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get reference body index
    ref_idx = asset.body_names.index(reference_body)
    ref_pos = asset.data.body_pos_w[:, ref_idx]  # [num_envs, 3]
    ref_quat = asset.data.body_quat_w[:, ref_idx]  # [num_envs, 4]
    
    # Initialize reward
    total_forward_reward = torch.zeros(env.num_envs, device=env.device)
    
    # Get positions of tracked bodies
    tracked_indices = [asset.body_names.index(name) for name in body_names]
    tracked_pos = asset.data.body_pos_w[:, tracked_indices]  # [num_envs, num_bodies, 3]
    
    # Convert to reference body's frame
    relative_pos = tracked_pos - ref_pos.unsqueeze(1)  # Position relative to reference
    local_pos = quat_apply_inverse(ref_quat.unsqueeze(1), relative_pos)
    
    # Reward based on forward position (x in local frame)
    forward_reward = torch.nn.functional.softplus(local_pos[..., 0])
    total_forward_reward = torch.sum(forward_reward, dim=1)
    
    # Normalize reward
    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        total_forward_reward = RewNormalizer.normalize(normaliser_name, total_forward_reward)
        RewNormalizer.update_stats(normaliser_name, total_forward_reward)
    
    return total_forward_reward


def hips_overextended_penalty(env: ManagerBasedRLEnv,
                            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                            hip_body: str = "pelvis",
                            left_foot: str = "left_ankle_roll_link",
                            right_foot: str = "right_ankle_roll_link",
                            normalise: bool = False,
                            normaliser_name: str = "hips_overextended") -> torch.Tensor:
    """Penalize when the hips are in front of both feet in the robot's local frame.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        hip_body: Name of the hip/pelvis body
        left_foot: Name of the left foot body
        right_foot: Name of the right foot body
        normalise: Whether to normalize the reward
        normaliser_name: Name of the normalizer stats
    """
    # Get the robot asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get body indices
    hip_idx = asset.body_names.index(hip_body)
    left_foot_idx = asset.body_names.index(left_foot)
    right_foot_idx = asset.body_names.index(right_foot)
    
    # Get positions
    hip_pos = asset.data.body_pos_w[:, hip_idx]  # [num_envs, 3]
    left_foot_pos = asset.data.body_pos_w[:, left_foot_idx]  # [num_envs, 3]
    right_foot_pos = asset.data.body_pos_w[:, right_foot_idx]  # [num_envs, 3]
    
    # Get forward direction in world frame (from root orientation)
    root_quat = asset.data.root_quat_w
    forward_vec = quat_apply(root_quat, torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3))
    
    # Project positions onto forward axis
    hip_proj = torch.sum(hip_pos * forward_vec, dim=1)
    left_foot_proj = torch.sum(left_foot_pos * forward_vec, dim=1)
    right_foot_proj = torch.sum(right_foot_pos * forward_vec, dim=1)
    
    # Penalize when hips are in front of both feet
    max_foot_proj = torch.maximum(left_foot_proj, right_foot_proj)
    penalty = torch.nn.functional.relu(hip_proj - max_foot_proj)
    
    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        penalty = RewNormalizer.normalize(normaliser_name, penalty)
        RewNormalizer.update_stats(normaliser_name, penalty)
    
    return penalty

def foot_contact_cycling(env: ManagerBasedRLEnv,
                        sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                        left_foot: str = "left_ankle_roll_link",
                        right_foot: str = "right_ankle_roll_link",
                        contact_threshold: float = 1.0,
                        double_contact_penalty: float = 0.5,
                        no_contact_penalty: float = 1.0,
                        min_contact_time: float = 0.3,
                        balance_weight: float = 1.5,
                        single_leg_time_penalty: float = 2.0,
                        velocity_threshold: float = 0.1,
                        normalise: bool = False,
                        normaliser_name: str = "foot_contact_cycling") -> torch.Tensor:
    """Reward alternating foot contacts while penalizing double stance or no contact.
    Also rewards balanced time spent on each foot for a more symmetric gait.
    Only applies rewards when the robot is moving above the velocity threshold."""
    # Get the contact sensor and robot asset
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Check if robot is moving (velocity above threshold)
    lin_vel_xy = asset.data.root_lin_vel_w[:, :2]
    is_moving = torch.norm(lin_vel_xy, dim=1) > velocity_threshold
    
    # Get indices for both feet using body_names (do once and cache)
    if not hasattr(env, '_foot_indices'):
        env._foot_indices = (contact_sensor.body_names.index(left_foot), 
                           contact_sensor.body_names.index(right_foot))
    left_idx, right_idx = env._foot_indices
    
    # Compute contact states in one operation
    forces = contact_sensor.data.net_forces_w_history[:, -1, [left_idx, right_idx]]  # Only use latest history
    contacts = torch.norm(forces, dim=-1) > contact_threshold
    left_contact, right_contact = contacts[:, 0], contacts[:, 1]
    
    # Get contact times
    contact_times = contact_sensor.data.current_contact_time[:, [left_idx, right_idx]]
    left_time, right_time = contact_times[:, 0], contact_times[:, 1]
    
    # Calculate stable contacts (single tensor operation)
    stable_contacts = contacts & (contact_times > min_contact_time)
    stable_left, stable_right = stable_contacts[:, 0], stable_contacts[:, 1]
    
    # Initialize state tensors if needed (do once)
    if not hasattr(env, '_prev_foot_contact'):
        env._prev_foot_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._was_left_foot = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._total_left_time = torch.zeros(env.num_envs, device=env.device)
        env._total_right_time = torch.zeros(env.num_envs, device=env.device)
        env._consecutive_same_foot = torch.zeros(env.num_envs, device=env.device)  # Track consecutive same foot contacts
    
    # Update total contact times (only when moving)
    env._total_left_time += (left_contact & is_moving).float() * env.step_dt
    env._total_right_time += (right_contact & is_moving).float() * env.step_dt
    
    # Compute transitions and update states (minimize tensor operations)
    # Only count transitions when the robot is moving
    transitions = ((env._was_left_foot & stable_right & ~left_contact) | 
                  (~env._was_left_foot & stable_left & ~right_contact)) & is_moving
    
    # Track consecutive same foot contacts (only when moving)
    env._consecutive_same_foot = torch.where(
        transitions | ~is_moving,  # Reset counter on transition or when not moving
        torch.zeros_like(env._consecutive_same_foot),
        env._consecutive_same_foot + env.step_dt  # Increment counter otherwise
    )
    
    # Update states (in-place operations)
    env._prev_foot_contact = left_contact | right_contact
    env._was_left_foot = torch.where(stable_left, torch.ones_like(env._was_left_foot),
                                   torch.where(stable_right, torch.zeros_like(env._was_left_foot),
                                             env._was_left_foot))
    
    # Calculate balance reward (higher when times are similar)
    total_time = env._total_left_time + env._total_right_time
    # Avoid division by zero
    balance_mask = total_time > 0
    balance_reward = torch.zeros_like(total_time)
    if balance_mask.any():
        # Calculate ratio of time spent on each foot (closer to 0.5 is better)
        left_ratio = torch.zeros_like(total_time)
        left_ratio[balance_mask] = env._total_left_time[balance_mask] / total_time[balance_mask]
        # Convert to balance metric (1.0 when perfectly balanced, 0.0 when completely unbalanced)
        # 4 * p * (1-p) gives 1.0 when p=0.5 and decreases as p approaches 0 or 1
        balance_reward[balance_mask] = 4.0 * left_ratio[balance_mask] * (1.0 - left_ratio[balance_mask])
    
    # Penalty for staying on the same foot too long (increases exponentially)
    same_foot_penalty = torch.clamp(env._consecutive_same_foot, max=5.0) ** 2 / 25.0
    
    # Compute reward (single tensor operation)
    # Only apply rewards when the robot is moving
    reward = torch.where(
        is_moving,
        transitions.float() - 
        (left_contact & right_contact).float() * double_contact_penalty - 
        (~left_contact & ~right_contact).float() * no_contact_penalty + 
        balance_reward * balance_weight - 
        same_foot_penalty * single_leg_time_penalty,
        torch.zeros_like(total_time)  # Zero reward when not moving
    )
    
    # Reset total times when episode terminates
    if hasattr(env, 'termination_manager') and env.termination_manager.terminated.any():
        env._total_left_time[env.termination_manager.terminated] = 0.0
        env._total_right_time[env.termination_manager.terminated] = 0.0
        env._consecutive_same_foot[env.termination_manager.terminated] = 0.0
    
    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
    
    return reward

def movement_direction_alignment(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    alignment_threshold: float = 0.7,  # Reduced for less strict initial alignment (~45 degrees)
    misalignment_time_threshold: float = 0.3,  # Reduced to respond faster
    penalty_scale: float = 2.0,  # Scale factor for penalty strength
    velocity_threshold: float = 0.1,  # Only apply when moving
    normalise: bool = False,
    normaliser_name: str = "movement_direction_alignment"
) -> torch.Tensor:
    """Penalize movement that's not aligned with the torso direction.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        alignment_threshold: Cosine similarity threshold (0.7 is about 45 degrees)
        misalignment_time_threshold: Time in seconds before applying penalty
        penalty_scale: How strongly to scale the penalty
        velocity_threshold: Minimum velocity to consider "moving"
        normalise: Whether to normalize the reward
        normaliser_name: Name of the normalizer to use
        
    Returns:
        Tensor containing the penalty values (higher when movement is not aligned)
    """
    # Get the robot asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get the robot's velocity in world frame
    lin_vel = asset.data.root_lin_vel_w
    lin_vel_xy = lin_vel[:, :2]
    vel_magnitude = torch.norm(lin_vel_xy, dim=1)
    
    # Check if robot is moving
    is_moving = vel_magnitude > velocity_threshold
    
    # Get the robot's forward direction using quat_apply (simpler approach)
    forward_dir = quat_apply(
        asset.data.root_quat_w, 
        torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3)
    )
    forward_dir_xy = forward_dir[:, :2]
    forward_dir_mag = torch.norm(forward_dir_xy, dim=1)
    
    # Initialize misalignment timers if they don't exist
    if not hasattr(env, '_misalignment_timers'):
        env._misalignment_timers = torch.zeros(env.num_envs, device=env.device)
    
    # Initialize penalty tensor
    penalty = torch.zeros(env.num_envs, device=env.device)
    
    # Only calculate for environments that are moving
    if is_moving.any():
        # Normalize vectors for moving environments
        lin_vel_xy_norm = torch.zeros_like(lin_vel_xy)
        lin_vel_xy_norm[is_moving] = lin_vel_xy[is_moving] / vel_magnitude[is_moving].clamp(min=1e-8).unsqueeze(1)
        
        forward_dir_norm = torch.zeros_like(forward_dir_xy)
        valid_forward = forward_dir_mag > 0
        forward_dir_norm[valid_forward] = forward_dir_xy[valid_forward] / forward_dir_mag[valid_forward].clamp(min=1e-8).unsqueeze(1)
        
        # Calculate cosine similarity (dot product of normalized vectors)
        cos_sim = torch.sum(lin_vel_xy_norm * forward_dir_norm, dim=1)
        
        # Determine which environments are currently misaligned
        current_misaligned = (cos_sim < alignment_threshold) & is_moving
        
        # Update misalignment timers
        env._misalignment_timers = torch.where(
            current_misaligned,
            env._misalignment_timers + env.step_dt,  # Increment by time step
            torch.zeros_like(env._misalignment_timers)  # Reset timer
        )
        
        # Apply penalty only to environments that have been misaligned longer than threshold
        penalty_mask = (env._misalignment_timers > misalignment_time_threshold) & is_moving
        
        if penalty_mask.any():
            # Calculate penalty - stronger penalty for greater misalignment
            # This creates a continuous penalty that increases as alignment decreases
            misalignment = torch.clamp(alignment_threshold - cos_sim, min=0.0)
            # Scale by velocity to penalize more when moving faster in wrong direction
            penalty[penalty_mask] = misalignment[penalty_mask] * vel_magnitude[penalty_mask] * penalty_scale
    
    # Reset timers when episode terminates
    if hasattr(env, 'termination_manager') and env.termination_manager.terminated.any():
        env._misalignment_timers[env.termination_manager.terminated] = 0.0
    
    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        penalty = RewNormalizer.normalize(normaliser_name, penalty)
        RewNormalizer.update_stats(normaliser_name, penalty)
    
    return penalty

def foot_contact_cycling_upgraded(env: ManagerBasedRLEnv,
                    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                    left_foot: str = "left_ankle_roll_link",
                    right_foot: str = "right_ankle_roll_link",
                    contact_threshold: float = 1.0,
                    double_contact_penalty: float = 0.5,
                    no_contact_penalty: float = 1.0,
                    min_contact_time: float = 0.3,
                    balance_weight: float = 1.5,
                    foot_position_weight: float = 0.5,
                    stride_length_weight: float = 0.3,
                    same_foot_time_penalty: float = 2.0,
                    velocity_threshold: float = 0.3,  # Minimum velocity to consider "moving"
                    normalise: bool = False,
                    normaliser_name: str = "foot_contact_cycling") -> torch.Tensor:
    """Reward alternating foot contacts while encouraging human-like walking patterns.
    Only applies rewards when the robot is moving above the velocity threshold."""
    # Get the contact sensor and robot asset
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    # Check if robot is moving (velocity above threshold)
    lin_vel_xy = asset.data.root_lin_vel_w[:, :2]
    is_moving = torch.norm(lin_vel_xy, dim=1) > velocity_threshold

    # Get indices for both feet using body_names (do once and cache)
    if not hasattr(env, '_foot_indices'):
        env._foot_indices = (contact_sensor.body_names.index(left_foot), 
                           contact_sensor.body_names.index(right_foot))
        # Also get the body indices in the asset
        env._foot_asset_indices = (asset.body_names.index(left_foot),
                                    asset.body_names.index(right_foot))
    left_idx, right_idx = env._foot_indices
    left_asset_idx, right_asset_idx = env._foot_asset_indices

    # Compute contact states in one operation
    forces = contact_sensor.data.net_forces_w_history[:, -1, [left_idx, right_idx]]  # Only use latest history
    contacts = torch.norm(forces, dim=-1) > contact_threshold
    left_contact, right_contact = contacts[:, 0], contacts[:, 1]

    # Get contact times
    contact_times = contact_sensor.data.current_contact_time[:, [left_idx, right_idx]]
    left_time, right_time = contact_times[:, 0], contact_times[:, 1]

    # Calculate stable contacts (single tensor operation)
    stable_contacts = contacts & (contact_times > min_contact_time)
    stable_left, stable_right = stable_contacts[:, 0], stable_contacts[:, 1]

    # Initialize state tensors if needed (do once)
    if not hasattr(env, '_prev_foot_contact'):
        env._prev_foot_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._was_left_foot = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._total_left_time = torch.zeros(env.num_envs, device=env.device)
        env._total_right_time = torch.zeros(env.num_envs, device=env.device)
        env._left_always_front = torch.zeros(env.num_envs, device=env.device)
        env._right_always_front = torch.zeros(env.num_envs, device=env.device)
        env._prev_left_front = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._stride_lengths = torch.zeros(env.num_envs, device=env.device)
        env._consecutive_same_foot_time = torch.zeros(env.num_envs, device=env.device)

    # Update total contact times (only when moving)
    env._total_left_time += (left_contact & is_moving).float() * env.step_dt
    env._total_right_time += (right_contact & is_moving).float() * env.step_dt

    # Compute transitions and update states (only when moving)
    transitions = ((env._was_left_foot & stable_right & ~left_contact) | 
                  (~env._was_left_foot & stable_left & ~right_contact)) & is_moving

    # Get foot positions in world frame
    left_foot_pos = asset.data.body_pos_w[:, left_asset_idx]
    right_foot_pos = asset.data.body_pos_w[:, right_asset_idx]

    # Get forward direction from root orientation
    forward_dir = quat_apply(asset.data.root_quat_w, 
                                torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3))
    # Safe normalization with clamp
    forward_dir = forward_dir / torch.norm(forward_dir, dim=1, keepdim=True).clamp(min=1e-8)

    # Project foot positions onto forward axis
    left_proj = torch.sum(left_foot_pos * forward_dir, dim=1)
    right_proj = torch.sum(right_foot_pos * forward_dir, dim=1)

    # Determine which foot is in front
    left_in_front = left_proj > right_proj

    # Calculate stride length (distance between feet along forward direction)
    stride_length = torch.abs(left_proj - right_proj)

    # Optimal stride length is roughly proportional to leg length
    target_stride_min = 0.4
    target_stride_max = 0.6

    # Reward for appropriate stride length (bell curve with peak at target range)
    stride_reward = torch.exp(-4.0 * torch.square(
        torch.clamp(stride_length - target_stride_min, min=0.0) - 
        torch.clamp(stride_length - target_stride_max, min=0.0)
    ))

    # Track if the same foot is always in front (penalize this pattern)
    if transitions.any():
        # When a transition happens, check if foot positions changed
        foot_position_changed = left_in_front != env._prev_left_front
        
        # Update the counters for feet always being in front (only when moving)
        env._left_always_front = torch.where(
            transitions & left_in_front & ~foot_position_changed,
            env._left_always_front + 1.0,
            torch.where(transitions & foot_position_changed | ~is_moving, 
                      torch.zeros_like(env._left_always_front), env._left_always_front)
        )
        
        env._right_always_front = torch.where(
            transitions & ~left_in_front & ~foot_position_changed,
            env._right_always_front + 1.0,
            torch.where(transitions & foot_position_changed | ~is_moving, 
                      torch.zeros_like(env._right_always_front), env._right_always_front)
        )
        
        # Save current foot positions for next comparison
        env._prev_left_front = left_in_front
        
        # Record stride length when transitioning
        env._stride_lengths = torch.where(transitions, stride_length, env._stride_lengths)

    # Penalty for keeping the same foot always in front (sigmoid to cap the penalty)
    same_foot_penalty = 2.0 / (1.0 + torch.exp(-0.5 * (env._left_always_front + env._right_always_front))) - 1.0

    # Update states (in-place operations)
    env._prev_foot_contact = left_contact | right_contact
    env._was_left_foot = torch.where(stable_left, torch.ones_like(env._was_left_foot),
                                    torch.where(stable_right, torch.zeros_like(env._was_left_foot),
                                                env._was_left_foot))

    # Calculate balance reward (higher when times are similar)
    total_time = env._total_left_time + env._total_right_time
    # Avoid division by zero
    balance_mask = total_time > 0
    balance_reward = torch.zeros_like(total_time)
    if balance_mask.any():
        # Calculate ratio of time spent on each foot (closer to 0.5 is better)
        left_ratio = torch.zeros_like(total_time)
        # Safe division with mask
        left_ratio[balance_mask] = env._total_left_time[balance_mask] / total_time[balance_mask].clamp(min=1e-8)
        # Convert to balance metric (1.0 when perfectly balanced, 0.0 when completely unbalanced)
        balance_reward[balance_mask] = 4.0 * left_ratio[balance_mask] * (1.0 - left_ratio[balance_mask])

    # Update consecutive same foot time (only when moving)
    env._consecutive_same_foot_time = torch.where(
        transitions | ~is_moving,  # Reset on transition or when not moving
        torch.zeros_like(env._consecutive_same_foot_time),
        env._consecutive_same_foot_time + env.step_dt  # Increment otherwise
    )
    
    # Exponential penalty for staying on same foot too long
    same_foot_time_penalty_value = torch.clamp(env._consecutive_same_foot_time, max=5.0) ** 2 / 25.0
    
    # Compute reward (only apply when moving)
    reward = torch.where(
        is_moving,
        transitions.float() - 
        (left_contact & right_contact).float() * double_contact_penalty - 
        (~left_contact & ~right_contact).float() * no_contact_penalty + 
        balance_reward * balance_weight + 
        stride_reward * stride_length_weight - 
        same_foot_penalty * foot_position_weight - 
        same_foot_time_penalty_value * same_foot_time_penalty,
        torch.zeros_like(total_time)  # Zero reward when not moving
    )
    
    # Reset counters on termination
    if hasattr(env, 'termination_manager') and env.termination_manager.terminated.any():
        env._total_left_time[env.termination_manager.terminated] = 0.0
        env._total_right_time[env.termination_manager.terminated] = 0.0
        env._left_always_front[env.termination_manager.terminated] = 0.0
        env._right_always_front[env.termination_manager.terminated] = 0.0
        env._stride_lengths[env.termination_manager.terminated] = 0.0
        env._consecutive_same_foot_time[env.termination_manager.terminated] = 0.0

    if normalise:
        if normaliser_name not in RewNormalizer.stats:
            RewNormalizer.stats[normaliser_name] = RewardStats()
        reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)

    return reward