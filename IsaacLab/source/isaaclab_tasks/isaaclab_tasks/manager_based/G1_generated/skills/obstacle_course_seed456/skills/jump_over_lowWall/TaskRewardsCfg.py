from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.reward_normalizer import get_normalizer, RewardStats
from genhrl.generation.objects import get_object_volume
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
import torch

from isaaclab.envs import mdp
import genhrl.generation.mdp.rewards as custom_rewards
import genhrl.generation.mdp.terminations as custom_terminations
import genhrl.generation.mdp.observations as custom_observations
import genhrl.generation.mdp.events as custom_events
import genhrl.generation.mdp.curriculums as custom_curriculums


def main_jump_over_lowWall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_jump_over_lowWall_reward") -> torch.Tensor:
    '''Main reward for jump_over_lowWall.

    This reward guides the robot through the entire jump sequence: approaching the wall, clearing it, and landing stably on the other side.
    It combines three phases: approach, clearance, and landing, each with specific conditions and reward components.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects using approved patterns
    # Object3 is the low wall for robot to jump over
    low_wall = env.scene['Object3']
    # Object1 is the large sphere for robot to push (target for next skill)
    large_sphere = env.scene['Object1']

    # Access required robot parts using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_z = pelvis_pos[:, 2]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    left_foot_pos_z = left_foot_pos[:, 2]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    right_foot_pos_z = right_foot_pos[:, 2]

    # Object dimensions (hardcoded from description, as per requirements)
    low_wall_height = 0.5 # 0.5m in z
    low_wall_thickness = 0.3 # 0.3m in x

    # Low wall positions (relative to its root position)
    # Accessing object positions using approved pattern
    low_wall_x = low_wall.data.root_pos_w[:, 0]
    low_wall_z = low_wall.data.root_pos_w[:, 2]
    low_wall_front_x = low_wall_x - (low_wall_thickness / 2.0) # Front edge of the wall in x
    low_wall_back_x = low_wall_x + (low_wall_thickness / 2.0) # Back edge of the wall in x
    low_wall_top_z = low_wall_z + low_wall_height # Top surface of the wall in z

    # Large sphere position (for defining the landing zone for the next skill)
    # Accessing object positions using approved pattern
    large_sphere_x = large_sphere.data.root_pos_w[:, 0]

    # 1. Approach and Pre-Jump Reward
    # Rewards the robot for moving its pelvis towards the front of the low wall.
    # This reward is active only when the pelvis is in front of the wall.
    # Using relative distance for reward calculation
    approach_distance = torch.abs(pelvis_pos_x - low_wall_front_x)
    approach_reward = -approach_distance # Negative reward for distance, encouraging approach
    approach_condition = pelvis_pos_x < low_wall_front_x # Condition: pelvis is before the front of the wall
    # Using torch.where for conditional reward, ensuring continuity
    approach_reward = torch.where(approach_condition, approach_reward, torch.tensor(0.0, device=env.device))

    # 2. Clearance and Jump Apex Reward
    # Rewards the robot for increasing its pelvis and feet z-height relative to the low wall's top surface.
    # This reward is active when the robot's pelvis is horizontally aligned with or just past the wall's front edge.
    # Using relative distances for reward calculation
    clearance_pelvis_reward = (pelvis_pos_z - low_wall_top_z) * 2.0 # Encourage higher clearance for pelvis
    clearance_left_foot_reward = (left_foot_pos_z - low_wall_top_z) * 2.0 # Encourage higher clearance for left foot
    clearance_right_foot_reward = (right_foot_pos_z - low_wall_top_z) * 2.0 # Encourage higher clearance for right foot

    # Condition: pelvis is over or past the wall's front edge
    clearance_condition = pelvis_pos_x >= low_wall_front_x
    # Sum of clearance rewards, active only when the condition is met, ensuring continuity
    clearance_reward = torch.where(clearance_condition, clearance_pelvis_reward + clearance_left_foot_reward + clearance_right_foot_reward, torch.tensor(0.0, device=env.device))

    # 3. Landing and Post-Jump Positioning Reward
    # Rewards the robot for landing stably on the ground on the far side of the low wall,
    # and for positioning its pelvis within the safe zone between the low wall and the large sphere.
    target_pelvis_z = 0.7 # Desired stable standing height for the robot (hardcoded from plan)
    # Penalty for deviation from target stable standing height, using relative distance
    landing_stability_reward = -torch.abs(pelvis_pos_z - target_pelvis_z) * 5.0 # Strong penalty for not being at target height

    # Target x-zone: between low_wall_back_x and large_sphere_x. Calculate the center of this zone.
    target_x_zone_center = (low_wall_back_x + large_sphere_x) / 2.0
    # Penalty for deviation from the target x-zone center, using relative distance
    landing_position_reward = -torch.abs(pelvis_pos_x - target_x_zone_center)

    # Condition: pelvis is past the back of the low wall
    landing_condition = pelvis_pos_x > low_wall_back_x
    # Sum of landing rewards, active only when the condition is met, ensuring continuity
    landing_reward = torch.where(landing_condition, landing_stability_reward + landing_position_reward, torch.tensor(0.0, device=env.device))

    # Combine all reward components
    reward = approach_reward + clearance_reward + landing_reward

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_lowWall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_lowWall_reward") -> torch.Tensor:
    '''Shaping reward for collision avoidance with the low wall.

    Penalizes the robot if any part of its body (pelvis, feet) collides with or gets too close to the low wall.
    This encourages the robot to jump over cleanly rather than pushing through.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts using approved patterns
    low_wall = env.scene['Object3']
    robot = env.scene["robot"]

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Object dimensions (hardcoded from description)
    low_wall_thickness = 0.3 # 0.3m in x
    low_wall_height = 0.5 # 0.5m in z
    low_wall_width_y = 5.0 # 5m in y (from description)

    # Low wall positions (using approved pattern)
    low_wall_x = low_wall.data.root_pos_w[:, 0]
    low_wall_y = low_wall.data.root_pos_w[:, 1]
    low_wall_z = low_wall.data.root_pos_w[:, 2]

    # Define a collision box around the wall with a small buffer for avoidance
    buffer = 0.1 # Small buffer around the wall
    wall_min_x = low_wall_x - (low_wall_thickness / 2.0) - buffer
    wall_max_x = low_wall_x + (low_wall_thickness / 2.0) + buffer
    wall_min_z = low_wall_z - buffer # Allow robot to be slightly below wall base
    wall_max_z = low_wall_z + low_wall_height + buffer # Add buffer above wall
    # The wall is 5m in y-axis, so we consider a wide range for y to cover the entire wall
    wall_min_y = low_wall_y - (low_wall_width_y / 2.0) - buffer
    wall_max_y = low_wall_y + (low_wall_width_y / 2.0) + buffer

    # Check collision condition for pelvis using relative positions to wall boundaries
    pelvis_collision_x_cond = (pelvis_pos[:, 0] > wall_min_x) & (pelvis_pos[:, 0] < wall_max_x)
    pelvis_collision_y_cond = (pelvis_pos[:, 1] > wall_min_y) & (pelvis_pos[:, 1] < wall_max_y)
    pelvis_collision_z_cond = (pelvis_pos[:, 2] > wall_min_z) & (pelvis_pos[:, 2] < wall_max_z)
    pelvis_collision_cond = pelvis_collision_x_cond & pelvis_collision_y_cond & pelvis_collision_z_cond

    # Check collision condition for left foot using relative positions to wall boundaries
    left_foot_collision_x_cond = (left_foot_pos[:, 0] > wall_min_x) & (left_foot_pos[:, 0] < wall_max_x)
    left_foot_collision_y_cond = (left_foot_pos[:, 1] > wall_min_y) & (left_foot_pos[:, 1] < wall_max_y)
    left_foot_collision_z_cond = (left_foot_pos[:, 2] > wall_min_z) & (left_foot_pos[:, 2] < wall_max_z)
    left_foot_collision_cond = left_foot_collision_x_cond & left_foot_collision_y_cond & left_foot_collision_z_cond

    # Check collision condition for right foot using relative positions to wall boundaries
    right_foot_collision_x_cond = (right_foot_pos[:, 0] > wall_min_x) & (right_foot_pos[:, 0] < wall_max_x)
    right_foot_collision_y_cond = (right_foot_pos[:, 1] > wall_min_y) & (right_foot_pos[:, 1] < wall_max_y)
    right_foot_collision_z_cond = (right_foot_pos[:, 2] > wall_min_z) & (right_foot_pos[:, 2] < wall_max_z)
    right_foot_collision_cond = right_foot_collision_x_cond & right_foot_collision_y_cond & right_foot_collision_z_cond

    # Apply a continuous penalty based on how deep into the wall the part is, or a fixed penalty if within the zone.
    # Using a fixed negative penalty for simplicity as per plan, but a continuous one could be more nuanced.
    collision_penalty_value = -1.0 # A fixed penalty for being inside the collision zone

    # Applying penalty using torch.where for continuity
    pelvis_collision_reward = torch.where(pelvis_collision_cond, torch.tensor(collision_penalty_value, device=env.device), torch.tensor(0.0, device=env.device))
    left_foot_collision_reward = torch.where(left_foot_collision_cond, torch.tensor(collision_penalty_value, device=env.device), torch.tensor(0.0, device=env.device))
    right_foot_collision_reward = torch.where(right_foot_collision_cond, torch.tensor(collision_penalty_value, device=env.device), torch.tensor(0.0, device=env.device))

    reward = pelvis_collision_reward + left_foot_collision_reward + right_foot_collision_reward

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def maintain_upright_posture_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "maintain_upright_posture_reward") -> torch.Tensor:
    '''Shaping reward for maintaining an upright and stable posture.

    Encourages the robot to maintain an upright and stable posture throughout the skill, especially during landing.
    It penalizes large deviations of the pelvis's z-axis from the target standing height (0.7m) and encourages a stable y-position.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required robot parts using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2]
    pelvis_pos_y = pelvis_pos[:, 1]

    # Target stable standing height (relative to ground)
    target_pelvis_z = 0.7 # Hardcoded from plan

    # Reward for maintaining pelvis z close to target_pelvis_z
    # This is a continuous negative reward, always active, penalizing deviation from target height.
    # Using relative distance for reward calculation
    pelvis_z_stability_reward = -torch.abs(pelvis_pos_z - target_pelvis_z) * 0.5

    # Reward for maintaining pelvis y close to 0 (assuming robot starts at y=0 and wall is centered at y=0)
    # This is a continuous negative reward, always active, penalizing deviation from the central y-axis.
    # Using relative distance for reward calculation (relative to 0)
    pelvis_y_stability_reward = -torch.abs(pelvis_pos_y) * 0.2

    reward = pelvis_z_stability_reward + pelvis_y_stability_reward

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Main reward for the jump_over_lowWall skill, weighted at 1.0 as it's the primary objective.
    Main_JumpOverLowWallReward = RewTerm(func=main_jump_over_lowWall_reward, weight=1.0,
                                params={"normalise": True, "normaliser_name": "main_jump_over_lowWall_reward"})

    # Shaping reward for collision avoidance with the low wall, weighted at 0.4 to encourage clean jumps.
    CollisionAvoidanceLowWallReward = RewTerm(func=collision_avoidance_lowWall_reward, weight=0.4,
                                params={"normalise": True, "normaliser_name": "collision_avoidance_lowWall_reward"})

    # Shaping reward for maintaining an upright posture, weighted at 0.3 for overall stability.
    MaintainUprightPostureReward = RewTerm(func=maintain_upright_posture_reward, weight=0.3,
                                params={"normalise": True, "normaliser_name": "maintain_upright_posture_reward"})