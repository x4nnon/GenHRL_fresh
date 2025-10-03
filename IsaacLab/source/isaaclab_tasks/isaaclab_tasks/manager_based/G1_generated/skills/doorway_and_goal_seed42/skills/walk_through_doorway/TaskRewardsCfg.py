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


def main_walk_through_doorway_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the walk_through_doorway skill.
    Encourages the robot's pelvis to move +0.5m past both Heavy Cube walls in the x-direction,
    clearing the doorway and reaching a target position beyond it.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts
    object1 = env.scene['Object1'] # Heavy Cube (Wall 1)
    object2 = env.scene['Object2'] # Heavy Cube (Wall 2)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Calculate target x position: 0.5m past both walls
    # Find the rightmost x position of either wall and add 0.5m
    # CORRECT: Using relative distance to object positions
    rightmost_wall_x = torch.maximum(object1.data.root_pos_w[:, 0], object2.data.root_pos_w[:, 0])
    target_x_pos = rightmost_wall_x + 0.5

    # Calculate the distance in x-direction to the target
    # CORRECT: Using relative distance for reward calculation
    distance_x = target_x_pos - pelvis_pos_x

    # Primary reward: Negative absolute distance to the target x-position.
    # This reward is maximized when pelvis_pos_x is equal to target_x_pos.
    # It penalizes both being too far behind and overshooting.
    # CORRECT: Continuous reward based on relative distance
    primary_reward = -torch.abs(distance_x)

    # Add a small bonus for being past both walls, but not too far.
    # This encourages clearing the doorway. The "cleared" threshold is 0.25m past the rightmost wall.
    # The bonus is designed to be positive when the robot is past the walls and close to the target.
    # CORRECT: Using relative distance for condition and reward
    cleared_doorway_condition = pelvis_pos_x > (rightmost_wall_x + 0.25)
    primary_reward = primary_reward + torch.where(cleared_doorway_condition, 0.5 - torch.abs(pelvis_pos_x - target_x_pos), 0.0)

    # Penalize if pelvis z is too low (falling or crouching excessively).
    # CORRECT: Using relative distance (from ground) for penalty
    pelvis_z_penalty = torch.where(pelvis_pos_z < 0.5, -5.0, 0.0) # Penalize if pelvis is below 0.5m
    primary_reward = primary_reward + pelvis_z_penalty

    # Combine all reward components
    reward = primary_reward

    # MANDATORY: Reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def doorway_clearance_x_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "doorway_x_clearance_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to stay within the x-bounds of the doorway opening.
    Penalizes the robot if its pelvis moves too far left or right relative to the center of the doorway.
    This reward is active only when the robot is within the y-range of the doorway.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts
    object1 = env.scene['Object1'] # Heavy Cube (Wall 1) - left wall
    object2 = env.scene['Object2'] # Heavy Cube (Wall 2) - right wall
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]

    # Calculate the center x-position of the doorway.
    # The doorway gap is 0.5m. Walls are positioned to form this gap.
    # Assuming Object1 and Object2 define the x-bounds of the doorway.
    # CORRECT: Using relative positions of objects to define doorway center
    doorway_center_x = (object1.data.root_pos_w[:, 0] + object2.data.root_pos_w[:, 0]) / 2.0
    doorway_half_width = 0.5 / 2.0 # Half of the 0.5m doorway gap

    # Calculate distance from pelvis x-position to the doorway center x-position
    # CORRECT: Using relative distance for reward calculation
    distance_to_center_x = torch.abs(pelvis_pos_x - doorway_center_x)

    # Reward for staying within the doorway's x-bounds.
    # Max reward when distance_to_center_x is 0 (pelvis is at the center).
    # Penalty increases linearly as pelvis moves away from the center.
    # CORRECT: Continuous reward based on relative distance
    shaping_reward_1 = -distance_to_center_x

    # Add a stronger penalty if completely outside the doorway's x-bounds.
    # This applies when the distance to center exceeds the half-width of the doorway.
    # CORRECT: Using relative distance for condition and penalty
    outside_left = pelvis_pos_x < (doorway_center_x - doorway_half_width)
    outside_right = pelvis_pos_x > (doorway_center_x + doorway_half_width)
    outside_doorway_condition = outside_left | outside_right

    shaping_reward_1 = shaping_reward_1 + torch.where(outside_doorway_condition, -5.0 * (distance_to_center_x - doorway_half_width), 0.0)

    # This reward should only be active when the robot is within the y-range of the doorway.
    # Wall dimensions: x=0.5m (thickness), y=5m (length).
    # Assuming walls are aligned along the y-axis, and the robot walks along y.
    # The front of Wall 1 (Object1) and the back of Wall 2 (Object2) define the y-extent of the doorway.
    # Assuming wall thickness is 0.5m (from "x of 0.5" for the cube, interpreted as thickness along robot's path).
    # CORRECT: Using relative positions of objects to define activation range
    wall_thickness_y = 0.5
    wall1_front_y = object1.data.root_pos_w[:, 1] - (wall_thickness_y / 2.0)
    wall2_back_y = object2.data.root_pos_w[:, 1] + (wall_thickness_y / 2.0)

    in_doorway_y_condition = (pelvis_pos_y > wall1_front_y) & (pelvis_pos_y < wall2_back_y)

    # Apply the reward only when the robot is within the doorway's y-range
    reward = torch.where(in_doorway_y_condition, shaping_reward_1, 0.0)

    # MANDATORY: Reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def maintain_pelvis_height_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_height_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to maintain a stable, upright posture
    by keeping its pelvis at a desired height (around 0.7m).
    This helps prevent the robot from falling or crouching excessively.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required robot part
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Desired pelvis height
    # CORRECT: Using a fixed desired height for stability, this is an absolute Z position
    desired_pelvis_z = 0.7

    # Calculate the absolute difference from the desired height
    # CORRECT: Using relative distance (from desired height) for reward calculation
    distance_z = torch.abs(pelvis_pos_z - desired_pelvis_z)

    # Reward is negative of the absolute distance, so it's maximized at 0.7m.
    # CORRECT: Continuous reward based on relative distance
    reward = -distance_z

    # This reward is always active to encourage general stability.

    # MANDATORY: Reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Primary reward for clearing the doorway and reaching the target y-position
    MainWalkThroughDoorwayReward = RewTerm(func=main_walk_through_doorway_reward, weight=1.0,
                                           params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for maintaining x-axis alignment within the doorway
    DoorwayClearanceXReward = RewTerm(func=doorway_clearance_x_reward, weight=0.4,
                                      params={"normalise": True, "normaliser_name": "doorway_x_clearance_reward"})

    # Shaping reward for maintaining stable pelvis height
    MaintainPelvisHeightReward = RewTerm(func=maintain_pelvis_height_reward, weight=0.2,
                                         params={"normalise": True, "normaliser_name": "pelvis_height_reward"})
    

    