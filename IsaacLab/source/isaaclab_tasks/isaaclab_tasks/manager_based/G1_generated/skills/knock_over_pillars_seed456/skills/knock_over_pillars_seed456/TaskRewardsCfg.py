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


def main_knock_over_pillar_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the 'knock_over_pillars_seed456' skill.
    This reward encourages the robot to approach Object1, knock it over, and maintain pelvis stability.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Object dimensions (from problem description) - CRITICAL: Hardcoded from object config, not accessed from object instance
    pillar_height = 2.0
    pillar_radius = 0.3

    # Access the required objects and robot parts using approved patterns
    robot = env.scene["robot"]
    object1 = env.scene['Object1'] # CRITICAL: Accessing object directly by its scene name

    pelvis_idx = robot.body_names.index('pelvis') # CRITICAL: Accessing robot part index
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CRITICAL: Accessing robot part position

    object1_pos = object1.data.root_pos_w # CRITICAL: Accessing object position

    # Calculate horizontal distance to pillar (relative distance)
    # CRITICAL: Using relative distances between robot pelvis and object
    distance_x_to_pillar = object1_pos[:, 0] - pelvis_pos[:, 0]
    distance_y_to_pillar = object1_pos[:, 1] - pelvis_pos[:, 1]
    horizontal_distance = torch.norm(torch.stack([distance_x_to_pillar, distance_y_to_pillar], dim=-1), dim=-1)

    # Reward for approaching the pillar (negative of distance, so closer is higher reward)
    # CRITICAL: Continuous reward based on relative distance
    approach_reward = -horizontal_distance

    # Reward for knocking over the pillar (pillar's Z-height should decrease)
    # The goal is for the pillar to be lying on the floor, meaning its Z-height should be approximately its radius.
    # CRITICAL: Using relative Z-position of the object to the ground (implicitly 0)
    # CRITICAL: Continuous reward based on object's Z-position
    pillar_z_pos = object1_pos[:, 2]
    
    # Target Z for a fallen pillar is its radius.
    target_fallen_z = pillar_radius
    
    # Reward for pillar's Z being close to its radius (lying down)
    # This term will be more positive as pillar_z_pos gets closer to target_fallen_z
    # The prompt's reward design plan for pillar_z_reward was:
    # pillar_z_reward = -(object1.data.root_pos_w[:, 2] - pillar_radius)
    # pillar_z_reward = torch.where(object1.data.root_pos_w[:, 2] < (pillar_height / 2.0), -pillar_z_reward, 0.0)
    # This can be simplified and made more robust.
    # A negative absolute difference from the target_fallen_z will reward being close to it.
    pillar_fall_reward = -torch.abs(pillar_z_pos - target_fallen_z)

    # Activation condition for knock_over_reward: robot is within 1.0m horizontally
    # CRITICAL: Using relative distance for activation condition
    interaction_threshold = 1.0
    knock_over_condition = horizontal_distance < interaction_threshold
    
    # Apply pillar_fall_reward only when the robot is close enough to interact
    # CRITICAL: Conditional reward for smooth transition
    knock_over_reward = torch.where(knock_over_condition, pillar_fall_reward, torch.tensor(0.0, device=env.device))

    # Reward for maintaining stable pelvis height (around 0.7m)
    # CRITICAL: Using relative Z-position of pelvis to a target height (implicitly 0)
    target_pelvis_z = 0.7
    pelvis_stability_reward = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Combine rewards with weights
    # CRITICAL: Continuous combination of rewards
    reward = approach_reward * 0.5 + knock_over_reward * 0.4 + pelvis_stability_reward * 0.1

    # CRITICAL: Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def hand_interaction_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "hand_interaction_reward") -> torch.Tensor:
    """
    Shaping reward 1: Encourages the robot to use its hand to interact with the pillar.
    Rewards the robot for having its right hand close to the pillar's surface at a suitable height.
    This reward is active once the robot is within a certain horizontal distance to the pillar.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Object dimensions (from problem description) - CRITICAL: Hardcoded from object config
    pillar_height = 2.0
    pillar_radius = 0.3

    # Access the required objects and robot parts
    robot = env.scene["robot"]
    object1 = env.scene['Object1'] # CRITICAL: Accessing object directly

    right_palm_idx = robot.body_names.index('right_palm_link') # CRITICAL: Accessing robot part index
    right_palm_pos = robot.data.body_pos_w[:, right_palm_idx] # CRITICAL: Accessing robot part position

    object1_pos = object1.data.root_pos_w # CRITICAL: Accessing object position

    # Calculate horizontal distance from right hand to pillar center (relative distance)
    # CRITICAL: Using relative distances between robot hand and object
    hand_dist_x = object1_pos[:, 0] - right_palm_pos[:, 0]
    hand_dist_y = object1_pos[:, 1] - right_palm_pos[:, 1]
    hand_horizontal_distance = torch.norm(torch.stack([hand_dist_x, hand_dist_y], dim=-1), dim=-1)

    # Target Z-height for hand interaction (e.g., middle of the pillar)
    # CRITICAL: Using object dimension for target height
    target_hand_z = pillar_height / 2.0

    # Reward for hand being close to pillar surface (radius away from center)
    # CRITICAL: Continuous reward based on relative distance to pillar surface
    hand_to_pillar_surface_reward = -torch.abs(hand_horizontal_distance - pillar_radius)
    
    # Reward for hand being at a suitable height
    # CRITICAL: Continuous reward based on relative Z-position of hand to target height
    hand_height_reward = -torch.abs(right_palm_pos[:, 2] - target_hand_z)

    # Activation condition: robot's pelvis is close enough to interact with the pillar
    # CRITICAL: Using relative distance for activation condition
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_horizontal_distance = torch.norm(object1_pos[:, :2] - pelvis_pos[:, :2], dim=-1)
    interaction_condition = pelvis_horizontal_distance < 1.5 # Slightly larger threshold for hand to reach

    # Combine hand interaction rewards and apply activation condition
    # CRITICAL: Conditional reward for smooth transition
    reward = (hand_to_pillar_surface_reward * 0.7 + hand_height_reward * 0.3)
    reward = torch.where(interaction_condition, reward, torch.tensor(0.0, device=env.device))

    # CRITICAL: Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def stability_and_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stability_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward 2: Encourages collision avoidance with the pillar during approach,
    feet staying on or near the ground, and penalizes the robot for falling over.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Object dimensions (from problem description) - CRITICAL: Hardcoded from object config
    pillar_radius = 0.3

    # Access the required objects and robot parts
    robot = env.scene["robot"]
    object1 = env.scene['Object1'] # CRITICAL: Accessing object directly

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    left_ankle_idx = robot.body_names.index('left_ankle_roll_link')
    left_ankle_pos = robot.data.body_pos_w[:, left_ankle_idx]

    right_ankle_idx = robot.body_names.index('right_ankle_roll_link')
    right_ankle_pos = robot.data.body_pos_w[:, right_ankle_idx]

    object1_pos = object1.data.root_pos_w

    # Collision avoidance with pillar during approach (prevent running into it too hard)
    # Penalize if pelvis is too close to pillar center, but not yet interacting
    # CRITICAL: Using relative distances between robot pelvis and object
    pelvis_dist_x = object1_pos[:, 0] - pelvis_pos[:, 0]
    pelvis_dist_y = object1_pos[:, 1] - pelvis_pos[:, 1]
    pelvis_horizontal_distance = torch.norm(torch.stack([pelvis_dist_x, pelvis_dist_y], dim=-1), dim=-1)

    # A small buffer beyond the pillar radius for collision penalty
    clearance_buffer = 0.1
    collision_penalty_threshold = pillar_radius + clearance_buffer
    
    # Penalize if pelvis is closer than the threshold, using an exponential penalty for continuity
    # CRITICAL: Continuous penalty based on relative distance
    collision_penalty = torch.where(pelvis_horizontal_distance < collision_penalty_threshold,
                                    -torch.exp(-(pelvis_horizontal_distance - pillar_radius)), # Exponential penalty
                                    torch.tensor(0.0, device=env.device))

    # Reward for feet staying on or near the ground (avoiding floating or clipping)
    # Assuming ground is at z=0. Reward for z-position of feet being slightly above 0
    # CRITICAL: Using relative Z-position of feet to a target ground clearance
    foot_ground_clearance = 0.05 # Small positive value for feet to be just above ground
    left_foot_reward = -torch.abs(left_ankle_pos[:, 2] - foot_ground_clearance)
    right_foot_reward = -torch.abs(right_ankle_pos[:, 2] - foot_ground_clearance)

    # Penalty for falling over (pelvis Z too low)
    # CRITICAL: Using relative Z-position of pelvis to a fall threshold
    fall_penalty_threshold = 0.4 # If pelvis Z drops below this, robot is likely falling
    # Larger negative reward for falling, continuous penalty
    fall_penalty = torch.where(pelvis_pos[:, 2] < fall_penalty_threshold,
                               -10.0 * (fall_penalty_threshold - pelvis_pos[:, 2]),
                               torch.tensor(0.0, device=env.device))

    # Combine rewards with weights
    # CRITICAL: Continuous combination of rewards
    reward = collision_penalty * 0.3 + (left_foot_reward + right_foot_reward) * 0.3 + fall_penalty * 0.4

    # CRITICAL: Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    """
    Reward terms for the 'knock_over_pillars_seed456' skill.
    """
    # Primary reward for approaching and knocking over the pillar, and pelvis stability
    MainKnockOverPillarReward = RewTerm(func=main_knock_over_pillar_reward, weight=1.0,
                                        params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for hand interaction with the pillar
    HandInteractionReward = RewTerm(func=hand_interaction_reward, weight=0.5,
                                    params={"normalise": True, "normaliser_name": "hand_interaction_reward"})

    # Shaping reward for stability and collision avoidance
    StabilityAndAvoidanceReward = RewTerm(func=stability_and_avoidance_reward, weight=0.2, # Reduced weight as it's more of a safety/stability reward
                                          params={"normalise": True, "normaliser_name": "stability_avoidance_reward"})