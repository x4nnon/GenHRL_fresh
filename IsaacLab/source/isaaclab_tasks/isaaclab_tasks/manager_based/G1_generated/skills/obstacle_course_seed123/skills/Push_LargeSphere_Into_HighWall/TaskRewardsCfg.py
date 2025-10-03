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


def primary_push_sphere_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "primary_push_sphere_reward") -> torch.Tensor:
    """
    Primary reward for the Push_LargeSphere_Into_HighWall skill.
    This reward encourages the large sphere (Object1) to move towards and past the high wall (Object4)'s initial x-position,
    indicating it has been pushed through the wall. It also encourages the robot's pelvis to be positioned behind Object1
    to facilitate pushing, and then to stay within a reasonable range after the push, not overshooting Object2.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    large_sphere = env.scene['Object1']
    high_wall = env.scene['Object4']
    small_sphere = env.scene['Object2']

    # Access the required robot part(s) using approved patterns
    robot = env.scene["robot"] # Accessing robot object
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_z = pelvis_pos[:, 2] # Used for stability check, not direct reward here

    # Object dimensions (hardcoded from config as per requirements)
    large_sphere_radius = 1.0
    high_wall_x_dim = 0.3
    small_sphere_radius = 0.2

    # Calculate distances using relative positions
    # Distance from large sphere to high wall (x-component)
    # We want the sphere's front edge to pass the wall's back edge
    # This is a relative distance calculation.
    sphere_front_x = large_sphere.data.root_pos_w[:, 0] + large_sphere_radius
    wall_back_x = high_wall.data.root_pos_w[:, 0] + (high_wall_x_dim / 2.0)
    distance_sphere_to_wall_x = sphere_front_x - wall_back_x

    # Distance from robot pelvis to large sphere (x-component)
    # Robot should be behind the sphere to push it. This is a relative distance.
    pelvis_to_sphere_x = large_sphere.data.root_pos_w[:, 0] - pelvis_pos_x

    # Distance from robot pelvis to small sphere (x-component) for next skill. This is a relative distance.
    # pelvis_to_small_sphere_x = small_sphere.data.root_pos_w[:, 0] - pelvis_pos_x # This variable is not used, removed to avoid unused variable warning.

    # Reward for pushing the large sphere past the high wall
    # Reward is higher as sphere moves past the wall (negative distance_sphere_to_wall_x)
    # Use a sigmoid-like function to cap the reward once far past the wall, ensuring continuity.
    push_progress_reward = 1.0 / (1.0 + torch.exp(distance_sphere_to_wall_x * 2.0))

    # Reward for robot being behind the large sphere to push it
    # Encourage pelvis to be within a reasonable pushing range behind the sphere (e.g., 0.5m to 1.5m behind its center)
    # This reward is active when the sphere is still in front of the wall.
    # The target is 1.0m behind the sphere's center.
    approach_sphere_condition = (large_sphere.data.root_pos_w[:, 0] < high_wall.data.root_pos_w[:, 0] + high_wall_x_dim)
    pelvis_behind_sphere_reward = -torch.abs(pelvis_to_sphere_x - 1.0) # Target 1.0m behind sphere center
    pelvis_behind_sphere_reward = torch.where(approach_sphere_condition, pelvis_behind_sphere_reward, 0.0)
    # Normalize pelvis_behind_sphere_reward to be between 0 and 1 for better scaling
    pelvis_behind_sphere_reward = torch.exp(pelvis_behind_sphere_reward) # Max 1 when distance is 0, decays exponentially

    # Penalty for overshooting the small sphere (next skill target)
    # Robot should not go significantly past the high wall's original position, and certainly not past the small sphere.
    # This is a continuous penalty based on how far past the small sphere the robot's pelvis is.
    overshoot_threshold_x = small_sphere.data.root_pos_w[:, 0] - small_sphere_radius # Front edge of small sphere
    overshoot_penalty = torch.where(pelvis_pos_x > overshoot_threshold_x,
                                    -torch.abs(pelvis_pos_x - overshoot_threshold_x),
                                    0.0)

    reward = push_progress_reward + pelvis_behind_sphere_reward + overshoot_penalty

    # Mandatory normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_hand_proximity_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_hand_proximity_reward") -> torch.Tensor:
    """
    Shaping reward that encourages the robot's hands to be close to the large sphere (Object1) to facilitate pushing.
    It is active when the large sphere has not yet passed the high wall.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    large_sphere = env.scene['Object1']
    high_wall = env.scene['Object4']

    # Access the required robot part(s) using approved patterns
    robot = env.scene["robot"] # Accessing robot object
    left_hand_idx = robot.body_names.index('left_palm_link')
    right_hand_idx = robot.body_names.index('right_palm_link')
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]

    # Object dimensions (hardcoded from config as per requirements)
    large_sphere_radius = 1.0
    high_wall_x_dim = 0.3

    # Calculate distances using relative positions (Euclidean distance)
    distance_left_hand_to_sphere = torch.norm(left_hand_pos - large_sphere.data.root_pos_w, dim=-1)
    distance_right_hand_to_sphere = torch.norm(right_hand_pos - large_sphere.data.root_pos_w, dim=-1)

    # Condition: Sphere has not yet passed the wall
    # This condition uses relative positions to determine if the sphere is still in front of the wall.
    sphere_not_past_wall_condition = (large_sphere.data.root_pos_w[:, 0] + large_sphere_radius < high_wall.data.root_pos_w[:, 0] + (high_wall_x_dim / 2.0) + 0.1) # Add small buffer

    # Reward for hands being close to the sphere, capped at sphere radius for contact.
    # This is a continuous reward, penalizing deviation from the target distance (sphere radius for contact).
    hand_proximity_reward_left = -torch.abs(distance_left_hand_to_sphere - large_sphere_radius)
    hand_proximity_reward_right = -torch.abs(distance_right_hand_to_sphere - large_sphere_radius)

    # Apply condition: reward is only active when the sphere is still in front of the wall.
    reward = torch.where(sphere_not_past_wall_condition, hand_proximity_reward_left + hand_proximity_reward_right, 0.0)
    # Normalize the reward to be positive and between 0 and 1 for better scaling
    reward = torch.exp(reward) # Max 1 when hands are at target distance, decays exponentially

    # Mandatory normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_pelvis_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_pelvis_stability_reward") -> torch.Tensor:
    """
    Shaping reward that encourages the robot to maintain a stable upright posture (pelvis z-height around 0.7m)
    throughout the skill, which is crucial for effective pushing and preparing for the next skill.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot part(s) using approved patterns
    robot = env.scene["robot"] # Accessing robot object
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2] # Z-component of pelvis position (absolute height, allowed for stability)

    # Target pelvis height (hardcoded as per plan)
    target_pelvis_z = 0.7

    # Reward for maintaining pelvis at target height.
    # This is a continuous reward, penalizing deviation from the target Z-height.
    reward = -torch.abs(pelvis_pos_z - target_pelvis_z)
    # Normalize the reward to be positive and between 0 and 1 for better scaling
    reward = torch.exp(reward * 2.0) # Multiply by 2.0 to make it decay faster, max 1 when at target height

    # Mandatory normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

@configclass
class TaskRewardsCfg:
    # Primary reward for pushing the sphere and robot positioning
    PrimaryPushSphereReward = RewTerm(func=primary_push_sphere_reward, weight=1.0,
                                      params={"normalise": True, "normaliser_name": "primary_push_sphere_reward"})

    # Shaping reward for hand proximity to the sphere
    ShapingHandProximityReward = RewTerm(func=shaping_hand_proximity_reward, weight=0.4,
                                         params={"normalise": True, "normaliser_name": "shaping_hand_proximity_reward"})

    # Shaping reward for maintaining pelvis stability
    ShapingPelvisStabilityReward = RewTerm(func=shaping_pelvis_stability_reward, weight=0.2,
                                            params={"normalise": True, "normaliser_name": "shaping_pelvis_stability_reward"})