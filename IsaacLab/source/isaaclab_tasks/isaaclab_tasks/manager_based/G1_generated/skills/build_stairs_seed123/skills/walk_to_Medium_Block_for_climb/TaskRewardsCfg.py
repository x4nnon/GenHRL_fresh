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


def walk_to_medium_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_medium_block_reward") -> torch.Tensor:
    """
    Primary reward for walking the robot's pelvis to a target position relative to Object2 (Medium Block).
    The target is slightly in front of the Medium Block, centered in Y, and at a stable standing height.
    """
    # Get normalizer instance as per mandatory normalization rule.
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns.
    robot = env.scene["robot"] # Accessing robot using approved pattern.
    object2 = env.scene['Object2'] # Medium Block - Accessing object using approved pattern.
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern.

    # Object2 dimensions (hardcoded from object configuration as per rules).
    # From object configuration: Medium Block measuring x=1m y=1m and z=0.6m.
    object2_size_x = 1.0 # Medium Block x dimension
    object2_size_y = 1.0 # Medium Block y dimension
    object2_size_z = 0.6 # Medium Block z dimension

    # Calculate target position relative to Object2.
    # Target x: slightly in front of Object2's face (0.2m clearance from the front face).
    # Target y: Centered with Object2 in y.
    # Target z: Desired stable pelvis height (0.7m).
    # All target calculations are relative to Object2's position, adhering to the relative distance rule.
    target_x = object2.data.root_pos_w[:, 0] + (object2_size_x / 2.0) + 0.2
    target_y = object2.data.root_pos_w[:, 1]
    target_z = 0.7 # This is a desired absolute height, which is allowed for Z-axis for stable standing.

    # Calculate distance components using relative distances.
    # Rewards are based on relative distances between robot pelvis and the calculated target.
    distance_x = target_x - pelvis_pos[:, 0]
    distance_y = target_y - pelvis_pos[:, 1]
    distance_z = target_z - pelvis_pos[:, 2]

    # Reward is negative absolute distance for each component, promoting continuous movement towards the target.
    # This ensures a continuous reward function.
    reward = -torch.abs(distance_x) - torch.abs(distance_y) - torch.abs(distance_z)

    # Mandatory reward normalization as per rules.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def feet_on_ground_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "feet_on_ground_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to keep its feet on the ground, promoting stability.
    It penalizes the vertical distance of the feet from the ground (z=0).
    """
    # Get normalizer instance as per mandatory normalization rule.
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot parts using approved patterns.
    robot = env.scene["robot"] # Accessing robot using approved pattern.
    left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Accessing robot part index using approved pattern.
    right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Accessing robot part index using approved pattern.

    # Get z-positions of feet. Z-position is the only absolute position allowed for ground contact.
    left_foot_pos_z = robot.data.body_pos_w[:, left_foot_idx, 2] # Accessing robot part position using approved pattern.
    right_foot_pos_z = robot.data.body_pos_w[:, right_foot_idx, 2] # Accessing robot part position using approved pattern.

    # Reward is negative absolute distance of feet from ground (z=0).
    # This is a continuous reward based on the absolute z-position, which is allowed for ground contact.
    reward = -torch.abs(left_foot_pos_z) - torch.abs(right_foot_pos_z)

    # Mandatory reward normalization as per rules.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def prevent_overshoot_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "prevent_overshoot_reward") -> torch.Tensor:
    """
    Shaping reward to prevent the robot from overshooting Object2.
    It penalizes the robot if its pelvis moves significantly past the front face of Object2.
    """
    # Get normalizer instance as per mandatory normalization rule.
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns.
    robot = env.scene["robot"] # Accessing robot using approved pattern.
    object2 = env.scene['Object2'] # Medium Block - Accessing object using approved pattern.
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern.
    pelvis_pos_x = robot.data.body_pos_w[:, pelvis_idx, 0] # Accessing robot part position using approved pattern.

    # Object2 dimensions (hardcoded from object configuration as per rules).
    # From object configuration: Medium Block measuring x=1m y=1m and z=0.6m.
    object2_size_x = 1.0 # Medium Block x dimension

    # Define the "past" threshold: front face of Object2 + a small buffer (0.1m).
    # This threshold is calculated relative to Object2's position, adhering to the relative distance rule.
    overshoot_threshold_x = object2.data.root_pos_w[:, 0] + (object2_size_x / 2.0) + 0.1

    # Calculate the distance past the threshold (relative distance).
    distance_past_threshold = pelvis_pos_x - overshoot_threshold_x

    # Condition: pelvis_pos_x is greater than the overshoot_threshold_x.
    activation_condition = (pelvis_pos_x > overshoot_threshold_x)

    # Reward is negative for being past the threshold, proportional to how far past it is.
    # Only apply penalty if past the threshold, otherwise reward is 0.0.
    # This uses torch.where for a continuous reward that activates only when the condition is met.
    reward = torch.where(activation_condition, -torch.abs(distance_past_threshold), torch.tensor(0.0, device=env.device))

    # Mandatory reward normalization as per rules.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Primary reward for reaching the target position near the Medium Block.
    # Weight set to 1.0 as per prompt guidelines for primary rewards.
    WalkToMediumBlockReward = RewTerm(func=walk_to_medium_block_reward, weight=1.0,
                                      params={"normalise": True, "normaliser_name": "walk_to_medium_block_reward"})

    # Shaping reward for keeping feet on the ground to promote stability.
    # Weight set to 0.4 as per prompt guidelines for supporting rewards (<1.0).
    FeetOnGroundReward = RewTerm(func=feet_on_ground_reward, weight=0.4,
                                 params={"normalise": True, "normaliser_name": "feet_on_ground_reward"})

    # Shaping reward to prevent the robot from overshooting the Medium Block.
    # Weight set to 0.3 as per prompt guidelines for supporting rewards (<1.0).
    PreventOvershootReward = RewTerm(func=prevent_overshoot_reward, weight=0.3,
                                     params={"normalise": True, "normaliser_name": "prevent_overshoot_reward"})