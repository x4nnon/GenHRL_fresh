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


def position_medium_block_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for positioning the Medium Block (Object2) relative to the Small Block (Object1).
    Encourages Object2 to be placed at a specific offset (1m in X, 1m in Y) from Object1's center,
    with its base at a Z-height of 0.3m (implying it's placed on top of Object1, or at the height of Object1's top surface).
    The reward is a negative sum of absolute distances in X, Y, and Z from the target.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object1 = env.scene['Object1'] # Small Block for robot interaction
    object2 = env.scene['Object2'] # Medium Block for robot interaction

    # Access object positions using approved patterns
    object1_pos = object1.data.root_pos_w
    object2_pos = object2.data.root_pos_w

    # Define target offsets for Object2 relative to Object1.
    # These values are hardcoded based on the block dimensions (1m x 1m) and desired stair configuration.
    # Object1 (Small Block) dimensions: 1m x 1m x 0.3m (height)
    # Object2 (Medium Block) dimensions: 1m x 1m x 0.6m (height)
    # The goal is for Object2's center to be 1m in X and 1m in Y from Object1's center.
    # The Z target for Object2's base is 0.3m (which is the top surface height of Object1 if Object1 is on the ground).
    # Object's root_pos_w is its center. So, for Object2's base to be at Z=0.3, its center Z should be 0.3 + (Object2_height / 2).
    object2_height = 0.6 # Hardcoded from object configuration as per rule 8.
    target_offset_x = 1.0 # Hardcoded based on task description and stair configuration.
    target_offset_y = 1.0 # Hardcoded based on task description and stair configuration.
    target_z_object2_center = 0.3 + (object2_height / 2.0) # Target Z for Object2's center, derived from Object1's height and Object2's half-height.

    # Calculate the target position for Object2 relative to Object1
    # This ensures the reward is based on relative distances, not hard-coded world coordinates.
    target_object2_pos_x = object1_pos[:, 0] + target_offset_x
    target_object2_pos_y = object1_pos[:, 1] + target_offset_y
    target_object2_pos_z = target_z_object2_center # Z target is absolute for the base, but relative to ground.

    # Calculate the distance components between Object2's current position and its target position
    # Using absolute differences for each dimension to create a continuous negative reward.
    distance_x = torch.abs(object2_pos[:, 0] - target_object2_pos_x)
    distance_y = torch.abs(object2_pos[:, 1] - target_object2_pos_y)
    distance_z = torch.abs(object2_pos[:, 2] - target_object2_pos_z)

    # Reward is negative absolute distance, so closer is higher reward.
    # This creates a smooth, continuous reward landscape.
    reward = -distance_x - distance_y - distance_z

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def pelvis_to_medium_block_shaping_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward_1") -> torch.Tensor:
    """
    Shaping reward to encourage the robot's pelvis to get close to the Medium Block (Object2)
    and position itself on the "pushing side" relative to the target direction.
    The reward is active only when the pelvis is within a certain XY range of Object2 and behind it
    (assuming pushing from negative X and negative Y towards positive X and Y offsets).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required object using approved patterns
    object2 = env.scene['Object2'] # Medium Block for robot interaction
    object2_pos = object2.data.root_pos_w

    # Access the required robot part (pelvis) using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Calculate distance components from pelvis to Object2
    # Using relative distances as required by rule 1.
    dist_pelvis_obj2_x = object2_pos[:, 0] - pelvis_pos[:, 0]
    dist_pelvis_obj2_y = object2_pos[:, 1] - pelvis_pos[:, 1]
    dist_pelvis_obj2_z = object2_pos[:, 2] - pelvis_pos[:, 2]

    # Define activation conditions for the shaping reward.
    # Condition 1: Pelvis is within 1.5m of Object2 in XY plane. This is an arbitrary threshold as per prompt's reward design plan.
    # Condition 2: Pelvis is positioned to facilitate pushing towards positive X and Y.
    # This means pelvis_x should be less than object2_x, and pelvis_y should be less than object2_y.
    # This encourages pushing from the "origin side" of the block.
    activation_condition = (torch.abs(dist_pelvis_obj2_x) < 1.5) & \
                           (torch.abs(dist_pelvis_obj2_y) < 1.5) & \
                           (pelvis_pos[:, 0] < object2_pos[:, 0]) & \
                           (pelvis_pos[:, 1] < object2_pos[:, 1])

    # Reward for being close to Object2 (negative absolute distance in XY).
    # This is a continuous reward that gets higher as the robot gets closer, as per rule 7.
    reward_component = -torch.abs(dist_pelvis_obj2_x) - torch.abs(dist_pelvis_obj2_y)

    # Apply the activation condition: reward is 0 if condition is not met.
    # This uses torch.where for batched conditional application.
    reward = torch.where(activation_condition, reward_component, torch.tensor(0.0, device=env.device))

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
    """
    Configuration class for the rewards used in the 'position_medium_block' skill.
    Defines the main reward and supporting shaping rewards with their respective weights.
    """
    # Main reward for positioning the Medium Block (Object2) relative to the Small Block (Object1).
    # Weight is 1.0 as it's the primary objective, as per rule 2 in reward structure rules.
    PositionMediumBlockMainReward = RewTerm(func=position_medium_block_main_reward, weight=1.0,
                                            params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward to encourage the robot to approach Object2 from the correct side for pushing.
    # Weight is 0.6 to provide significant guidance without overshadowing the main goal, as per rule 2 in reward structure rules.
    PelvisToMediumBlockShapingReward = RewTerm(func=pelvis_to_medium_block_shaping_reward, weight=0.6,
                                               params={"normalise": True, "normaliser_name": "shaping_reward_1"})