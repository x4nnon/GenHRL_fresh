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

# Hardcoded object dimensions from the object configuration
# Object1: Small Block (x=1m y=1m z=0.3m)
OBJECT1_HEIGHT = 0.3
OBJECT1_HALF_HEIGHT = OBJECT1_HEIGHT / 2.0
OBJECT1_HALF_WIDTH_Y = 0.5 # Assuming 1m width, so 0.5m half-width

# Object2: Medium Block (x=1m y=1m z=0.6m)
OBJECT2_HEIGHT = 0.6
OBJECT2_HALF_HEIGHT = OBJECT2_HEIGHT / 2.0
OBJECT2_HALF_WIDTH_Y = 0.5 # Assuming 1m width, so 0.5m half-width

def push_object1_to_target_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "primary_object_placement_reward") -> torch.Tensor:
    """
    Primary reward for guiding Object1 (Small Block) to its target position relative to Object2 (Medium Block).
    The target position for Object1 is adjacent to Object2, with its top surface aligned with Object2's top surface.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access objects using approved patterns
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']

    # Get current positions of objects
    object1_pos = object1.data.root_pos_w
    object2_pos = object2.data.root_pos_w

    # Calculate target position for Object1 relative to Object2
    # Object1 should be adjacent to Object2 along one axis (e.g., Y-axis)
    # Object1's center Z should be such that its top is at Object2's top height
    # All positions are relative to Object2's position.
    # This uses relative distances by defining the target based on Object2's current position and object dimensions.
    target_object1_pos_x = object2_pos[:, 0] # Align X-coordinates
    target_object1_pos_y = object2_pos[:, 1] + OBJECT2_HALF_WIDTH_Y + OBJECT1_HALF_WIDTH_Y # Place Object1 adjacent to Object2 along Y
    target_object1_pos_z = object2_pos[:, 2] + OBJECT2_HALF_HEIGHT - OBJECT1_HALF_HEIGHT # Align top surfaces

    # Calculate distances between Object1's current position and its target relative to Object2
    # Using absolute differences for each dimension to create a continuous negative reward
    # This ensures the reward is based on relative distances.
    dist_x = torch.abs(object1_pos[:, 0] - target_object1_pos_x)
    dist_y = torch.abs(object1_pos[:, 1] - target_object1_pos_y)
    dist_z = torch.abs(object1_pos[:, 2] - target_object1_pos_z)

    # The reward is maximized (closer to 0) when Object1 is at the target position
    # This is a continuous reward.
    reward = -dist_x - dist_y - dist_z

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def robot_hand_to_object1_proximity_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_hand_proximity_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot's hands to be close to Object1, facilitating the push.
    This reward is active when Object1 is still far from its final target position.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot and object using approved patterns
    robot = env.scene["robot"]
    object1 = env.scene['Object1']
    object2 = env.scene['Object2'] # Needed for activation condition

    # Get indices and positions for robot hands using approved patterns
    right_hand_idx = robot.body_names.index('right_palm_link')
    left_hand_idx = robot.body_names.index('left_palm_link')
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]
    object1_pos = object1.data.root_pos_w

    # Calculate distance from both hands to Object1
    # Using absolute differences for each dimension, ensuring relative distances.
    dist_right_hand_x = torch.abs(object1_pos[:, 0] - right_hand_pos[:, 0])
    dist_right_hand_y = torch.abs(object1_pos[:, 1] - right_hand_pos[:, 1])
    dist_right_hand_z = torch.abs(object1_pos[:, 2] - right_hand_pos[:, 2])

    dist_left_hand_x = torch.abs(object1_pos[:, 0] - left_hand_pos[:, 0])
    dist_left_hand_y = torch.abs(object1_pos[:, 1] - left_hand_pos[:, 1])
    dist_left_hand_z = torch.abs(object1_pos[:, 2] - left_hand_pos[:, 2])

    # Choose the closer hand for the reward by taking the minimum absolute distance for each dimension
    min_dist_x = torch.min(dist_right_hand_x, dist_left_hand_x)
    min_dist_y = torch.min(dist_right_hand_y, dist_left_hand_y)
    min_dist_z = torch.min(dist_right_hand_z, dist_left_hand_z)

    # Calculate Object1's target position for the activation condition, based on relative positions.
    target_object1_pos_x = object2.data.root_pos_w[:, 0]
    target_object1_pos_y = object2.data.root_pos_w[:, 1] + OBJECT2_HALF_WIDTH_Y + OBJECT1_HALF_WIDTH_Y
    target_object1_pos_z = object2.data.root_pos_w[:, 2] + OBJECT2_HALF_HEIGHT - OBJECT1_HALF_HEIGHT

    # Calculate current Euclidean distance of Object1 to its target
    # This is a relative distance calculation.
    current_object1_dist_to_target = torch.sqrt(
        (object1_pos[:, 0] - target_object1_pos_x)**2 +
        (object1_pos[:, 1] - target_object1_pos_y)**2 +
        (object1_pos[:, 2] - target_object1_pos_z)**2
    )

    # Activation condition: Reward is active if Object1 is further than 0.5m from its target
    # This threshold is a relative distance, not an absolute position.
    object1_target_dist_threshold = 0.5
    activation_condition = (current_object1_dist_to_target > object1_target_dist_threshold)

    # Reward is negative sum of minimum distances, encouraging hands to be close
    # This is a continuous reward based on relative distances.
    reward = -min_dist_x - min_dist_y - min_dist_z
    # Apply activation condition: set reward to 0 if condition is not met
    reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device))

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def robot_post_push_stability_and_clearance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_post_push_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to maintain a stable standing posture and move away
    from the newly placed block after the push is complete.
    This reward is active when Object1 is very close to its final target position.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot and objects using approved patterns
    robot = env.scene["robot"]
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']

    # Get pelvis position using approved patterns
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Get object positions
    object1_pos = object1.data.root_pos_w
    object2_pos = object2.data.root_pos_w

    # Define target stable pelvis height. This is a fixed value for a stable standing posture.
    pelvis_target_z = 0.7 # Standard stable standing height

    # Calculate Object1's target position for the activation condition, based on relative positions.
    target_object1_pos_x = object2_pos[:, 0]
    target_object1_pos_y = object2_pos[:, 1] + OBJECT2_HALF_WIDTH_Y + OBJECT1_HALF_WIDTH_Y
    target_object1_pos_z = object2_pos[:, 2] + OBJECT2_HALF_HEIGHT - OBJECT1_HALF_HEIGHT

    # Calculate current Euclidean distance of Object1 to its target
    # This is a relative distance calculation.
    current_object1_dist_to_target = torch.sqrt(
        (object1_pos[:, 0] - target_object1_pos_x)**2 +
        (object1_pos[:, 1] - target_object1_pos_y)**2 +
        (object1_pos[:, 2] - target_object1_pos_z)**2
    )

    # Activation condition: Reward is active if Object1 is closer than 0.2m to its target
    # This threshold is a relative distance.
    object1_target_dist_threshold = 0.2
    activation_condition = (current_object1_dist_to_target < object1_target_dist_threshold)

    # Reward for stable pelvis height: negative absolute difference from target Z
    # This is a continuous reward based on the pelvis's Z-position relative to a target height.
    reward_pelvis_z = -torch.abs(pelvis_pos_z - pelvis_target_z)

    # Reward for moving away from the combined Object1/Object2 structure
    # Approximate center of the combined structure for clearance calculation, based on relative positions.
    combined_center_x = (object1_pos[:, 0] + object2_pos[:, 0]) / 2.0
    combined_center_y = (object1_pos[:, 1] + object2_pos[:, 1]) / 2.0

    # Calculate 2D distance from pelvis to the combined structure's center
    # This is a relative distance calculation.
    dist_pelvis_to_structure_xy = torch.sqrt(
        (pelvis_pos_x - combined_center_x)**2 +
        (pelvis_pos_y - combined_center_y)**2
    )

    # Encourage moving away: positive reward for distance beyond a minimum safe distance
    # This is a continuous reward based on relative distance and a threshold.
    min_safe_distance = 1.0 # Robot should be at least 1m away from the structure
    # Use torch.max to ensure reward is non-negative for clearance, only rewarding distances beyond the threshold
    reward_clearance = torch.max(torch.tensor(0.0, device=env.device), dist_pelvis_to_structure_xy - min_safe_distance)

    # Combine pelvis stability and clearance rewards
    # This is a continuous reward.
    reward = reward_pelvis_z + reward_clearance
    # Apply activation condition: set reward to 0 if condition is not met
    reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device))

    # Mandatory normalization
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
    Reward terms for the push_Small_Block_into_stair_top_position skill.
    """
    # Primary reward for placing Object1 at its target relative to Object2
    ObjectPlacementReward = RewTerm(
        func=push_object1_to_target_reward,
        weight=1.0, # High weight as this is the main goal
        params={"normalise": True, "normaliser_name": "primary_object_placement_reward"}
    )

    # Shaping reward for robot hands to be close to Object1 during the push phase
    HandProximityReward = RewTerm(
        func=robot_hand_to_object1_proximity_reward,
        weight=0.4, # Moderate weight to guide interaction
        params={"normalise": True, "normaliser_name": "shaping_hand_proximity_reward"}
    )

    # Shaping reward for robot stability and moving away after the push is complete
    PostPushStabilityAndClearanceReward = RewTerm(
        func=robot_post_push_stability_and_clearance_reward,
        weight=0.6, # Moderate weight to encourage a good final state
        params={"normalise": True, "normaliser_name": "shaping_post_push_reward"}
    )