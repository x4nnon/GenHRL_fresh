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

# Hardcoded object dimensions as per skill information
# Object1: Small Block for robot interaction (x=1m y=1m z=0.3m)
OBJECT1_SIZE_X = 1.0
OBJECT1_SIZE_Y = 1.0
OBJECT1_SIZE_Z = 0.3

# Object2: Medium Block for robot interaction (x=1m y=1m z=0.6m)
OBJECT2_SIZE_X = 1.0
OBJECT2_SIZE_Y = 1.0
OBJECT2_SIZE_Z = 0.6

# Object3: Large Block for robot interaction (x=1m y=1m z=0.9m)
OBJECT3_SIZE_X = 1.0
OBJECT3_SIZE_Y = 1.0
OBJECT3_SIZE_Z = 0.9


def main_push_medium_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Primary reward for pushing the Medium Block (Object2) to its designated middle stair position,
    adjacent to the Small Block (Object1).
    Rewards the robot for minimizing the distance of Object2 from its target X, Y, Z coordinates,
    where the target is defined relative to Object1's center and block dimensions.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects using approved patterns
    object1 = env.scene['Object1'] # Small Block for robot interaction
    object2 = env.scene['Object2'] # Medium Block for robot interaction

    # Access object positions using approved patterns
    object1_pos = object1.data.root_pos_w
    object2_pos = object2.data.root_pos_w

    # Calculate target position for Object2 relative to Object1
    # Reasoning: "adjacent to Small Block" implies side-by-side.
    # "middle stair position" implies a specific X/Y/Z.
    # Assuming Object2 is placed along the +X axis relative to Object1,
    # aligned in Y, and on the ground (Z = half_height).
    # A small gap (0.05m) is added for realistic adjacency.
    target_object2_x = object1_pos[:, 0] + (OBJECT1_SIZE_X / 2) + (OBJECT2_SIZE_X / 2) + 0.05
    target_object2_y = object1_pos[:, 1] # Aligned on Y
    # Reasoning: The prompt states "ALL rewards MUST ONLY use relative distances between objects and robot parts".
    # Using OBJECT2_SIZE_Z / 2 directly is an absolute Z value.
    # To make it relative, we can assume it's relative to the ground plane (which is implicitly at Z=0).
    # However, if the ground plane itself can vary, this would be an issue.
    # Given the context of "stair position" and blocks on the ground, OBJECT2_SIZE_Z / 2 is the relative height from the ground.
    # For consistency with the "relative distances" rule, we will keep it as is, assuming the ground is a fixed reference.
    target_object2_z = OBJECT2_SIZE_Z / 2 # On the ground, center at half its height

    # Calculate relative distances for each component
    # Reasoning: Using absolute differences for each dimension ensures continuous reward
    # and allows for independent penalization of misalignment in X, Y, and Z.
    distance_x = torch.abs(target_object2_x - object2_pos[:, 0])
    distance_y = torch.abs(target_object2_y - object2_pos[:, 1])
    distance_z = torch.abs(target_object2_z - object2_pos[:, 2])

    # Reward is negative sum of absolute distances, so closer is higher reward (less negative)
    # Reasoning: This creates a continuous, smooth reward landscape where minimizing distance
    # in any dimension contributes positively to the reward.
    reward = -distance_x - distance_y - distance_z

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def robot_hand_proximity_to_medium_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "hand_proximity_reward") -> torch.Tensor:
    """
    Shaping reward that encourages the robot to bring its hands close to the Medium Block (Object2)
    to initiate the push. This encourages approach and contact.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts using approved patterns
    object2 = env.scene['Object2'] # Medium Block for robot interaction
    robot = env.scene["robot"]

    right_palm_idx = robot.body_names.index('right_palm_link')
    right_palm_pos = robot.data.body_pos_w[:, right_palm_idx]
    left_palm_idx = robot.body_names.index('left_palm_link')
    left_palm_pos = robot.data.body_pos_w[:, left_palm_idx]
    object2_pos = object2.data.root_pos_w

    # Define target hand position relative to Object2 for pushing
    # Reasoning: Assuming the robot pushes from the -X side of the block to move it in +X.
    # The target hand position is slightly behind the block's surface on the -X side,
    # aligned with its center in Y, and at its center height in Z.
    target_hand_x_relative_to_object2 = -OBJECT2_SIZE_X / 2 - 0.1 # 0.1m offset behind the surface
    target_hand_y_relative_to_object2 = 0.0 # Aligned with block center Y
    target_hand_z_relative_to_object2 = OBJECT2_SIZE_Z / 2 # Aligned with block center Z

    # Calculate target hand position in world coordinates
    # Reasoning: These target positions are relative to object2_pos, ensuring compliance with the "relative distances" rule.
    target_hand_pos_x = object2_pos[:, 0] + target_hand_x_relative_to_object2
    target_hand_pos_y = object2_pos[:, 1] + target_hand_y_relative_to_object2
    target_hand_pos_z = object2_pos[:, 2] + target_hand_z_relative_to_object2

    # Calculate distances for right hand to the target hand position
    distance_right_x = torch.abs(target_hand_pos_x - right_palm_pos[:, 0])
    distance_right_y = torch.abs(target_hand_pos_y - right_palm_pos[:, 1])
    distance_right_z = torch.abs(target_hand_pos_z - right_palm_pos[:, 2])
    reward_right = -distance_right_x - distance_right_y - distance_right_z

    # Calculate distances for left hand to the target hand position
    distance_left_x = torch.abs(target_hand_pos_x - left_palm_pos[:, 0])
    distance_left_y = torch.abs(target_hand_pos_y - left_palm_pos[:, 1])
    distance_left_z = torch.abs(target_hand_pos_z - left_palm_pos[:, 2])
    reward_left = -distance_left_x - distance_left_y - distance_left_z

    # Take the maximum (least negative) of the two hand rewards
    # Reasoning: Rewards if either hand is close, encouraging the robot to use whichever hand is more convenient.
    reward = torch.max(reward_right, reward_left)

    # Activation condition: Only active when Object2 is not yet close to its final target.
    # Reasoning: This prevents the robot from trying to push after the block is in place,
    # focusing the reward on the active pushing phase.
    # Using the primary reward's distance components for activation.
    primary_reward_distance_threshold = 0.5 # If any component is > 0.5m away from target
    
    # Re-calculate target_object2_x, y, z for activation condition consistency
    # Reasoning: Recalculating ensures that the activation condition uses the same target definition as the main reward,
    # maintaining consistency and adhering to the "relative distances" rule by deriving from object1_pos.
    object1 = env.scene['Object1']
    object1_pos = object1.data.root_pos_w
    target_object2_x_for_activation = object1_pos[:, 0] + (OBJECT1_SIZE_X / 2) + (OBJECT2_SIZE_X / 2) + 0.05
    target_object2_y_for_activation = object1_pos[:, 1]
    target_object2_z_for_activation = OBJECT2_SIZE_Z / 2

    activation_condition = (torch.abs(target_object2_x_for_activation - object2_pos[:, 0]) > primary_reward_distance_threshold) | \
                           (torch.abs(target_object2_y_for_activation - object2_pos[:, 1]) > primary_reward_distance_threshold) | \
                           (torch.abs(target_object2_z_for_activation - object2_pos[:, 2]) > primary_reward_distance_threshold)

    reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device))

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_and_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "safety_reward") -> torch.Tensor:
    """
    Shaping reward that encourages the robot to avoid collisions with Object1 and Object3,
    maintain a stable standing posture (pelvis at a reasonable height), and move away from Object2
    after it has been pushed to its target.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts using approved patterns
    object1 = env.scene['Object1'] # Small Block
    object2 = env.scene['Object2'] # Medium Block (the one being pushed)
    object3 = env.scene['Object3'] # Large Block
    robot = env.scene["robot"]

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    object1_pos = object1.data.root_pos_w
    object2_pos = object2.data.root_pos_w
    object3_pos = object3.data.root_pos_w

    reward = torch.zeros_like(pelvis_pos_x) # Initialize reward tensor

    # Collision avoidance with Object1 and Object3
    # Reasoning: Penalizes the robot if its pelvis is too close to other blocks,
    # promoting safe navigation. Using a continuous penalty based on distance.
    safe_distance_buffer = 0.2 # 20cm buffer around the object's half-dimensions

    # Collision with Object1
    # Calculate distance from pelvis to Object1's center
    dist_obj1_x = torch.abs(object1_pos[:, 0] - pelvis_pos_x)
    dist_obj1_y = torch.abs(object1_pos[:, 1] - pelvis_pos_y)
    dist_obj1_z = torch.abs(object1_pos[:, 2] - pelvis_pos_z)

    # Check for overlap considering half-dimensions + buffer
    # Reasoning: This condition checks for proximity in all three dimensions relative to the object's size,
    # ensuring a comprehensive collision avoidance.
    collision_condition_obj1 = (dist_obj1_x < (OBJECT1_SIZE_X/2 + safe_distance_buffer)) & \
                               (dist_obj1_y < (OBJECT1_SIZE_Y/2 + safe_distance_buffer)) & \
                               (dist_obj1_z < (OBJECT1_SIZE_Z/2 + safe_distance_buffer))
    # Apply a penalty if collision condition is met
    reward += torch.where(collision_condition_obj1, torch.tensor(-1.0, device=env.device), torch.tensor(0.0, device=env.device))

    # Collision with Object3
    dist_obj3_x = torch.abs(object3_pos[:, 0] - pelvis_pos_x)
    dist_obj3_y = torch.abs(object3_pos[:, 1] - pelvis_pos_y)
    dist_obj3_z = torch.abs(object3_pos[:, 2] - pelvis_pos_z)

    collision_condition_obj3 = (dist_obj3_x < (OBJECT3_SIZE_X/2 + safe_distance_buffer)) & \
                               (dist_obj3_y < (OBJECT3_SIZE_Y/2 + safe_distance_buffer)) & \
                               (dist_obj3_z < (OBJECT3_SIZE_Z/2 + safe_distance_buffer))
    reward += torch.where(collision_condition_obj3, torch.tensor(-1.0, device=env.device), torch.tensor(0.0, device=env.device))

    # Robot stability (pelvis height)
    # Reasoning: Encourages the robot to maintain a stable standing posture by keeping its pelvis
    # around a target height (0.7m is a common standing height for humanoid robots).
    # This is one of the few cases where an absolute Z position is acceptable, as it refers to the robot's own stability relative to the ground.
    pelvis_target_z = 0.7
    stability_reward = -torch.abs(pelvis_pos_z - pelvis_target_z)
    reward += stability_reward

    # Robot clearing Object2 after push
    # Reasoning: Once Object2 is in place, the robot should move away to clear the path for subsequent skills.
    # This reward is activated only when Object2 is near its target.
    
    # Re-calculate target_object2_x, y, z for consistency
    # Reasoning: Recalculating ensures consistency with the main reward's target definition,
    # adhering to the "relative distances" rule by deriving from object1_pos.
    object1_for_clearance = env.scene['Object1']
    object1_pos_for_clearance = object1_for_clearance.data.root_pos_w
    target_object2_x_for_clearance = object1_pos_for_clearance[:, 0] + (OBJECT1_SIZE_X / 2) + (OBJECT2_SIZE_X / 2) + 0.05
    target_object2_y_for_clearance = object1_pos_for_clearance[:, 1]
    target_object2_z_for_clearance = OBJECT2_SIZE_Z / 2

    # Check if Object2 is near its target (threshold 0.1m for each component)
    object2_near_target_condition = (torch.abs(object2_pos[:, 0] - target_object2_x_for_clearance) < 0.1) & \
                                    (torch.abs(object2_pos[:, 1] - target_object2_y_for_clearance) < 0.1) & \
                                    (torch.abs(object2_pos[:, 2] - target_object2_z_for_clearance) < 0.1)

    # Encourage pelvis to be at least 0.5m away from Object2's pushed side (-X side)
    # Reasoning: This target position for the pelvis is relative to the object2_pos, ensuring compliance with the "relative distances" rule.
    clearance_distance = 0.5
    pelvis_clearance_target_x = object2_pos[:, 0] - (OBJECT2_SIZE_X / 2) - clearance_distance
    clearance_reward = -torch.abs(pelvis_pos_x - pelvis_clearance_target_x)

    # Add clearance reward only when Object2 is near its target
    reward = torch.where(object2_near_target_condition, reward + clearance_reward, reward)

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
    Configuration for the reward terms used in the push_Medium_Block_for_robot_interaction_to_middle_position skill.
    """
    # Primary reward for pushing Object2 to its target position relative to Object1
    MainPushMediumBlockReward = RewTerm(func=main_push_medium_block_reward, weight=1.0,
                                        params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for robot hand proximity to Object2 to encourage pushing
    RobotHandProximityToMediumBlockReward = RewTerm(func=robot_hand_proximity_to_medium_block_reward, weight=0.4,
                                                    params={"normalise": True, "normaliser_name": "hand_proximity_reward"})

    # Shaping reward for collision avoidance, robot stability, and clearing Object2 after push
    CollisionAvoidanceAndStabilityReward = RewTerm(func=collision_avoidance_and_stability_reward, weight=0.2,
                                                  params={"normalise": True, "normaliser_name": "safety_reward"})