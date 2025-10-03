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


def main_push_block_onto_platform_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for pushing the 'third 0.5m cubed block' (Object3) entirely onto the 'platform' (Object4).
    This reward combines terms for centering, correct height, and being within boundaries.
    """
    # Get normalizer instance as per mandatory normalization rule.
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns (env.scene['ObjectName']).
    object3 = env.scene['Object3'] # third 0.5m cubed block
    object4 = env.scene['Object4'] # platform

    # Access object positions using approved patterns (object.data.root_pos_w).
    object3_pos = object3.data.root_pos_w
    object4_pos = object4.data.root_pos_w

    # Object dimensions (hardcoded from configuration as per rules, as there is no way to access object size dynamically).
    # Object3 dimensions (0.5m cubed block)
    object3_half_size_x = 0.25
    object3_half_size_y = 0.25
    object3_half_size_z = 0.25

    # Object4 dimensions (platform: x=2m y=2m, z=0.001)
    object4_half_size_x = 1.0
    object4_half_size_y = 1.0
    object4_half_size_z = 0.0005 # half of 0.001m height

    # 1. Reward for Object3 being centered on Object4 (x, y)
    # This encourages the block to move towards the center of the platform.
    # Using relative distances as per rule 1.
    dist_x_obj3_obj4 = object3_pos[:, 0] - object4_pos[:, 0]
    dist_y_obj3_obj4 = object3_pos[:, 1] - object4_pos[:, 1]
    # Using negative absolute distance for continuous reward, closer to zero is better.
    center_reward = -torch.abs(dist_x_obj3_obj4) - torch.abs(dist_y_obj3_obj4)

    # 2. Reward for Object3 being at the correct height relative to Object4
    # Object3's bottom surface should be at Object4's top surface.
    # Calculate the target Z position for Object3's center using relative positions and hardcoded dimensions.
    target_z_obj3 = object4_pos[:, 2] + object4_half_size_z + object3_half_size_z
    # Using negative absolute distance for continuous reward, closer to target_z_obj3 is better.
    height_reward = -torch.abs(object3_pos[:, 2] - target_z_obj3)

    # 3. Penalty for Object3 being outside Object4's boundaries (x, y)
    # This ensures the block is fully contained within the platform's surface.
    # Calculate the min/max edges for Object4 using its position and hardcoded dimensions.
    x_min_obj4 = object4_pos[:, 0] - object4_half_size_x
    x_max_obj4 = object4_pos[:, 0] + object4_half_size_x
    y_min_obj4 = object4_pos[:, 1] - object4_half_size_y
    y_max_obj4 = object4_pos[:, 1] + object4_half_size_y

    # Calculate the min/max edges for Object3 using its position and hardcoded dimensions.
    x_min_obj3 = object3_pos[:, 0] - object3_half_size_x
    x_max_obj3 = object3_pos[:, 0] + object3_half_size_x
    y_min_obj3 = object3_pos[:, 1] - object3_half_size_y
    y_max_obj3 = object3_pos[:, 1] + object3_half_size_y

    # Calculate overlap for continuous penalty.
    # Overlap is the length of the intersection of the two intervals.
    # Ensure tensor operations work with batched environments.
    overlap_x = torch.max(torch.tensor(0.0, device=env.device), torch.min(x_max_obj3, x_max_obj4) - torch.max(x_min_obj3, x_min_obj4))
    overlap_y = torch.max(torch.tensor(0.0, device=env.device), torch.min(y_max_obj3, y_max_obj4) - torch.max(y_min_obj3, y_min_obj4))

    # Penalty is higher if overlap is less than the full object size.
    # The penalty is the sum of the non-overlapping parts.
    boundary_penalty = - (object3_half_size_x * 2 - overlap_x) - (object3_half_size_y * 2 - overlap_y)
    # Ensure penalty is non-positive (0 if fully on, negative if off)
    boundary_penalty = torch.min(boundary_penalty, torch.tensor(0.0, device=env.device))

    # Combine all reward components. Rewards are continuous.
    reward = center_reward + height_reward + boundary_penalty

    # Mandatory reward normalization as per rule 2.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def robot_hand_proximity_to_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "hand_proximity_reward") -> torch.Tensor:
    """
    Shaping reward that encourages the robot's hands to be close to the 'third 0.5m cubed block' (Object3)
    when Object3 is not yet fully on the platform. This promotes contact and pushing action.
    """
    # Get normalizer instance as per mandatory normalization rule.
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts using approved patterns.
    robot = env.scene["robot"]
    object3 = env.scene['Object3'] # third 0.5m cubed block
    object4 = env.scene['Object4'] # platform

    # Access robot part positions using approved patterns.
    right_hand_idx = robot.body_names.index('right_palm_link')
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]
    left_hand_idx = robot.body_names.index('left_palm_link')
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]

    object3_pos = object3.data.root_pos_w
    object4_pos = object4.data.root_pos_w

    # Object dimensions (hardcoded from configuration as per rules).
    object3_half_size_x = 0.25
    object3_half_size_y = 0.25
    object3_half_size_z = 0.25
    object4_half_size_x = 1.0
    object4_half_size_y = 1.0
    object4_half_size_z = 0.0005

    # Check if Object3 is on Object4 (approximate condition for activation).
    # This condition determines when the shaping reward should be active.
    # Using relative distances and hardcoded dimensions.
    target_z_obj3 = object4_pos[:, 2] + object4_half_size_z + object3_half_size_z
    is_on_platform_z = torch.abs(object3_pos[:, 2] - target_z_obj3) < 0.05 # within 5cm height tolerance

    x_min_obj4 = object4_pos[:, 0] - object4_half_size_x
    x_max_obj4 = object4_pos[:, 0] + object4_half_size_x
    y_min_obj4 = object4_pos[:, 1] - object4_half_size_y
    y_max_obj4 = object4_pos[:, 1] + object4_half_size_y

    x_min_obj3 = object3_pos[:, 0] - object3_half_size_x
    x_max_obj3 = object3_pos[:, 0] + object3_half_size_x
    y_min_obj3 = object3_pos[:, 1] - object3_half_size_y
    y_max_obj3 = object3_pos[:, 1] + object3_half_size_y

    # Check if Object3 is fully within Object4's x,y boundaries.
    # Ensure tensor operations work with batched environments.
    is_on_platform_x = (x_min_obj3 >= x_min_obj4) & (x_max_obj3 <= x_max_obj4)
    is_on_platform_y = (y_min_obj3 >= y_min_obj4) & (y_max_obj3 <= y_max_obj4)

    # Activation condition: Reward is active when Object3 is NOT fully on the platform.
    activation_condition = ~(is_on_platform_z & is_on_platform_x & is_on_platform_y)

    # Calculate distance from each hand to Object3's center.
    # Encourage hands to be slightly behind the block (negative x relative to block's center)
    # and at an appropriate height for pushing (e.g., slightly above the block's bottom).
    # Using relative distances and hardcoded offsets.
    target_hand_x_offset = -0.1 # 10cm behind the block's center in x (assuming push in positive x)
    target_hand_y_offset = 0.0 # Centered on block's y
    target_hand_z = object3_pos[:, 2] - object3_half_size_z + 0.1 # 10cm above block's bottom

    # Calculate rewards for each hand based on proximity to the target pushing point.
    # Rewards are continuous.
    reward_right_hand = -torch.abs(right_hand_pos[:, 0] - (object3_pos[:, 0] + target_hand_x_offset)) \
                        -torch.abs(right_hand_pos[:, 1] - (object3_pos[:, 1] + target_hand_y_offset)) \
                        -torch.abs(right_hand_pos[:, 2] - target_hand_z)

    reward_left_hand = -torch.abs(left_hand_pos[:, 0] - (object3_pos[:, 0] + target_hand_x_offset)) \
                       -torch.abs(left_hand_pos[:, 1] - (object3_pos[:, 1] + target_hand_y_offset)) \
                       -torch.abs(left_hand_pos[:, 2] - target_hand_z)

    # Take the maximum reward from either hand, as only one hand might be actively pushing.
    hand_proximity_reward = torch.max(reward_right_hand, reward_left_hand)

    # Apply the reward only when the activation condition is met, otherwise apply a small penalty.
    # Ensure tensor operations work with batched environments.
    reward = torch.where(activation_condition, hand_proximity_reward, torch.tensor(-0.1, device=env.device))

    # Mandatory reward normalization as per rule 2.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def robot_stability_after_push_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stability_reward") -> torch.Tensor:
    """
    Shaping reward that encourages the robot to maintain a stable, upright posture (pelvis at a reasonable height)
    and to not overshoot the platform after pushing Object3 onto it.
    This reward is active only when Object3 IS fully resting on Object4.
    """
    # Get normalizer instance as per mandatory normalization rule.
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts using approved patterns.
    robot = env.scene["robot"]
    object3 = env.scene['Object3'] # third 0.5m cubed block
    object4 = env.scene['Object4'] # platform

    # Access robot part positions using approved patterns.
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    object3_pos = object3.data.root_pos_w
    object4_pos = object4.data.root_pos_w

    # Object dimensions (hardcoded from configuration as per rules).
    object3_half_size_x = 0.25
    object3_half_size_y = 0.25
    object3_half_size_z = 0.25
    object4_half_size_x = 1.0
    object4_half_size_y = 1.0
    object4_half_size_z = 0.0005

    # Check if Object3 is on Object4 (condition for activation).
    # This condition determines when the shaping reward should be active.
    # Using relative distances and hardcoded dimensions.
    target_z_obj3 = object4_pos[:, 2] + object4_half_size_z + object3_half_size_z
    is_on_platform_z = torch.abs(object3_pos[:, 2] - target_z_obj3) < 0.05 # within 5cm height tolerance

    x_min_obj4 = object4_pos[:, 0] - object4_half_size_x
    x_max_obj4 = object4_pos[:, 0] + object4_half_size_x
    y_min_obj4 = object4_pos[:, 1] - object4_half_size_y
    y_max_obj4 = object4_pos[:, 1] + object4_half_size_y

    x_min_obj3 = object3_pos[:, 0] - object3_half_size_x
    x_max_obj3 = object3_pos[:, 0] + object3_half_size_x
    y_min_obj3 = object3_pos[:, 1] - object3_half_size_y
    y_max_obj3 = object3_pos[:, 1] + object3_half_size_y

    # Ensure tensor operations work with batched environments.
    is_on_platform_x = (x_min_obj3 >= x_min_obj4) & (x_max_obj3 <= x_max_obj4)
    is_on_platform_y = (y_min_obj3 >= y_min_obj4) & (y_max_obj3 <= y_max_obj4)

    # Activation condition: Reward is active when Object3 IS fully on the platform.
    activation_condition = (is_on_platform_z & is_on_platform_x & is_on_platform_y)

    # 1. Pelvis height reward for stability.
    # Encourages the robot's pelvis to be at a stable, upright height.
    # Using relative distance to a target height.
    target_pelvis_z = 0.7 # A common stable pelvis height for humanoid robots
    pelvis_height_reward = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # 2. Penalty for overshooting the platform.
    # Encourages the robot to stop near the platform after the push, not run past it.
    # The robot's pelvis x position relative to the platform's far edge.
    # A small buffer (e.g., 0.5m) past the platform is acceptable, but beyond that is penalized.
    # Using relative distance and hardcoded offset.
    overshoot_threshold_x = object4_pos[:, 0] + object4_half_size_x + 0.5 # 0.5m past platform edge
    # Penalize linearly if pelvis_pos_x is greater than the threshold.
    # Ensure tensor operations work with batched environments.
    overshoot_penalty = torch.where(pelvis_pos[:, 0] > overshoot_threshold_x, -(pelvis_pos[:, 0] - overshoot_threshold_x), torch.tensor(0.0, device=env.device))

    # Combine rewards for stability and position. Rewards are continuous.
    combined_reward = pelvis_height_reward + overshoot_penalty

    # Apply the reward only when the activation condition is met, otherwise apply a small penalty.
    # Ensure tensor operations work with batched environments.
    reward = torch.where(activation_condition, combined_reward, torch.tensor(-0.1, device=env.device))

    # Mandatory reward normalization as per rule 2.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_reward") -> torch.Tensor:
    """
    Shaping reward that penalizes collisions between the robot's body parts and any objects in the scene,
    especially other blocks (Object1, Object2) and the platform (Object4), but also the target block (Object3)
    if not intended for pushing contact (i.e., not hands).
    """
    # Get normalizer instance as per mandatory normalization rule.
    RewNormalizer = get_normalizer(env.device)

    # Access required objects using approved patterns.
    object1 = env.scene['Object1'] # first 0.5m cubed block
    object2 = env.scene['Object2'] # second 0.5m cubed block
    object3 = env.scene['Object3'] # third 0.5m cubed block
    object4 = env.scene['Object4'] # platform
    robot = env.scene["robot"]

    # Define robot parts to monitor for collisions (example: torso, legs, head).
    # These are parts that should generally avoid contact unless specifically for interaction.
    robot_parts_to_monitor = ['pelvis', 'left_knee_link', 'right_knee_link', 'head_link',
                              'left_ankle_roll_link', 'right_ankle_roll_link',
                              'left_upper_arm_link', 'right_upper_arm_link',
                              'left_lower_arm_link', 'right_lower_arm_link'] # Added arms for more comprehensive collision

    # Filter to only include parts that exist on this robot to avoid index errors
    existing_part_names = [name for name in robot_parts_to_monitor if name in robot.body_names]

    # Initialize collision penalty tensor for batched environments.
    collision_penalty = torch.zeros(env.num_envs, device=env.device)

    # Object dimensions (hardcoded from configuration as per rules).
    block_half_size = 0.25 # All 0.5m cubed blocks
    platform_half_height = 0.0005 # Platform is 0.001m height
    object4_half_size_x = 1.0
    object4_half_size_y = 1.0

    # Iterate through robot parts and objects to check for proximity/collision.
    for part_name in existing_part_names:
        part_idx = robot.body_names.index(part_name) # Access robot part index using approved pattern.
        part_pos = robot.data.body_pos_w[:, part_idx] # Access robot part position using approved pattern.

        # Check against Object1 (first block).
        # Penalize if robot part is too close to the block's center.
        # Using relative distances.
        dist_x_obj1 = torch.abs(part_pos[:, 0] - object1.data.root_pos_w[:, 0])
        dist_y_obj1 = torch.abs(part_pos[:, 1] - object1.data.root_pos_w[:, 1])
        dist_z_obj1 = torch.abs(part_pos[:, 2] - object1.data.root_pos_w[:, 2])
        # A simple proximity check: if any dimension is within a certain threshold of the block's half size + buffer.
        # This creates a continuous penalty based on how deep the penetration is.
        # Ensure tensor operations work with batched environments.
        proximity_threshold = block_half_size + 0.1 # 10cm buffer around the block
        penalty_x_obj1 = torch.max(torch.tensor(0.0, device=env.device), proximity_threshold - dist_x_obj1)
        penalty_y_obj1 = torch.max(torch.tensor(0.0, device=env.device), proximity_threshold - dist_y_obj1)
        penalty_z_obj1 = torch.max(torch.tensor(0.0, device=env.device), proximity_threshold - dist_z_obj1)
        collision_penalty -= (penalty_x_obj1 + penalty_y_obj1 + penalty_z_obj1) * 0.5 # Scale penalty

        # Check against Object2 (second block) - similar logic as Object1.
        dist_x_obj2 = torch.abs(part_pos[:, 0] - object2.data.root_pos_w[:, 0])
        dist_y_obj2 = torch.abs(part_pos[:, 1] - object2.data.root_pos_w[:, 1])
        dist_z_obj2 = torch.abs(part_pos[:, 2] - object2.data.root_pos_w[:, 2])
        penalty_x_obj2 = torch.max(torch.tensor(0.0, device=env.device), proximity_threshold - dist_x_obj2)
        penalty_y_obj2 = torch.max(torch.tensor(0.0, device=env.device), proximity_threshold - dist_y_obj2)
        penalty_z_obj2 = torch.max(torch.tensor(0.0, device=env.device), proximity_threshold - dist_z_obj2)
        collision_penalty -= (penalty_x_obj2 + penalty_y_obj2 + penalty_z_obj2) * 0.5

        # Check against Object3 (target block) - only penalize if not hands.
        # Hands are expected to touch Object3 for pushing, so they are excluded from this penalty.
        # Using relative distances.
        if part_name not in ['left_palm_link', 'right_palm_link', 'left_two_link', 'right_two_link', 'left_five_link', 'right_five_link', 'left_six_link', 'right_six_link']:
            dist_x_obj3 = torch.abs(part_pos[:, 0] - object3.data.root_pos_w[:, 0])
            dist_y_obj3 = torch.abs(part_pos[:, 1] - object3.data.root_pos_w[:, 1])
            dist_z_obj3 = torch.abs(part_pos[:, 2] - object3.data.root_pos_w[:, 2])
            penalty_x_obj3 = torch.max(torch.tensor(0.0, device=env.device), proximity_threshold - dist_x_obj3)
            penalty_y_obj3 = torch.max(torch.tensor(0.0, device=env.device), proximity_threshold - dist_y_obj3)
            penalty_z_obj3 = torch.max(torch.tensor(0.0, device=env.device), proximity_threshold - dist_z_obj3)
            collision_penalty -= (penalty_x_obj3 + penalty_y_obj3 + penalty_z_obj3) * 0.5

        # Check against Object4 (platform) - penalize if body parts are inside the platform.
        # The platform is very thin, so focus on Z-axis penetration and XY containment.
        # Using relative positions and hardcoded dimensions.
        platform_top_z = object4.data.root_pos_w[:, 2] + platform_half_height
        platform_bottom_z = object4.data.root_pos_w[:, 2] - platform_half_height

        platform_x_min = object4.data.root_pos_w[:, 0] - object4_half_size_x
        platform_x_max = object4.data.root_pos_w[:, 0] + object4_half_size_x
        platform_y_min = object4.data.root_pos_w[:, 1] - object4_half_size_y
        platform_y_max = object4.data.root_pos_w[:, 1] + object4_half_size_y

        # Check if part is within platform's x,y bounds.
        # Ensure tensor operations work with batched environments.
        is_within_platform_xy = (part_pos[:, 0] > platform_x_min) & (part_pos[:, 0] < platform_x_max) & \
                                (part_pos[:, 1] > platform_y_min) & (part_pos[:, 1] < platform_y_max)

        # Penalize if part is within platform XY and Z is below platform top surface (i.e., penetrating).
        # The penalty is proportional to the depth of penetration.
        # Ensure tensor operations work with batched environments.
        penetration_depth_z = torch.max(torch.tensor(0.0, device=env.device), platform_top_z - part_pos[:, 2])
        collision_platform_penalty = torch.where(is_within_platform_xy, penetration_depth_z, torch.tensor(0.0, device=env.device))
        collision_penalty -= collision_platform_penalty * 2.0 # Higher penalty for platform collision

    # The reward is the accumulated collision penalty. Rewards are continuous.
    reward = collision_penalty

    # Mandatory reward normalization as per rule 2.
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
    Configuration for the reward terms for the 'push_third_0_5m_cubed_block_onto_platform' skill.
    """
    # Main reward for successfully pushing the block onto the platform.
    # This is the primary goal and has the highest weight (1.0) as per rule.
    MainPushBlockOntoPlatformReward = RewTerm(
        func=main_push_block_onto_platform_reward,
        weight=1.0,
        params={"normalise": True, "normaliser_name": "main_reward"}
    )

    # Shaping reward to guide the robot's hands towards the block for pushing.
    # Active when the block is not yet on the platform.
    # Weight is moderate (0.6) as per rule.
    RobotHandProximityToBlockReward = RewTerm(
        func=robot_hand_proximity_to_block_reward,
        weight=0.6, # Moderate weight to encourage interaction
        params={"normalise": True, "normaliser_name": "hand_proximity_reward"}
    )

    # Shaping reward for robot stability and position after the push is completed.
    # Active when the block is successfully on the platform.
    # Weight is moderate (0.5) as per rule.
    RobotStabilityAfterPushReward = RewTerm(
        func=robot_stability_after_push_reward,
        weight=0.5, # Moderate weight to encourage good final posture
        params={"normalise": True, "normaliser_name": "stability_reward"}
    )

    # General collision avoidance penalty for all non-intended contacts.
    # Weight is lower (0.2) as it's a general penalty, not task-specific guidance.
    CollisionAvoidanceReward = RewTerm(
        func=collision_avoidance_reward,
        weight=0.2, # Lower weight as it's a general penalty, not task-specific guidance
        params={"normalise": True, "normaliser_name": "collision_reward"}
    )