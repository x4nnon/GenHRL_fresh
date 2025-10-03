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
    Main reward for pushing the 'second 0.5m cubed block' (Object2) onto the 'platform' (Object4).
    This reward guides the block horizontally towards the platform and then provides a higher reward
    when it is fully resting on the platform, encouraging it to be centered.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object2 = env.scene['Object2'] # second 0.5m cubed block
    object4 = env.scene['Object4'] # platform

    # Object dimensions (hardcoded from object configuration as per requirements)
    # This adheres to the rule: "THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it."
    block_half_size = 0.5 / 2.0 # 0.25m for 0.5m cubed block
    platform_x_size = 2.0
    platform_y_size = 2.0
    platform_z_height = 0.001 # Assuming platform is thin, its top surface is at z=0.001

    # Get positions of Object2 and Object4 using approved patterns
    # This adheres to the rule: "ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w"
    object2_pos_x = object2.data.root_pos_w[:, 0]
    object2_pos_y = object2.data.root_pos_w[:, 1]
    object2_pos_z = object2.data.root_pos_w[:, 2]

    object4_pos_x = object4.data.root_pos_w[:, 0]
    object4_pos_y = object4.data.root_pos_w[:, 1]
    object4_pos_z = object4.data.root_pos_w[:, 2]

    # Calculate horizontal distance between Object2 and Object4 centers (relative distances)
    # This adheres to the rule: "ALL rewards MUST ONLY use relative distances between objects and robot parts"
    dist_x_obj2_obj4 = torch.abs(object2_pos_x - object4_pos_x)
    dist_y_obj2_obj4 = torch.abs(object2_pos_y - object4_pos_y)

    # Reward for moving Object2 closer to Object4 (horizontal alignment)
    # This is a continuous reward that increases as the block gets closer to the platform's center.
    # This adheres to the rule: "Continuous Rewards"
    horizontal_approach_reward = 1.0 / (1.0 + dist_x_obj2_obj4 + dist_y_obj2_obj4)

    # Conditions for Object2 being on Object4 (relative positioning)
    # Check if Object2 is within platform's x-bounds
    # The block's edges must be within the platform's edges.
    is_on_platform_x = (object2_pos_x - block_half_size >= object4_pos_x - platform_x_size / 2.0) & \
                       (object2_pos_x + block_half_size <= object4_pos_x + platform_x_size / 2.0)

    # Check if Object2 is within platform's y-bounds
    is_on_platform_y = (object2_pos_y - block_half_size >= object4_pos_y - platform_y_size / 2.0) & \
                       (object2_pos_y + block_half_size <= object4_pos_y + platform_y_size / 2.0)

    # Check if Object2 is resting on the platform (z-position)
    # Object2's bottom surface should be at platform_z_height. Its center z should be platform_z_height + block_half_size.
    # A small tolerance is used for the z-height check.
    target_object2_z = object4_pos_z + block_half_size
    is_on_platform_z = torch.abs(object2_pos_z - target_object2_z) < 0.05 # Small tolerance for z-height

    # Combined condition for Object2 being fully on Object4
    object2_fully_on_platform = is_on_platform_x & is_on_platform_y & is_on_platform_z

    # Reward for Object2 being fully on Object4
    # This reward is active and high when the block is correctly placed.
    # The term (dist_x_obj2_obj4 + dist_y_obj2_obj4) * 2.0 ensures a slight gradient even when on platform,
    # encouraging it to be centered, and prevents a flat reward. Max 5.0, min 0.0.
    containment_reward = torch.where(object2_fully_on_platform, 5.0 - (dist_x_obj2_obj4 + dist_y_obj2_obj4) * 2.0, 0.0)
    # Ensure containment_reward doesn't go below zero
    containment_reward = torch.max(containment_reward, torch.zeros_like(containment_reward))

    # Combine rewards: prioritize containment if met, otherwise focus on approach
    # This creates a smooth transition between approach and containment rewards.
    reward = torch.where(object2_fully_on_platform, containment_reward, horizontal_approach_reward)

    # Mandatory reward normalization
    # This adheres to the rule: "ALWAYS implement proper reward normalization"
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def hand_approach_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "hand_approach_reward") -> torch.Tensor:
    """
    Shaping reward that guides the robot's 'right_palm_link' to approach the 'second 0.5m cubed block' (Object2)
    and position it at an appropriate height for pushing. This reward is active only when Object2 is not yet on the platform.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    # This adheres to the rule: "ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w"
    # And: "ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]"
    robot = env.scene["robot"]
    object2 = env.scene['Object2'] # second 0.5m cubed block
    object4 = env.scene['Object4'] # platform (needed for activation condition)

    right_palm_link_idx = robot.body_names.index('right_palm_link')
    right_palm_link_pos = robot.data.body_pos_w[:, right_palm_link_idx]

    # Object dimensions (hardcoded from object configuration)
    block_half_size = 0.5 / 2.0 # 0.25m

    # Get positions of Object2
    object2_pos_x = object2.data.root_pos_w[:, 0]
    object2_pos_y = object2.data.root_pos_w[:, 1]
    object2_pos_z = object2.data.root_pos_w[:, 2]

    # Calculate horizontal distance between right_palm_link and Object2 (relative distances)
    # This adheres to the rule: "ALL rewards MUST ONLY use relative distances between objects and robot parts"
    dist_x_hand_obj2 = torch.abs(right_palm_link_pos[:, 0] - object2_pos_x)
    dist_y_hand_obj2 = torch.abs(right_palm_link_pos[:, 1] - object2_pos_y)

    # Reward for hand approaching the block horizontally
    # This is a continuous reward that increases as the hand gets closer to the block's horizontal center.
    approach_hand_reward = 1.0 / (1.0 + dist_x_hand_obj2 + dist_y_hand_obj2)

    # Target z-height for the hand relative to the block's center for pushing
    # A good pushing height would be around the block's center.
    target_hand_z = object2_pos_z
    # Reward for vertical alignment, continuous and higher when closer to target_hand_z.
    height_alignment_reward = 1.0 / (1.0 + torch.abs(right_palm_link_pos[:, 2] - target_hand_z))

    # Combine approach and height alignment rewards
    shaping_reward = approach_hand_reward + height_alignment_reward

    # Activation condition: Only active when Object2 is NOT yet fully on Object4
    # Re-using logic for object2_fully_on_platform from the primary reward for consistency.
    platform_x_size = 2.0
    platform_y_size = 2.0
    platform_z_height = 0.001
    object4_pos_x = object4.data.root_pos_w[:, 0]
    object4_pos_y = object4.data.root_pos_w[:, 1]
    object4_pos_z = object4.data.root_pos_w[:, 2]
    target_object2_z = object4_pos_z + block_half_size

    is_on_platform_x = (object2_pos_x - block_half_size >= object4_pos_x - platform_x_size / 2.0) & \
                       (object2_pos_x + block_half_size <= object4_pos_x + platform_x_size / 2.0)
    is_on_platform_y = (object2_pos_y - block_half_size >= object4_pos_y - platform_y_size / 2.0) & \
                       (object2_pos_y + block_half_size <= object4_pos_y + platform_y_size / 2.0)
    is_on_platform_z = torch.abs(object2_pos_z - target_object2_z) < 0.05

    object2_fully_on_platform = is_on_platform_x & is_on_platform_y & is_on_platform_z

    # Apply activation condition: reward is 0 if block is already on platform
    reward = torch.where(~object2_fully_on_platform, shaping_reward, 0.0)

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def robot_stability_and_position_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "robot_stability_reward") -> torch.Tensor:
    """
    Shaping reward that encourages robot stability (pelvis height) and proper final positioning
    after the push, ensuring the robot does not move onto the platform or too far past it,
    and is ready to access the next block.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    robot = env.scene["robot"]
    object2 = env.scene['Object2'] # second 0.5m cubed block
    object4 = env.scene['Object4'] # platform
    object1 = env.scene['Object1'] # first 0.5m cubed block (for next skill positioning)
    object3 = env.scene['Object3'] # third 0.5m cubed block (for next skill positioning)

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Object dimensions (hardcoded from object configuration)
    block_half_size = 0.5 / 2.0 # 0.25m
    platform_x_size = 2.0
    platform_y_size = 2.0
    platform_z_height = 0.001

    # Get positions of Object2 and Object4
    object2_pos_x = object2.data.root_pos_w[:, 0]
    object2_pos_y = object2.data.root_pos_w[:, 1]
    object2_pos_z = object2.data.root_pos_w[:, 2]

    object4_pos_x = object4.data.root_pos_w[:, 0]
    object4_pos_y = object4.data.root_pos_w[:, 1]
    object4_pos_z = object4.data.root_pos_w[:, 2]

    # Pelvis height stability reward
    # Encourages the pelvis to stay at a stable, typical height.
    # This uses an absolute z-position for the pelvis, which is allowed sparingly for height.
    target_pelvis_z = 0.7 # A reasonable default stable pelvis height for this robot
    pelvis_height_reward = 1.0 / (1.0 + torch.abs(pelvis_pos[:, 2] - target_pelvis_z))

    # Penalty for robot being on the platform
    # Check if pelvis is within platform's x and y bounds (relative to platform center)
    is_pelvis_on_platform_x = (pelvis_pos[:, 0] >= object4_pos_x - platform_x_size / 2.0) & \
                              (pelvis_pos[:, 0] <= object4_pos_x + platform_x_size / 2.0)
    is_pelvis_on_platform_y = (pelvis_pos[:, 1] >= object4_pos_y - platform_y_size / 2.0) & \
                              (pelvis_pos[:, 1] <= object4_pos_y + platform_y_size / 2.0)
    # Check if pelvis is above the platform surface (relative to platform z)
    is_pelvis_on_platform_z = (pelvis_pos[:, 2] > object4_pos_z + 0.1) # Pelvis is significantly above platform surface

    pelvis_on_platform_penalty = torch.where(is_pelvis_on_platform_x & is_pelvis_on_platform_y & is_pelvis_on_platform_z, -2.0, 0.0)

    # Reward for robot being positioned to access next blocks (Object1 or Object3)
    # This encourages the robot to return to or stay near the initial block area after pushing.
    # Calculate the average x,y of Object1 and Object3 as a target for the pelvis.
    # This uses relative distances between pelvis and the average position of other blocks.
    avg_obj1_obj3_x = (object1.data.root_pos_w[:, 0] + object3.data.root_pos_w[:, 0]) / 2.0
    avg_obj1_obj3_y = (object1.data.root_pos_w[:, 1] + object3.data.root_pos_w[:, 1]) / 2.0

    # Calculate relative distances from pelvis to the average position of the other blocks
    dist_x_pelvis_next_blocks = torch.abs(pelvis_pos[:, 0] - avg_obj1_obj3_x)
    dist_y_pelvis_next_blocks = torch.abs(pelvis_pos[:, 1] - avg_obj1_obj3_y)

    # Reward for being close to the next blocks' area, active only when the current block is on the platform.
    pelvis_position_for_next_skill_reward = 1.0 / (1.0 + dist_x_pelvis_next_blocks + dist_y_pelvis_next_blocks)

    # Combined condition for Object2 being fully on Object4 (re-using logic from primary reward)
    # This reward component is primarily active when the block is on the platform, to guide final robot position.
    target_object2_z = object4_pos_z + block_half_size
    object2_fully_on_platform = (object2_pos_x - block_half_size >= object4_pos_x - platform_x_size / 2.0) & \
                                (object2_pos_x + block_half_size <= object4_pos_x + platform_x_size / 2.0) & \
                                (object2_pos_y - block_half_size >= object4_pos_y - platform_y_size / 2.0) & \
                                (object2_pos_y + block_half_size <= object4_pos_y + platform_y_size / 2.0) & \
                                (torch.abs(object2_pos_z - target_object2_z) < 0.05)

    # Combine rewards
    reward = pelvis_height_reward + pelvis_on_platform_penalty
    # Only add the next skill positioning reward if the current block is successfully placed
    reward += torch.where(object2_fully_on_platform, pelvis_position_for_next_skill_reward, 0.0)

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
    Configuration for the reward terms used in the push_second_0_5m_cubed_block_onto_platform skill.
    """
    # Primary reward for moving the block onto the platform
    # Weight is 1.0 as per requirement: "PROPER WEIGHTS - Set appropriate weights in TaskRewardsCfg (primary reward ~1.0, supporting rewards <1.0)"
    main_push_block_onto_platform_reward = RewTerm(
        func=main_push_block_onto_platform_reward,
        weight=1.0, # Main reward typically has a weight of 1.0
        params={"normalise": True, "normaliser_name": "main_reward"}
    )

    # Shaping reward for guiding the hand to the block for pushing
    # Weight is lower than main reward as per requirement.
    hand_approach_block_reward = RewTerm(
        func=hand_approach_block_reward,
        weight=0.5, # Shaping reward, typically lower weight than main reward
        params={"normalise": True, "normaliser_name": "hand_approach_reward"}
    )

    # Shaping reward for robot stability and final positioning
    # Weight is lower than main reward as per requirement.
    robot_stability_and_position_reward = RewTerm(
        func=robot_stability_and_position_reward,
        weight=0.3, # Shaping reward, typically lower weight than main reward
        params={"normalise": True, "normaliser_name": "robot_stability_reward"}
    )