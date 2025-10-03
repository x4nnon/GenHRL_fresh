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


def ascend_third_step_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "ascend_third_step_main_reward") -> torch.Tensor:
    """
    Main reward for the ascend_third_step skill.
    Encourages the robot to stand stably on top of the Large Block (Object3).
    Rewards for feet being centered and at the correct height on the block, and for the pelvis being centered
    and at a target height above the block, active only when feet are properly positioned.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    large_block = env.scene['Object3'] # Accessing Object3 (Large Block)
    robot = env.scene["robot"] # Accessing the robot

    # Get indices for robot body parts
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    # Get positions of robot body parts (absolute positions in world frame)
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Get Large Block's root position (absolute position in world frame)
    large_block_pos = large_block.data.root_pos_w

    # Hardcode Object3 dimensions based on the provided configuration (x=1m y=1m z=0.9m)
    # This adheres to the rule: "THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it."
    large_block_height = 0.9
    large_block_half_x = 0.5
    large_block_half_y = 0.5

    # Calculate target Z position for feet (top surface of large block)
    # This is a relative target based on the block's position and its height
    target_foot_z = large_block_pos[:, 2] + large_block_height / 2.0

    # Calculate target Z position for pelvis (0.7m above block surface, as per plan)
    # This is a relative target based on the block's position and a desired offset
    target_pelvis_z = target_foot_z + 0.7

    # Calculate relative horizontal distances of feet to the center of the large block
    # This adheres to the rule: "ALL rewards MUST ONLY use relative distances between objects and robot parts"
    dist_left_foot_x = large_block_pos[:, 0] - left_foot_pos[:, 0]
    dist_left_foot_y = large_block_pos[:, 1] - left_foot_pos[:, 1]
    dist_right_foot_x = large_block_pos[:, 0] - right_foot_pos[:, 0]
    dist_right_foot_y = large_block_pos[:, 1] - right_foot_pos[:, 1]

    # Calculate relative vertical distances of feet to the target Z (top of large block)
    # A small positive offset (0.02m) is used to encourage feet to be slightly above or exactly on the surface
    dist_left_foot_z = target_foot_z - left_foot_pos[:, 2]
    dist_right_foot_z = target_foot_z - right_foot_pos[:, 2]

    # Reward for feet being on top of the block and centered
    # Uses absolute differences for continuous, negative reward (closer to zero is better)
    # This adheres to the rule: "Use smooth, continuous rewards."
    reward_feet_on_block = -torch.abs(dist_left_foot_x) - torch.abs(dist_left_foot_y) - torch.abs(dist_left_foot_z - 0.02) \
                         - torch.abs(dist_right_foot_x) - torch.abs(dist_right_foot_y) - torch.abs(dist_right_foot_z - 0.02)

    # Define conditions for feet being "on" the block (within horizontal bounds and above bottom surface)
    # These conditions use relative positions to the block's bounding box
    # Horizontal bounds check: foot_pos > block_center - half_dim AND foot_pos < block_center + half_dim
    left_foot_on_block_x = (left_foot_pos[:, 0] > large_block_pos[:, 0] - large_block_half_x) & \
                           (left_foot_pos[:, 0] < large_block_pos[:, 0] + large_block_half_x)
    left_foot_on_block_y = (left_foot_pos[:, 1] > large_block_pos[:, 1] - large_block_half_y) & \
                           (left_foot_pos[:, 1] < large_block_pos[:, 1] + large_block_half_y)
    # Vertical bounds check: foot_pos > block_bottom_surface (within 0.1m of top surface)
    left_foot_on_block_z = (left_foot_pos[:, 2] > large_block_pos[:, 2] + large_block_height / 2.0 - 0.1)

    right_foot_on_block_x = (right_foot_pos[:, 0] > large_block_pos[:, 0] - large_block_half_x) & \
                            (right_foot_pos[:, 0] < large_block_pos[:, 0] + large_block_half_x)
    right_foot_on_block_y = (right_foot_pos[:, 1] > large_block_pos[:, 1] - large_block_half_y) & \
                            (right_foot_pos[:, 1] < large_block_pos[:, 1] + large_block_half_y)
    right_foot_on_block_z = (right_foot_pos[:, 2] > large_block_pos[:, 2] + large_block_height / 2.0 - 0.1)

    # Combined condition: both feet are considered to be "on" the block
    feet_on_block_condition = (left_foot_on_block_x & left_foot_on_block_y & left_foot_on_block_z) & \
                              (right_foot_on_block_x & right_foot_on_block_y & right_foot_on_block_z)

    # Calculate relative distances of pelvis to the center of the large block and target height
    dist_pelvis_x = large_block_pos[:, 0] - pelvis_pos[:, 0]
    dist_pelvis_y = large_block_pos[:, 1] - pelvis_pos[:, 1]
    dist_pelvis_z = target_pelvis_z - pelvis_pos[:, 2]

    # Reward for pelvis stability and centering, active only once feet are on the block
    # Uses absolute differences for continuous, negative reward
    reward_pelvis_stability = -torch.abs(dist_pelvis_x) - torch.abs(dist_pelvis_y) - torch.abs(dist_pelvis_z)
    # Apply reward only when feet_on_block_condition is true, otherwise 0.0
    # This uses torch.where for batch compatibility, adhering to "Handle Tensor Operations Correctly"
    reward_pelvis_stability = torch.where(feet_on_block_condition, reward_pelvis_stability, torch.tensor(0.0, device=env.device))

    # Combine primary rewards
    reward = reward_feet_on_block + reward_pelvis_stability

    # Mandatory reward normalization
    # This adheres to the rule: "MANDATORY REWARD NORMALIZATION - EVERY reward function MUST include normalization"
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def ascend_third_step_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "ascend_third_step_collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward 1: Penalizes the robot for collisions with the sides/body of the Large Block (Object3).
    Encourages the robot to clear the block's edges rather than hitting them.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts
    large_block = env.scene['Object3'] # Accessing Object3 (Large Block)
    robot = env.scene["robot"] # Accessing the robot

    # Get indices for robot body parts
    pelvis_idx = robot.body_names.index('pelvis')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    left_knee_idx = robot.body_names.index('left_knee_link')
    right_knee_idx = robot.body_names.index('right_knee_link')

    # Get positions of robot body parts
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    left_knee_pos = robot.data.body_pos_w[:, left_knee_idx]
    right_knee_pos = robot.data.body_pos_w[:, right_knee_idx]

    # Get Large Block's root position
    large_block_pos = large_block.data.root_pos_w

    # Hardcode Object3 dimensions (x=1m y=1m z=0.9m)
    # This adheres to the rule: "THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it."
    large_block_height = 0.9
    large_block_half_x = 0.5
    large_block_half_y = 0.5

    # Define a small buffer to make collision detection slightly more lenient or to define "near" collision
    # This is an arbitrary threshold, but it's used for defining a "zone" for collision, not a hardcoded position.
    buffer = 0.1

    collision_penalty = torch.zeros(env.num_envs, device=env.device) # Initialize penalty tensor for batching

    # List of robot parts to check for collision
    robot_parts_to_check = [pelvis_pos, left_foot_pos, right_foot_pos, left_knee_pos, right_knee_pos]

    for part_pos in robot_parts_to_check:
        # Check if part is horizontally within block bounds (excluding a buffer zone)
        # This uses relative positions to the block's center and its half-dimensions
        is_within_x = (part_pos[:, 0] > large_block_pos[:, 0] - large_block_half_x + buffer) & \
                      (part_pos[:, 0] < large_block_pos[:, 0] + large_block_half_x - buffer)
        is_within_y = (part_pos[:, 1] > large_block_pos[:, 1] - large_block_half_y + buffer) & \
                      (part_pos[:, 1] < large_block_pos[:, 1] + large_block_half_y - buffer)

        # Check if part is vertically intersecting the block's body (not just on top surface)
        # This means the part's Z position is between the block's bottom and top surfaces, with a buffer
        is_intersecting_z = (part_pos[:, 2] > large_block_pos[:, 2] - large_block_height / 2.0 + buffer) & \
                            (part_pos[:, 2] < large_block_pos[:, 2] + large_block_height / 2.0 - buffer)

        # If a part is horizontally inside and vertically intersecting the block's body, apply penalty
        collision_condition = is_within_x & is_within_y & is_intersecting_z
        # Apply a large negative reward for collision, using torch.where for batch compatibility
        # This adheres to the rule: "Handle Tensor Operations Correctly"
        collision_penalty += torch.where(collision_condition, torch.tensor(-10.0, device=env.device), torch.tensor(0.0, device=env.device))

    reward = collision_penalty # The reward is the accumulated penalty

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def ascend_third_step_forward_progress_clearance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "ascend_third_step_forward_progress_clearance_reward") -> torch.Tensor:
    """
    Shaping reward 2: Encourages the robot to move its pelvis and feet forward towards the Large Block (Object3)
    and to maintain a positive vertical clearance over the block's top edge before landing.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts
    large_block = env.scene['Object3'] # Accessing Object3 (Large Block)
    robot = env.scene["robot"] # Accessing the robot

    # Get indices for robot body parts
    pelvis_idx = robot.body_names.index('pelvis')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')

    # Get positions of robot body parts
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Get Large Block's root position
    large_block_pos = large_block.data.root_pos_w

    # Hardcode Object3 dimensions (x=1m y=1m z=0.9m)
    # This adheres to the rule: "THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it."
    large_block_height = 0.9
    large_block_half_x = 0.5

    # Calculate the block's front edge x-coordinate (assuming robot approaches from negative x)
    # This is a relative position based on the block's center and its half-dimension
    block_front_edge_x = large_block_pos[:, 0] - large_block_half_x
    # Calculate the block's top surface Z-coordinate
    block_top_z = large_block_pos[:, 2] + large_block_height / 2.0

    # Reward for pelvis moving towards the block's center in x-direction
    # This encourages forward movement until the pelvis is over the block.
    # Uses absolute difference for continuous, negative reward (closer to zero is better)
    # This adheres to the rule: "ALL rewards MUST ONLY use relative distances between objects and robot parts"
    reward_pelvis_approach_x = -torch.abs(pelvis_pos[:, 0] - large_block_pos[:, 0])

    # Define thresholds for feet clearance reward activation
    # These are arbitrary thresholds for defining "zones" for reward activation, not hardcoded positions.
    clearance_threshold_x = 0.2 # Feet are considered "approaching" if within 0.2m horizontally of the block's front edge
    # Feet should be below the top surface to receive clearance reward, indicating they are still in the air
    # A small buffer (0.05m) is used to ensure they are not already considered "on top"
    z_below_top_threshold = block_top_z - 0.05

    # Condition for left foot approaching: horizontally near the block's front edge AND below the top surface
    left_foot_approaching_condition = (left_foot_pos[:, 0] > block_front_edge_x - clearance_threshold_x) & \
                                      (left_foot_pos[:, 0] < large_block_pos[:, 0] + large_block_half_x) & \
                                      (left_foot_pos[:, 2] < z_below_top_threshold)

    # Condition for right foot approaching: same as left foot
    right_foot_approaching_condition = (right_foot_pos[:, 0] > block_front_edge_x - clearance_threshold_x) & \
                                       (right_foot_pos[:, 0] < large_block_pos[:, 0] + large_block_half_x) & \
                                       (right_foot_pos[:, 2] < z_below_top_threshold)

    # Reward for positive z-clearance when approaching
    # (foot_z - block_top_z) will be positive if the foot is above the block's top surface
    # Scaled by 5.0 to make it significant, active only when approaching condition is met
    # This uses torch.where for batch compatibility, adhering to "Handle Tensor Operations Correctly"
    reward_left_foot_clearance = torch.where(left_foot_approaching_condition,
                                             (left_foot_pos[:, 2] - block_top_z) * 5.0,
                                             torch.tensor(0.0, device=env.device))
    reward_right_foot_clearance = torch.where(right_foot_approaching_condition,
                                              (right_foot_pos[:, 2] - block_top_z) * 5.0,
                                              torch.tensor(0.0, device=env.device))

    # Combine rewards
    reward = reward_pelvis_approach_x + reward_left_foot_clearance + reward_right_foot_clearance

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
    # Main reward for ascending the third step (Large Block)
    # Weight is 1.0 as per "PROPER WEIGHTS" rule.
    ascend_third_step_main_reward = RewTerm(func=ascend_third_step_main_reward, weight=1.0,
                                            params={"normalise": True, "normaliser_name": "ascend_third_step_main_reward"})

    # Shaping reward for collision avoidance with the Large Block
    # Weight is 0.6, less than main reward as per "PROPER WEIGHTS" rule.
    ascend_third_step_collision_avoidance_reward = RewTerm(func=ascend_third_step_collision_avoidance_reward, weight=0.6,
                                                           params={"normalise": True, "normaliser_name": "ascend_third_step_collision_avoidance_reward"})

    # Shaping reward for encouraging forward progress and vertical clearance over the block
    # Weight is 0.4, less than main reward as per "PROPER WEIGHTS" rule.
    ascend_third_step_forward_progress_clearance_reward = RewTerm(func=ascend_third_step_forward_progress_clearance_reward, weight=0.4,
                                                                  params={"normalise": True, "normaliser_name": "ascend_third_step_forward_progress_clearance_reward"})