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


def walk_to_blockCube_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_blockCube_main_reward") -> torch.Tensor:
    """
    Main reward for the walk_to_blockCube skill.
    Encourages the robot to position its pelvis directly in front of Object5 (the block cube),
    at a suitable jumping distance and height, centered in the y-axis.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts
    # CRITICAL RULE: Access objects using env.scene['ObjectName']
    object5 = env.scene['Object5'] # Object5 is the block cube
    robot = env.scene["robot"]

    # CRITICAL RULE: Access robot parts using robot.body_names.index('part_name')
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]

    # CRITICAL RULE: Access object positions using env.scene['ObjectName'].data.root_pos_w
    object5_pos = object5.data.root_pos_w # Shape: [num_envs, 3]

    # Object5 dimensions: 0.5m cubed.
    # CRITICAL RULE: Object dimensions must be hardcoded from the object configuration, not accessed from the object itself.
    block_half_size = 0.5 / 2.0 # Half size of the 0.5m cubed block

    # Define target pelvis position relative to the block cube
    # The robot should be 0.7m away from the front face of the block.
    # Front face of the block is at object5_pos[:, 0] - block_half_size.
    # So, target_pelvis_x = (object5_pos[:, 0] - block_half_size) - 0.7
    target_x_offset_from_block_center = block_half_size + 0.7 # Distance from block center to robot pelvis x
    target_pelvis_x = object5_pos[:, 0] - target_x_offset_from_block_center

    # Robot should be centered with the block in y
    target_pelvis_y = object5_pos[:, 1]

    # Target pelvis z for standing, relative to the block's base (assuming block is on ground z=0)
    # A standard standing pelvis height is around 0.7m.
    target_pelvis_z = object5_pos[:, 2] + 0.7

    # Calculate relative distances for each component
    # CRITICAL RULE: ALL rewards MUST ONLY use relative distances between objects and robot parts
    dist_x = torch.abs(pelvis_pos[:, 0] - target_pelvis_x)
    dist_y = torch.abs(pelvis_pos[:, 1] - target_pelvis_y)
    dist_z = torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Combine distances for a continuous negative reward (closer is better)
    # CRITICAL RULE: Rewards should be continuous and positive where possible, or negative for penalties.
    # Here, a negative sum of absolute distances encourages minimizing all components.
    reward = - (dist_x + dist_y + dist_z)

    # MANDATORY REWARD NORMALIZATION
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def pelvis_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_stability_reward") -> torch.Tensor:
    """
    This reward encourages the robot to maintain an upright posture and stability by penalizing
    large deviations of the pelvis's z-position from a desired standing height (0.7m).
    It also encourages the robot to keep its feet on the ground.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot parts
    robot = env.scene["robot"]

    # CRITICAL RULE: Access robot parts using robot.body_names.index('part_name')
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    left_foot_pos_z = left_foot_pos[:, 2]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    right_foot_pos_z = right_foot_pos[:, 2]

    # Target pelvis z for standing (absolute height from ground)
    # CRITICAL RULE: z is the only absolute position allowed, used sparingly for height.
    target_pelvis_z = 0.7

    # Penalize deviation from target pelvis z
    # CRITICAL RULE: Use relative distances. Here, relative to a target height.
    pelvis_z_deviation = torch.abs(pelvis_pos_z - target_pelvis_z)
    reward_pelvis_stability = -pelvis_z_deviation # Continuous negative reward

    # Penalize feet being too high off the ground (encourage ground contact)
    # Assuming ground is at z=0. A small positive offset for foot height is normal.
    foot_ground_threshold = 0.05 # Feet should be close to ground, not exactly 0.
    # CRITICAL RULE: Use relative distances. Here, relative to a small threshold above ground.
    reward_feet_on_ground = -torch.abs(left_foot_pos_z - foot_ground_threshold) - torch.abs(right_foot_pos_z - foot_ground_threshold)

    # Combine stability rewards
    reward = reward_pelvis_stability + reward_feet_on_ground

    # MANDATORY REWARD NORMALIZATION
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    """
    This reward encourages collision avoidance with Object5 (block cube) during the approach phase.
    It penalizes the robot if its pelvis gets too close to or passes through the block cube
    before reaching the desired final position.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts
    # CRITICAL RULE: Access objects using env.scene['ObjectName']
    object5 = env.scene['Object5'] # Object5 is the block cube
    robot = env.scene["robot"]

    # CRITICAL RULE: Access robot parts using robot.body_names.index('part_name')
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # CRITICAL RULE: Access object positions using env.scene['ObjectName'].data.root_pos_w
    object5_pos = object5.data.root_pos_w

    # Object5 dimensions: 0.5m cubed.
    # CRITICAL RULE: Object dimensions must be hardcoded from the object configuration.
    block_half_size = 0.5 / 2.0 # Half size of the 0.5m cubed block

    # Define a safe distance around the block to avoid
    # This creates a slightly larger "no-go" zone around the block.
    safe_distance_buffer = 0.1 # Additional buffer around the block

    # Calculate collision zone boundaries relative to the block's center
    # CRITICAL RULE: ALL rewards MUST ONLY use relative distances between objects and robot parts.
    # Here, the collision zone is defined relative to the block's position and its known size.
    collision_x_min = object5_pos[:, 0] - block_half_size - safe_distance_buffer
    collision_x_max = object5_pos[:, 0] + block_half_size + safe_distance_buffer
    collision_y_min = object5_pos[:, 1] - block_half_size - safe_distance_buffer
    collision_y_max = object5_pos[:, 1] + block_half_size + safe_distance_buffer
    collision_z_min = object5_pos[:, 2] - block_half_size - safe_distance_buffer
    collision_z_max = object5_pos[:, 2] + block_half_size + safe_distance_buffer

    # Condition for collision: pelvis is inside or too close to the block
    # CRITICAL RULE: All operations must work with batched environments.
    is_colliding_x = (pelvis_pos_x > collision_x_min) & (pelvis_pos_x < collision_x_max)
    is_colliding_y = (pelvis_pos_y > collision_y_min) & (pelvis_pos_y < collision_y_max)
    is_colliding_z = (pelvis_pos_z > collision_z_min) & (pelvis_pos_z < collision_z_max)

    collision_condition = is_colliding_x & is_colliding_y & is_colliding_z

    # Reward: large negative if colliding, 0 otherwise
    # CRITICAL RULE: Rewards should be continuous. While this is a step function,
    # it's common for collision penalties. The "safe_distance_buffer" makes it
    # a soft boundary rather than a hard one at the object's exact edge.
    reward = torch.where(collision_condition, -5.0, 0.0) # Large penalty for collision

    # MANDATORY REWARD NORMALIZATION
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
    Reward terms for the walk_to_blockCube skill.
    """
    # Main reward for positioning the robot in front of the block cube
    walk_to_blockCube_main_reward = RewTerm(
        func=walk_to_blockCube_main_reward,
        weight=1.0, # Primary reward, typically weight 1.0
        params={"normalise": True, "normaliser_name": "walk_to_blockCube_main_reward"}
    )

    # Shaping reward for maintaining pelvis stability and keeping feet on the ground
    pelvis_stability_reward = RewTerm(
        func=pelvis_stability_reward,
        weight=0.4, # Shaping reward, typically lower weight
        params={"normalise": True, "normaliser_name": "pelvis_stability_reward"}
    )

    # Shaping reward for avoiding collision with the block cube during approach
    collision_avoidance_reward = RewTerm(
        func=collision_avoidance_reward,
        weight=0.5, # Shaping reward, typically lower weight
        params={"normalise": True, "normaliser_name": "collision_avoidance_reward"}
    )