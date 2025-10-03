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


def approach_object2_position_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_object2_position_reward") -> torch.Tensor:
    """
    Primary reward for positioning the robot's pelvis directly behind 'second 0.5m cubed block' (Object2),
    on the side opposite the 'platform' (Object4), ready to push it.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts
    # CRITICAL RULE: ALWAYS access objects using env.scene['ObjectName']
    object2 = env.scene['Object2'] # second 0.5m cubed block
    object4 = env.scene['Object4'] # platform
    robot = env.scene["robot"]

    # CRITICAL RULE: ALWAYS access robot parts using robot.body_names.index('part_name')
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]

    # CRITICAL RULE: ALWAYS access object positions using env.scene['ObjectName'].data.root_pos_w
    object2_pos = object2.data.root_pos_w # Shape: [num_envs, 3]
    object4_pos = object4.data.root_pos_w # Shape: [num_envs, 3]

    # CRITICAL RULE: NEVER use hard-coded positions or arbitrary thresholds.
    # Object2 dimensions (0.5m cubed block) - CRITICAL RULE: Hardcode dimensions from object config, DO NOT access from object.
    object2_half_depth = 0.5 / 2.0 # Assuming 0.5m is the side length of the cube
    target_clearance = 0.1 # Desired distance from the block's face for pushing

    # Calculate the vector from Object2 to Object4 to determine the pushing direction.
    # This ensures the robot positions itself on the side opposite Object4 dynamically.
    # CRITICAL RULE: ALL rewards MUST ONLY use relative distances between objects and robot parts.
    object2_to_object4_vec = object4_pos[:, :2] - object2_pos[:, :2] # Only consider X-Y plane for direction

    # Normalize the direction vector to get a unit vector for alignment.
    direction_magnitude = torch.norm(object2_to_object4_vec, dim=1, keepdim=True)
    # Add a small epsilon to avoid division by zero if objects are at the same spot.
    direction_magnitude = torch.where(direction_magnitude == 0, torch.tensor(1e-6, device=env.device), direction_magnitude)
    norm_dir = object2_to_object4_vec / direction_magnitude # Shape: [num_envs, 2]

    # Calculate the desired pelvis position relative to Object2 based on the pushing direction.
    # The robot should be on the opposite side of Object2 from Object4.
    # So, we move *against* the normalized direction vector from Object2's center.
    desired_pelvis_xy_w = object2_pos[:, :2] - norm_dir * (object2_half_depth + target_clearance)

    # Calculate distance components to the desired X-Y position.
    # CRITICAL RULE: Use torch.abs for absolute distances.
    dist_x = torch.abs(desired_pelvis_xy_w[:, 0] - pelvis_pos[:, 0])
    dist_y = torch.abs(desired_pelvis_xy_w[:, 1] - pelvis_pos[:, 1])

    # Reward for approaching the target x,y position.
    # Use negative sum of absolute differences for a continuous positive reward that peaks at 0 distance.
    # CRITICAL RULE: Rewards should be continuous and positive (or negative for penalties).
    reward_approach_xy = -dist_x - dist_y

    # Reward for maintaining a stable pelvis height.
    target_pelvis_z = 0.7 # A reasonable stable height for the robot's pelvis.
    reward_pelvis_z = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Combine rewards.
    reward = reward_approach_xy + reward_pelvis_z

    # CRITICAL RULE: MANDATORY REWARD NORMALIZATION
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_other_blocks_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_other_blocks_reward") -> torch.Tensor:
    """
    Shaping reward that penalizes the robot for colliding with or getting too close to
    Object1 (first 0.5m cubed block), Object3 (third 0.5m cubed block), and Object4 (platform).
    This encourages clean navigation to Object2.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts
    object1 = env.scene['Object1'] # first 0.5m cubed block
    object3 = env.scene['Object3'] # third 0.5m cubed block
    object4 = env.scene['Object4'] # platform
    robot = env.scene["robot"]

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # CRITICAL RULE: Hardcode dimensions from object config, DO NOT access from object.
    block_half_size = 0.5 / 2.0 # For Object1 and Object3 (0.5m cubed blocks)
    # platform_height = 0.001 # Not directly used for collision distance, but good to note.

    # Define a safe distance threshold. This is a relative threshold.
    # It should be slightly larger than the sum of the block's half-size and a conceptual robot radius.
    safe_distance_threshold = block_half_size + 0.3 # 0.25 + 0.3 = 0.55m clearance

    # Calculate Euclidean distances from pelvis to Object1, Object3, and Object4.
    # CRITICAL RULE: ALL rewards MUST ONLY use relative distances between objects and robot parts.
    dist_obj1 = torch.norm(object1.data.root_pos_w - pelvis_pos, dim=1)
    dist_obj3 = torch.norm(object3.data.root_pos_w - pelvis_pos, dim=1)
    dist_obj4 = torch.norm(object4.data.root_pos_w - pelvis_pos, dim=1)

    # Penalize if too close to Object1, Object3, or Object4.
    # The penalty is inverse to the distance when within the threshold, making it stronger closer to the object.
    # CRITICAL RULE: Rewards should be continuous.
    # Add a small epsilon to avoid division by zero.
    reward_collision_obj1 = torch.where(dist_obj1 < safe_distance_threshold, -1.0 / (dist_obj1 + 1e-6), 0.0)
    reward_collision_obj3 = torch.where(dist_obj3 < safe_distance_threshold, -1.0 / (dist_obj3 + 1e-6), 0.0)
    reward_collision_obj4 = torch.where(dist_obj4 < safe_distance_threshold, -1.0 / (dist_obj4 + 1e-6), 0.0)

    # Sum of collision penalties.
    reward = reward_collision_obj1 + reward_collision_obj3 + reward_collision_obj4

    # CRITICAL RULE: MANDATORY REWARD NORMALIZATION
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def feet_on_ground_and_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "feet_on_ground_and_stability_reward") -> torch.Tensor:
    """
    Shaping reward that encourages the robot to keep its feet on the ground and maintain a stable standing posture.
    Penalizes feet being too high off the ground and rewards the pelvis being at a stable height.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required robot parts
    robot = env.scene["robot"]
    # CRITICAL RULE: ALWAYS access robot parts using robot.body_names.index('part_name')
    left_ankle_idx = robot.body_names.index('left_ankle_roll_link')
    right_ankle_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    # CRITICAL RULE: ALWAYS access robot part positions using robot.data.body_pos_w
    left_ankle_pos_z = robot.data.body_pos_w[:, left_ankle_idx, 2]
    right_ankle_pos_z = robot.data.body_pos_w[:, right_ankle_idx, 2]
    pelvis_pos_z = robot.data.body_pos_w[:, pelvis_idx, 2]

    # Target ground height (assuming ground is at z=0).
    # CRITICAL RULE: z is the only absolute position allowed, used sparingly for height.
    ground_z = 0.0
    feet_off_ground_threshold = 0.05 # Small tolerance for feet being slightly off the ground.

    # Penalize if feet are too high off the ground.
    # This is a continuous penalty, increasing with height above threshold.
    reward_feet_on_ground = 0.0
    # The original prompt's skeleton for this reward had `left_ankle_pos_z` directly,
    # but it should be `left_ankle_pos_z - feet_off_ground_threshold` to penalize *above* the threshold.
    reward_feet_on_ground -= torch.where(left_ankle_pos_z > feet_off_ground_threshold, left_ankle_pos_z - feet_off_ground_threshold, 0.0)
    reward_feet_on_ground -= torch.where(right_ankle_pos_z > feet_off_ground_threshold, right_ankle_pos_z - feet_off_ground_threshold, 0.0)

    # Reward for pelvis being at a stable height.
    # This is a continuous reward, always active, penalizing deviation from target_pelvis_z.
    target_pelvis_z = 0.7
    reward_pelvis_height = -torch.abs(pelvis_pos_z - target_pelvis_z)

    # Combine rewards.
    reward = reward_feet_on_ground + reward_pelvis_height

    # CRITICAL RULE: MANDATORY REWARD NORMALIZATION
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Primary reward for positioning the robot behind Object2.
    # CRITICAL RULE: Main reward with weight 1.0.
    ApproachObject2PositionReward = RewTerm(func=approach_object2_position_reward, weight=1.0,
                                            params={"normalise": True, "normaliser_name": "approach_object2_position_reward"})

    # Shaping reward for avoiding collisions with other blocks and the platform.
    # CRITICAL RULE: Supporting rewards with lower weights (<1.0).
    CollisionAvoidanceOtherBlocksReward = RewTerm(func=collision_avoidance_other_blocks_reward, weight=0.4,
                                                  params={"normalise": True, "normaliser_name": "collision_avoidance_other_blocks_reward"})

    # Shaping reward for maintaining feet on the ground and stable pelvis height.
    # CRITICAL RULE: Supporting rewards with lower weights (<1.0).
    FeetOnGroundAndStabilityReward = RewTerm(func=feet_on_ground_and_stability_reward, weight=0.2,
                                             params={"normalise": True, "normaliser_name": "feet_on_ground_and_stability_reward"})