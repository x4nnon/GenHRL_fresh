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


def walk_to_medium_block_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the walk_to_Medium_Block_for_robot_interaction skill.
    Rewards the robot for moving its pelvis to the side of the Medium Block (Object2)
    that is opposite to the Small Block (Object1), at a target horizontal distance,
    and for maintaining a stable upright posture. This positions the robot to push
    the Medium Block towards the Small Block.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    # CRITICAL RULE: ALWAYS access object positions using env.scene['ObjectName'].data.root_pos_w
    medium_block = env.scene['Object2']
    medium_block_pos = medium_block.data.root_pos_w
    # Also access the Small Block to determine the opposite side of the Medium Block
    small_block = env.scene['Object1']
    small_block_pos = small_block.data.root_pos_w

    robot = env.scene["robot"]
    # CRITICAL RULE: ALWAYS access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Target horizontal distance from pelvis to block center (e.g., 0.8m for pushing)
    # This value is chosen based on the skill description "within pushing distance" and typical robot dimensions.
    # CRITICAL RULE: NEVER use hard-coded positions or arbitrary thresholds. This is a target distance, not a position.
    target_horizontal_distance = 0.8 # meters

    # Compute the desired target position on the opposite side of the Medium Block relative to the Small Block.
    # CRITICAL RULE: ALL rewards MUST ONLY use relative distances between objects and robot parts.
    pelvis_xy = pelvis_pos[:, :2]
    medium_xy = medium_block_pos[:, :2]
    small_xy = small_block_pos[:, :2]
    # Direction pointing away from the Small Block, anchored at the Medium Block
    away_vec = medium_xy - small_xy
    away_norm = torch.norm(away_vec, dim=1, keepdim=True).clamp_min(1e-6)
    away_dir = away_vec / away_norm
    desired_xy = medium_xy + away_dir * target_horizontal_distance

    # Reward is negative distance from pelvis to desired opposite-side target position
    # CRITICAL RULE: Rewards should be continuous and positive where possible. This is continuous.
    position_error = torch.norm(pelvis_xy - desired_xy, dim=1)
    side_position_reward = -position_error

    # Additional shaping to avoid local minima on the wrong side of the block:
    # 1) Encourage positive progression along the away-from-small-block direction even before reaching the exact point.
    #    This provides a useful gradient to go around the block.
    rel_xy = pelvis_xy - medium_xy
    proj_along = torch.sum(rel_xy * away_dir, dim=1)  # signed distance along desired push direction
    proj_norm = proj_along / (target_horizontal_distance + 1e-6)
    alignment_reward = torch.tanh(proj_norm)  # in [-1, 1], positive when on the correct side

    # 2) Small penalty for lateral deviation from the push line to gently guide pathing.
    lateral_vec = rel_xy - away_dir * proj_along.unsqueeze(1)
    lateral_error = torch.norm(lateral_vec, dim=1)
    lateral_penalty = -0.1 * lateral_error

    # 3) Hinge penalty when on the wrong side of the medium block (proj_along < 0).
    wrong_side_penalty = -0.5 * torch.relu(-proj_along)

    # Target pelvis z-position for standing (0.7m).
    # This value is a sensible standing height for the robot.
    # CRITICAL RULE: NEVER use hard-coded positions or arbitrary thresholds. This is a target height, not a position.
    target_pelvis_z = 0.7 # meters

    # Calculate z-deviation from target standing height.
    # CRITICAL RULE: ALL rewards MUST ONLY use relative distances between objects and robot parts.
    # Here, it's the absolute difference from a target Z-height, which is a relative measure to a fixed point.
    pelvis_z_deviation = torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Reward for maintaining stable pelvis height.
    # Penalizes deviation from the target Z-height, encouraging upright posture.
    # CRITICAL RULE: Rewards should be continuous and positive where possible. This is continuous.
    z_reward = -pelvis_z_deviation

    # Combine rewards. Targeting the opposite-side position is the primary objective.
    # Alignment term helps escape wrong-side local minima; lateral penalty shapes the path.
    reward = side_position_reward + 0.5 * alignment_reward + lateral_penalty + wrong_side_penalty + 0.2 * z_reward

    # CRITICAL RULE: MANDATORY REWARD NORMALIZATION
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_reward") -> torch.Tensor:
    """
    Penalizes collisions between the robot's main body parts and all three blocks.
    This encourages safe navigation and prevents the robot from penetrating objects.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    # CRITICAL RULE: Access objects directly - objects should always exist in the scene
    small_block = env.scene['Object1']
    medium_block = env.scene['Object2']
    large_block = env.scene['Object3']

    # Robot body parts to monitor for collisions using approved patterns
    body_parts = ['pelvis', 'left_palm_link', 'right_palm_link', 'left_ankle_roll_link', 'right_ankle_roll_link', 'head_link']
    collision_reward = torch.zeros(env.num_envs, device=env.device)

    # Define a small buffer distance to avoid direct contact.
    # This buffer adds a small margin around the block's half-dimensions.
    # CRITICAL RULE: NEVER use hard-coded positions or arbitrary thresholds. This is a buffer distance.
    buffer_distance = 0.1 # meters

    # Hardcoded half-dimensions for each block, read from the object configuration.
    # CRITICAL RULE: THERE IS NO way to access the SIZE of an object. Hardcode values from config.
    small_block_half_dims = torch.tensor([0.5, 0.5, 0.15], device=env.device) # x=1m y=1m z=0.3m -> half: 0.5, 0.5, 0.15
    medium_block_half_dims = torch.tensor([0.5, 0.5, 0.3], device=env.device) # x=1m y=1m z=0.6m -> half: 0.5, 0.5, 0.3
    large_block_half_dims = torch.tensor([0.5, 0.5, 0.45], device=env.device) # x=1m y=1m z=0.9m -> half: 0.5, 0.5, 0.45

    for part_name in body_parts:
        # CRITICAL RULE: ALWAYS access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
        part_idx = env.scene["robot"].body_names.index(part_name)
        part_pos = env.scene["robot"].data.body_pos_w[:, part_idx]

        # Check against Small Block (Object1)
        # CRITICAL RULE: ALL rewards MUST ONLY use relative distances between objects and robot parts.
        # Calculate absolute differences in each dimension.
        dist_to_obj1 = torch.abs(part_pos - small_block.data.root_pos_w)
        # Collision condition: if any part of the robot is within the block's volume plus buffer.
        # CRITICAL RULE: All tensor operations correctly handle batched environments.
        collision_condition_obj1 = (dist_to_obj1[:, 0] < (small_block_half_dims[0] + buffer_distance)) & \
                                   (dist_to_obj1[:, 1] < (small_block_half_dims[1] + buffer_distance)) & \
                                   (dist_to_obj1[:, 2] < (small_block_half_dims[2] + buffer_distance))
        # Apply a large negative penalty for collision.
        # CRITICAL RULE: Rewards should be continuous and positive where possible. This is a discrete penalty for collision.
        collision_reward += torch.where(collision_condition_obj1, -10.0, 0.0)

        # Check against Medium Block (Object2)
        dist_to_obj2 = torch.abs(part_pos - medium_block.data.root_pos_w)
        collision_condition_obj2 = (dist_to_obj2[:, 0] < (medium_block_half_dims[0] + buffer_distance)) & \
                                   (dist_to_obj2[:, 1] < (medium_block_half_dims[1] + buffer_distance)) & \
                                   (dist_to_obj2[:, 2] < (medium_block_half_dims[2] + buffer_distance))
        # For Object2, which is the target, we still penalize penetration, but the main reward encourages proximity.
        # This ensures the robot doesn't try to phase through the block.
        collision_reward += torch.where(collision_condition_obj2, -10.0, 0.0)

        # Check against Large Block (Object3)
        dist_to_obj3 = torch.abs(part_pos - large_block.data.root_pos_w)
        collision_condition_obj3 = (dist_to_obj3[:, 0] < (large_block_half_dims[0] + buffer_distance)) & \
                                   (dist_to_obj3[:, 1] < (large_block_half_dims[1] + buffer_distance)) & \
                                   (dist_to_obj3[:, 2] < (large_block_half_dims[2] + buffer_distance))
        collision_reward += torch.where(collision_condition_obj3, -10.0, 0.0)

    # CRITICAL RULE: MANDATORY REWARD NORMALIZATION
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, collision_reward)
        RewNormalizer.update_stats(normaliser_name, collision_reward)
        return scaled_reward
    return collision_reward


def maintain_upright_posture_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "posture_reward") -> torch.Tensor:
    """
    Encourages the robot to maintain an upright and stable posture by penalizing large deviations
    of the pelvis's z-position from a target standing height.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot part using approved patterns
    robot = env.scene["robot"]
    # CRITICAL RULE: ALWAYS access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Target pelvis z-position for standing.
    # This value is a sensible standing height for the robot.
    # CRITICAL RULE: NEVER use hard-coded positions or arbitrary thresholds. This is a target height.
    target_pelvis_z = 0.7 # meters

    # Penalize deviation from target z-height.
    # CRITICAL RULE: ALL rewards MUST ONLY use relative distances between objects and robot parts.
    # This is a relative distance from a target Z-height.
    # A larger deviation means a larger negative reward, encouraging the robot to stay upright.
    # CRITICAL RULE: Rewards should be continuous and positive where possible. This is continuous.
    reward = -torch.abs(pelvis_pos_z - target_pelvis_z)

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
    """
    Configuration for the reward terms used in the walk_to_Medium_Block_for_robot_interaction skill.
    Defines the reward functions and their respective weights.
    """
    # Main reward for reaching the target horizontal distance from the Medium Block and maintaining posture.
    # Weight is 1.0 as it's the primary objective of the skill.
    # CRITICAL RULE: TaskRewardsCfg includes all reward components with appropriate weights.
    walk_to_Medium_Block_MainReward = RewTerm(func=walk_to_medium_block_main_reward, weight=1.0,
                                              params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for collision avoidance with all blocks.
    # Weight is 0.5 to provide a significant penalty for collisions without overshadowing the main goal.
    # CRITICAL RULE: TaskRewardsCfg includes all reward components with appropriate weights.
    CollisionAvoidanceReward = RewTerm(func=collision_avoidance_reward, weight=0.0,
                                       params={"normalise": True, "normaliser_name": "collision_reward"})

    # Shaping reward for maintaining an upright posture.
    # Weight is 0.2 to encourage stable walking without being too restrictive on movement.
    # CRITICAL RULE: TaskRewardsCfg includes all reward components with appropriate weights.
    MaintainUprightPostureReward = RewTerm(func=maintain_upright_posture_reward, weight=0.2,
                                          params={"normalise": True, "normaliser_name": "posture_reward"})