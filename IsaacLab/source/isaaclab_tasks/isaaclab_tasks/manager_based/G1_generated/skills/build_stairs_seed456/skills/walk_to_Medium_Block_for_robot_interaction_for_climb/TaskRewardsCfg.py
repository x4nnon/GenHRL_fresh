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


def walk_to_medium_block_primary_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_medium_block_primary_reward") -> torch.Tensor:
    """
    Primary reward for positioning the robot directly in front of the Medium Block (Object2) for jumping.
    This reward encourages the robot's pelvis and feet to align with a target position relative to Object2.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    # CRITICAL RULE: Access objects directly using env.scene['ObjectName']
    object2 = env.scene['Object2'] # Medium Block for robot interaction

    # Access the required robot part(s)
    # CRITICAL RULE: Access robot parts using robot.body_names.index('part_name')
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    left_foot_pos_x = left_foot_pos[:, 0]
    left_foot_pos_y = left_foot_pos[:, 1]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    right_foot_pos_x = right_foot_pos[:, 0]
    right_foot_pos_y = right_foot_pos[:, 1]

    # Object2 dimensions (Medium Block: x=1m, y=1m, z=0.6m)
    # CRITICAL RULE: Hardcode object dimensions from config, DO NOT access from object attributes
    object2_x_dim = 1.0
    object2_y_dim = 1.0
    object2_z_dim = 0.6

    # Target position relative to Object2's center
    # Robot should be in front of Object2, centered in y, and at a suitable x distance for jumping
    # Assuming Object2's root_pos_w is its center.
    # Target x-position: Object2's x-center - (Object2_x_dim / 2) - desired_clearance
    # Desired clearance for jumping: e.g., 0.7m from the front face of the block
    # CRITICAL RULE: All rewards MUST ONLY use relative distances between objects and robot parts
    target_x_offset = (object2_x_dim / 2.0) + 0.7 # 0.5 + 0.7 = 1.2m from center of block
    target_pelvis_x = object2.data.root_pos_w[:, 0] - target_x_offset
    target_pelvis_y = object2.data.root_pos_w[:, 1] # Align with Object2's y-center
    target_pelvis_z = 0.7 # Stable standing height

    # Reward for pelvis x-position (approaching from positive x, stopping before overshooting)
    # The robot should be at target_pelvis_x.
    # CRITICAL RULE: Use smooth, continuous rewards. Negative absolute difference is continuous.
    reward_pelvis_x = -torch.abs(pelvis_pos_x - target_pelvis_x)

    # Reward for pelvis y-position (centering with Object2)
    reward_pelvis_y = -torch.abs(pelvis_pos_y - target_pelvis_y)

    # Reward for pelvis z-position (stable standing height)
    reward_pelvis_z = -torch.abs(pelvis_pos_z - target_pelvis_z)

    # Reward for feet x-position (similar to pelvis, ensuring feet are also correctly placed)
    # Average foot x-position
    avg_foot_x = (left_foot_pos_x + right_foot_pos_x) / 2.0
    reward_feet_x = -torch.abs(avg_foot_x - target_pelvis_x)

    # Reward for feet y-position (centering with Object2)
    avg_foot_y = (left_foot_pos_y + right_foot_pos_y) / 2.0
    reward_feet_y = -torch.abs(avg_foot_y - target_pelvis_y)

    # Combine rewards
    primary_reward = (reward_pelvis_x * 0.3 + reward_pelvis_y * 0.3 + reward_pelvis_z * 0.2 +
                      reward_feet_x * 0.1 + reward_feet_y * 0.1)

    # CRITICAL RULE: Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, primary_reward)
        RewNormalizer.update_stats(normaliser_name, primary_reward)
        return scaled_reward
    return primary_reward


def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward that penalizes collisions between the robot's main body parts (pelvis, head)
    and any of the blocks (Object1, Object2, Object3).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    # CRITICAL RULE: Access objects directly using env.scene['ObjectName']
    object1 = env.scene['Object1'] # Small Block
    object2 = env.scene['Object2'] # Medium Block
    object3 = env.scene['Object3'] # Large Block

    # Access the required robot part(s)
    # CRITICAL RULE: Access robot parts using robot.body_names.index('part_name')
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    head_idx = robot.body_names.index('head_link')
    head_pos = robot.data.body_pos_w[:, head_idx]

    # Object dimensions (approximate for collision check)
    # CRITICAL RULE: Hardcode object dimensions from config, DO NOT access from object attributes
    # Small Block: x=1m, y=1m, z=0.3m
    obj1_x_dim, obj1_y_dim, obj1_z_dim = 1.0, 1.0, 0.3
    # Medium Block: x=1m, y=1m, z=0.6m
    obj2_x_dim, obj2_y_dim, obj2_z_dim = 1.0, 1.0, 0.6
    # Large Block: x=1m, y=1m, z=0.9m
    obj3_x_dim, obj3_y_dim, obj3_z_dim = 1.0, 1.0, 0.9

    # Define a small buffer for collision detection
    collision_buffer = 0.1 # 10 cm buffer

    reward_collision = torch.zeros_like(pelvis_pos[:, 0]) # Initialize reward tensor for batch

    # Helper function to check collision for a part and an object
    def check_collision(part_pos, obj_pos, obj_dims, buffer):
        # CRITICAL RULE: All rewards MUST ONLY use relative distances between objects and robot parts
        dist_x = torch.abs(part_pos[:, 0] - obj_pos[:, 0])
        dist_y = torch.abs(part_pos[:, 1] - obj_pos[:, 1])
        dist_z = torch.abs(part_pos[:, 2] - obj_pos[:, 2])
        is_colliding = (dist_x < (obj_dims[0]/2.0 + buffer)) & \
                       (dist_y < (obj_dims[1]/2.0 + buffer)) & \
                       (dist_z < (obj_dims[2]/2.0 + buffer))
        # CRITICAL RULE: Use smooth, continuous rewards. Here, a penalty is applied when colliding.
        return torch.where(is_colliding, 0.5, 0.0) # Penalty for collision

    # Check collision with Object1 (Small Block)
    reward_collision -= check_collision(pelvis_pos, object1.data.root_pos_w, (obj1_x_dim, obj1_y_dim, obj1_z_dim), collision_buffer)
    reward_collision -= check_collision(head_pos, object1.data.root_pos_w, (obj1_x_dim, obj1_y_dim, obj1_z_dim), collision_buffer)

    # Check collision with Object2 (Medium Block)
    reward_collision -= check_collision(pelvis_pos, object2.data.root_pos_w, (obj2_x_dim, obj2_y_dim, obj2_z_dim), collision_buffer)
    reward_collision -= check_collision(head_pos, object2.data.root_pos_w, (obj2_x_dim, obj2_y_dim, obj2_z_dim), collision_buffer)

    # Check collision with Object3 (Large Block)
    reward_collision -= check_collision(pelvis_pos, object3.data.root_pos_w, (obj3_x_dim, obj3_y_dim, obj3_z_dim), collision_buffer)
    reward_collision -= check_collision(head_pos, object3.data.root_pos_w, (obj3_x_dim, obj3_y_dim, obj3_z_dim), collision_buffer)

    # CRITICAL RULE: Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward_collision)
        RewNormalizer.update_stats(normaliser_name, reward_collision)
        return scaled_reward
    return reward_collision


def forward_progress_and_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "forward_progress_and_stability_reward") -> torch.Tensor:
    """
    Shaping reward that encourages forward progress towards Object2 along the x-axis
    and maintains a stable, upright posture by penalizing large deviations in pelvis z-height.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    # CRITICAL RULE: Access objects directly using env.scene['ObjectName']
    object2 = env.scene['Object2'] # Medium Block

    # Access the required robot part(s)
    # CRITICAL RULE: Access robot parts using robot.body_names.index('part_name')
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Object2 dimensions (Medium Block: x=1m, y=1m, z=0.6m)
    # CRITICAL RULE: Hardcode object dimensions from config, DO NOT access from object attributes
    object2_x_dim = 1.0

    # Target approach region: Robot should be behind the front face of Object2 (relative to its center)
    # Front face of Object2 is at object2.data.root_pos_w[:, 0] - (object2_x_dim / 2.0)
    # We want the robot to approach from the positive x direction towards this face.
    # Condition: pelvis_pos_x > (object2.data.root_pos_w[:, 0] - (object2_x_dim / 2.0) - 0.1)
    # This means the robot is still "behind" or just at the front face, not having overshot.
    # A small buffer (0.1m) is added to allow the robot to get very close to the front face.
    # CRITICAL RULE: All rewards MUST ONLY use relative distances between objects and robot parts
    approach_condition = pelvis_pos_x > (object2.data.root_pos_w[:, 0] - (object2_x_dim / 2.0) - 0.1)

    # Reward for reducing x-distance to the front face of Object2, but only when approaching
    # The target x for this progress reward is slightly in front of the block's front face,
    # to encourage getting close but not overshooting.
    # Let's use the target_pelvis_x from the primary reward as the goal for this progress.
    target_x_offset = (object2_x_dim / 2.0) + 0.7 # 0.5 + 0.7 = 1.2m from center of block
    target_pelvis_x_for_progress = object2.data.root_pos_w[:, 0] - target_x_offset

    # Reward is positive when pelvis_pos_x is greater than target_pelvis_x_for_progress
    # and decreases as it approaches target_pelvis_x_for_progress.
    # This encourages movement in the negative x direction towards the target.
    # CRITICAL RULE: Use smooth, continuous rewards. Negative absolute difference is continuous.
    reward_forward_progress = -torch.abs(pelvis_pos_x - target_pelvis_x_for_progress)
    # Apply the approach condition: only reward progress if the robot is still "behind" the target x-position
    reward_forward_progress = torch.where(approach_condition, reward_forward_progress, torch.tensor(0.0, device=env.device))

    # Reward for maintaining pelvis z-height near 0.7m (stability)
    target_pelvis_z = 0.7
    # CRITICAL RULE: Use smooth, continuous rewards. Negative absolute difference is continuous.
    reward_pelvis_stability_z = -torch.abs(pelvis_pos_z - target_pelvis_z)

    shaping_reward2 = reward_forward_progress * 0.5 + reward_pelvis_stability_z * 0.5

    # CRITICAL RULE: Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward2)
        RewNormalizer.update_stats(normaliser_name, shaping_reward2)
        return scaled_reward
    return shaping_reward2


@configclass
class TaskRewardsCfg:
    # CRITICAL RULE: Main reward with weight 1.0
    primary_walk_to_medium_block = RewTerm(func=walk_to_medium_block_primary_reward, weight=1.0,
                                           params={"normalise": True, "normaliser_name": "walk_to_medium_block_primary_reward"})

    # CRITICAL RULE: Supporting rewards with lower weights
    collision_avoidance = RewTerm(func=collision_avoidance_reward, weight=0.4,
                                  params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})

    forward_progress_and_stability = RewTerm(func=forward_progress_and_stability_reward, weight=0.3,
                                              params={"normalise": True, "normaliser_name": "forward_progress_and_stability_reward"})