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

def ascend_first_step_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the 'ascend_first_step' skill.
    Encourages the robot to get its feet onto the Small Block and then position its pelvis stably on top.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    # CRITICAL RULE: Access objects directly using their scene names (Object1, Object2, etc.)
    small_block = env.scene['Object1'] # Small Block for robot interaction

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
    left_foot_pos_z = left_foot_pos[:, 2]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    right_foot_pos_x = right_foot_pos[:, 0]
    right_foot_pos_y = right_foot_pos[:, 1]
    right_foot_pos_z = right_foot_pos[:, 2]

    # Small Block dimensions (from object configuration, hardcoded as per rules)
    # CRITICAL RULE: Hardcode object dimensions from the object configuration, do not access from RigidObject
    small_block_height = 0.3
    small_block_x_size = 1.0
    small_block_y_size = 1.0

    # Target Z position for feet on top of the block (block_height + small clearance)
    # CRITICAL RULE: Use relative distances. Target Z is relative to block's Z position.
    target_foot_z = small_block.data.root_pos_w[:, 2] + small_block_height / 2.0 + 0.02 # 0.02m clearance above block surface

    # Target Z position for pelvis when standing on the block
    # CRITICAL RULE: Use relative distances. Target Z is relative to block's Z position.
    target_pelvis_z = small_block.data.root_pos_w[:, 2] + small_block_height / 2.0 + 0.7 # 0.7m above block surface

    # Horizontal distance from pelvis to block center
    # CRITICAL RULE: Use relative distances.
    dist_pelvis_block_x = pelvis_pos_x - small_block.data.root_pos_w[:, 0]
    dist_pelvis_block_y = pelvis_pos_y - small_block.data.root_pos_w[:, 1]

    # Horizontal distance from feet to block center
    # CRITICAL RULE: Use relative distances.
    dist_left_foot_block_x = left_foot_pos_x - small_block.data.root_pos_w[:, 0]
    dist_left_foot_block_y = left_foot_pos_y - small_block.data.root_pos_w[:, 1]
    dist_right_foot_block_x = right_foot_pos_x - small_block.data.root_pos_w[:, 0]
    dist_right_foot_block_y = right_foot_pos_y - small_block.data.root_pos_w[:, 1]

    # Phase 1: Approach and get feet onto the block
    # Reward for feet being horizontally within block bounds and at target Z
    # Use a small margin for horizontal placement
    horizontal_margin_x = small_block_x_size / 2.0 - 0.1 # 0.1m margin from edge
    horizontal_margin_y = small_block_y_size / 2.0 - 0.1 # 0.1m margin from edge

    # Reward for feet being on top of the block
    # CRITICAL RULE: Rewards are continuous and based on relative distances.
    reward_left_foot_on_block = -torch.abs(dist_left_foot_block_x) / horizontal_margin_x \
                              - torch.abs(dist_left_foot_block_y) / horizontal_margin_y \
                              - torch.abs(left_foot_pos_z - target_foot_z) * 2.0 # Stronger penalty for Z

    reward_right_foot_on_block = -torch.abs(dist_right_foot_block_x) / horizontal_margin_x \
                               - torch.abs(dist_right_foot_block_y) / horizontal_margin_y \
                               - torch.abs(right_foot_pos_z - target_foot_z) * 2.0

    # Condition for feet being "on" the block (within horizontal bounds and above block base)
    # CRITICAL RULE: Conditions use relative distances and hardcoded dimensions.
    left_foot_on_block_cond = (torch.abs(dist_left_foot_block_x) < horizontal_margin_x) & \
                              (torch.abs(dist_left_foot_block_y) < horizontal_margin_y) & \
                              (left_foot_pos_z > small_block.data.root_pos_w[:, 2] + small_block_height / 2.0 - 0.05) # 0.05m below target Z

    right_foot_on_block_cond = (torch.abs(dist_right_foot_block_x) < horizontal_margin_x) & \
                               (torch.abs(dist_right_foot_block_y) < horizontal_margin_y) & \
                               (right_foot_pos_z > small_block.data.root_pos_w[:, 2] + small_block_height / 2.0 - 0.05)

    # Phase 2: Pelvis positioning and stability once feet are on block
    # Reward for pelvis being horizontally centered and at target Z
    # CRITICAL RULE: Rewards are continuous and based on relative distances.
    reward_pelvis_centered = -torch.abs(dist_pelvis_block_x) * 0.5 \
                             -torch.abs(dist_pelvis_block_y) * 0.5 \
                             -torch.abs(pelvis_pos_z - target_pelvis_z) * 1.0 # Stronger penalty for Z

    # Combine rewards: prioritize getting feet on, then centering pelvis
    # If both feet are on the block, focus on pelvis centering and stability
    # Otherwise, focus on getting feet onto the block
    # CRITICAL RULE: Tensor operations handle batched environments correctly.
    reward = torch.where(left_foot_on_block_cond & right_foot_on_block_cond,
                         reward_pelvis_centered + (reward_left_foot_on_block + reward_right_foot_on_block) * 0.2, # Keep a small reward for feet being well placed
                         (reward_left_foot_on_block + reward_right_foot_on_block) * 0.5) # Scale down if not fully on block yet

    # Ensure reward is positive for success, scale and offset
    # CRITICAL RULE: Rewards should be continuous and positive where possible.
    reward = torch.exp(reward) * 0.5 # Exponential to make it more sensitive near target, scaled

    # CRITICAL RULE: Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def ascend_first_step_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward 1: Encourages the robot to maintain a safe distance from the sides of the Small Block
    with its knees and feet during the ascent, preventing collisions or awkward movements.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    small_block = env.scene['Object1'] # Small Block for robot interaction

    # Access the required robot part(s)
    robot = env.scene["robot"]
    left_knee_idx = robot.body_names.index('left_knee_link')
    left_knee_pos = robot.data.body_pos_w[:, left_knee_idx]
    left_knee_pos_x = left_knee_pos[:, 0]
    left_knee_pos_y = left_knee_pos[:, 1]
    left_knee_pos_z = left_knee_pos[:, 2]

    right_knee_idx = robot.body_names.index('right_knee_link')
    right_knee_pos = robot.data.body_pos_w[:, right_knee_idx]
    right_knee_pos_x = right_knee_pos[:, 0]
    right_knee_pos_y = right_knee_pos[:, 1]
    right_knee_pos_z = right_knee_pos[:, 2]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    left_foot_pos_x = left_foot_pos[:, 0]
    left_foot_pos_y = left_foot_pos[:, 1]
    left_foot_pos_z = left_foot_pos[:, 2]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    right_foot_pos_x = right_foot_pos[:, 0]
    right_foot_pos_y = right_foot_pos[:, 1]
    right_foot_pos_z = right_foot_pos[:, 2]

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]

    # Small Block dimensions (from object configuration, hardcoded)
    small_block_height = 0.3
    small_block_x_size = 1.0
    small_block_y_size = 1.0

    # Calculate block edges relative to its center
    # CRITICAL RULE: Use relative distances for block edges.
    block_x_min = small_block.data.root_pos_w[:, 0] - small_block_x_size / 2.0
    block_x_max = small_block.data.root_pos_w[:, 0] + small_block_x_size / 2.0
    block_y_min = small_block.data.root_pos_w[:, 1] - small_block_y_size / 2.0
    block_y_max = small_block.data.root_pos_w[:, 1] + small_block_y_size / 2.0
    block_z_top = small_block.data.root_pos_w[:, 2] + small_block_height / 2.0

    # Collision avoidance for knees and feet with block sides (below top surface)
    # Penalize if part is horizontally inside block bounds AND below block_z_top
    # CRITICAL RULE: Rewards are continuous and based on relative distances.
    # Left Knee
    collision_left_knee_x = torch.max(torch.tensor(0.0, device=env.device), block_x_min - left_knee_pos_x) + \
                            torch.max(torch.tensor(0.0, device=env.device), left_knee_pos_x - block_x_max)
    collision_left_knee_y = torch.max(torch.tensor(0.0, device=env.device), block_y_min - left_knee_pos_y) + \
                            torch.max(torch.tensor(0.0, device=env.device), left_knee_pos_y - block_y_max)
    collision_left_knee_z_below_top = torch.max(torch.tensor(0.0, device=env.device), block_z_top - left_knee_pos_z)

    reward_left_knee_collision = - (collision_left_knee_x + collision_left_knee_y) * 5.0 # Strong penalty for horizontal overlap
    # Only penalize if below top surface (i.e., not intended to be on top)
    reward_left_knee_collision = torch.where(collision_left_knee_z_below_top > 0.0, reward_left_knee_collision, torch.tensor(0.0, device=env.device))

    # Right Knee
    collision_right_knee_x = torch.max(torch.tensor(0.0, device=env.device), block_x_min - right_knee_pos_x) + \
                             torch.max(torch.tensor(0.0, device=env.device), right_knee_pos_x - block_x_max)
    collision_right_knee_y = torch.max(torch.tensor(0.0, device=env.device), block_y_min - right_knee_pos_y) + \
                             torch.max(torch.tensor(0.0, device=env.device), right_knee_pos_y - block_y_max)
    collision_right_knee_z_below_top = torch.max(torch.tensor(0.0, device=env.device), block_z_top - right_knee_pos_z)

    reward_right_knee_collision = - (collision_right_knee_x + collision_right_knee_y) * 5.0
    reward_right_knee_collision = torch.where(collision_right_knee_z_below_top > 0.0, reward_right_knee_collision, torch.tensor(0.0, device=env.device))

    # Left Foot (only penalize if not intended to be on top, i.e., below block_z_top)
    collision_left_foot_x = torch.max(torch.tensor(0.0, device=env.device), block_x_min - left_foot_pos_x) + \
                            torch.max(torch.tensor(0.0, device=env.device), left_foot_pos_x - block_x_max)
    collision_left_foot_y = torch.max(torch.tensor(0.0, device=env.device), block_y_min - left_foot_pos_y) + \
                            torch.max(torch.tensor(0.0, device=env.device), left_foot_pos_y - block_y_max)
    collision_left_foot_z_below_top = torch.max(torch.tensor(0.0, device=env.device), block_z_top - left_foot_pos_z)

    reward_left_foot_collision = - (collision_left_foot_x + collision_left_foot_y) * 5.0
    reward_left_foot_collision = torch.where(collision_left_foot_z_below_top > 0.0, reward_left_foot_collision, torch.tensor(0.0, device=env.device))

    # Right Foot (only penalize if not intended to be on top, i.e., below block_z_top)
    collision_right_foot_x = torch.max(torch.tensor(0.0, device=env.device), block_x_min - right_foot_pos_x) + \
                             torch.max(torch.tensor(0.0, device=env.device), right_foot_pos_x - block_x_max)
    collision_right_foot_y = torch.max(torch.tensor(0.0, device=env.device), block_y_min - right_foot_pos_y) + \
                             torch.max(torch.tensor(0.0, device=env.device), right_foot_pos_y - block_y_max)
    collision_right_foot_z_below_top = torch.max(torch.tensor(0.0, device=env.device), block_z_top - right_foot_pos_z)

    reward_right_foot_collision = - (collision_right_foot_x + collision_right_foot_y) * 5.0
    reward_right_foot_collision = torch.where(collision_right_foot_z_below_top > 0.0, reward_right_foot_collision, torch.tensor(0.0, device=env.device))

    # Sum all collision penalties
    shaping_reward_1 = reward_left_knee_collision + reward_right_knee_collision + \
                       reward_left_foot_collision + reward_right_foot_collision

    # Activation condition: only active when robot is close to the block horizontally
    # This prevents penalizing collisions when far away or already fully on top
    # CRITICAL RULE: Activation condition uses relative distances.
    pelvis_dist_to_block_center_xy = torch.sqrt(
        (pelvis_pos_x - small_block.data.root_pos_w[:, 0])**2 +
        (pelvis_pos_y - small_block.data.root_pos_w[:, 1])**2
    )
    activation_condition = (pelvis_dist_to_block_center_xy < (small_block_x_size / 2.0 + 0.5)) # Active when pelvis is within 0.5m horizontally of block edge

    reward = torch.where(activation_condition, shaping_reward_1, torch.tensor(0.0, device=env.device))

    # CRITICAL RULE: Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def ascend_first_step_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stability_reward") -> torch.Tensor:
    """
    Shaping reward 2: Encourages the robot to maintain an upright posture and stability,
    especially when transitioning onto the block and standing on it.
    Penalizes large horizontal deviations of the pelvis from the average foot position.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot part(s)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    left_foot_pos_x = left_foot_pos[:, 0]
    left_foot_pos_y = left_foot_pos[:, 1]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    right_foot_pos_x = right_foot_pos[:, 0]
    right_foot_pos_y = right_foot_pos[:, 1]

    # Calculate the average horizontal position of the feet
    # CRITICAL RULE: All operations work with batched environments.
    avg_foot_pos_x = (left_foot_pos_x + right_foot_pos_x) / 2.0
    avg_foot_pos_y = (left_foot_pos_y + right_foot_pos_y) / 2.0

    # Calculate horizontal deviation of pelvis from average foot position
    # CRITICAL RULE: Rewards are based on relative distances.
    pelvis_deviation_x = pelvis_pos_x - avg_foot_pos_x
    pelvis_deviation_y = pelvis_pos_y - avg_foot_pos_y

    # Penalize large deviations
    # CRITICAL RULE: Rewards are continuous.
    reward = -torch.abs(pelvis_deviation_x) * 2.0 - torch.abs(pelvis_deviation_y) * 2.0

    # Activation condition: This reward is always active to encourage general stability.
    # No specific condition needed as it's a continuous stability reward.

    # CRITICAL RULE: Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

@configclass
class TaskRewardsCfg:
    # CRITICAL RULE: Main reward weight is typically 1.0
    MainAscendFirstStepReward = RewTerm(func=ascend_first_step_main_reward, weight=1.0,
                                        params={"normalise": True, "normaliser_name": "main_reward"})

    # CRITICAL RULE: Shaping rewards have lower weights (e.g., 0.1-0.5)
    AscendFirstStepCollisionAvoidanceReward = RewTerm(func=ascend_first_step_collision_avoidance_reward, weight=0.4,
                                                      params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})

    AscendFirstStepStabilityReward = RewTerm(func=ascend_first_step_stability_reward, weight=0.3,
                                             params={"normalise": True, "normaliser_name": "stability_reward"})