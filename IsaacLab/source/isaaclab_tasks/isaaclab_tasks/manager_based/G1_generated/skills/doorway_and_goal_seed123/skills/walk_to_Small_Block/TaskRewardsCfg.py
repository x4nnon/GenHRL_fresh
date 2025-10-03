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


def main_walk_to_small_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for walk_to_Small_Block.

    This reward encourages the robot's pelvis to reach the immediate vicinity of the Small Block (Object3).
    It considers alignment in x, y, and z dimensions, with a strong penalty for overshooting the y-target.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    # CRITICAL RULE: Access objects directly using env.scene['ObjectName']
    object3 = env.scene['Object3'] # Small Block
    object3_pos = object3.data.root_pos_w # CRITICAL RULE: Access object position using .data.root_pos_w

    # Access the required robot part(s)
    # CRITICAL RULE: Access robot parts using robot.body_names.index('part_name')
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CRITICAL RULE: Access robot part position using .data.body_pos_w

    # Object3 dimensions (from task description: 0.3m cubed)
    # CRITICAL RULE: Hardcode object dimensions from config, DO NOT access from object directly
    object3_size_y = 0.3

    # Target y-position: slightly before the block's y-center to be "next to" it
    # Assuming robot approaches from negative y towards positive y
    # Target is block's y-position minus half its depth, plus a small buffer
    # CRITICAL RULE: Use relative distances, not hard-coded positions
    target_y_pos = object3_pos[:, 1] - (object3_size_y / 2.0) - 0.1 # 0.1m buffer

    # Target x-position: align with the block's x-position
    target_x_pos = object3_pos[:, 0]

    # Target z-position: stable standing height
    # CRITICAL RULE: Z-height can be an absolute target if it represents a stable standing height
    target_z_pos = 0.7

    # Calculate distances
    # CRITICAL RULE: Use torch.abs for distances in specific dimensions
    distance_y = torch.abs(pelvis_pos[:, 1] - target_y_pos)
    distance_x = torch.abs(pelvis_pos[:, 0] - target_x_pos)
    distance_z = torch.abs(pelvis_pos[:, 2] - target_z_pos)

    # Reward for approaching target y-position (only if not past it)
    # Penalize heavily for overshooting the target y-position
    # CRITICAL RULE: Rewards must be continuous and smooth
    reward_y = -torch.where(pelvis_pos[:, 1] <= target_y_pos, distance_y, torch.abs(pelvis_pos[:, 1] - target_y_pos) * 5.0)

    # Reward for aligning with target x-position
    reward_x = -distance_x

    # Reward for maintaining stable z-height
    reward_z = -distance_z

    # Combine rewards with specified weights
    reward = reward_y * 0.6 + reward_x * 0.2 + reward_z * 0.2

    # CRITICAL RULE: Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def doorway_clearance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "doorway_clearance_reward") -> torch.Tensor:
    '''Shaping reward for passing through the doorway without collision.

    This reward penalizes the robot if its hands or pelvis get too close to the walls (Object1 and Object2)
    in the x-direction, encouraging safe passage through the doorway.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    object1 = env.scene['Object1'] # Heavy Cube (Wall 1)
    object2 = env.scene['Object2'] # Heavy Cube (Wall 2)
    object1_pos = object1.data.root_pos_w
    object2_pos = object2.data.root_pos_w

    # Access the required robot part(s)
    robot = env.scene["robot"]
    left_hand_idx = robot.body_names.index('left_palm_link')
    right_hand_idx = robot.body_names.index('right_palm_link')
    pelvis_idx = robot.body_names.index('pelvis')

    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Object1 and Object2 dimensions (from task description: x of 0.5m, y of 5m, z of 1.5m)
    # CRITICAL RULE: Hardcode object dimensions from config
    wall_width_x = 0.5 # This is the thickness of the wall in x-direction

    # Calculate the x-boundaries of the walls
    # Object1 (left wall) right edge: center_x + wall_width_x/2
    # Object2 (right wall) left edge: center_x - wall_width_x/2
    # CRITICAL RULE: Use relative positions to define boundaries
    object1_right_edge_x = object1_pos[:, 0] + (wall_width_x / 2.0)
    object2_left_edge_x = object2_pos[:, 0] - (wall_width_x / 2.0)

    # Define a safe clearance distance
    clearance = 0.1 # 10 cm buffer

    # Penalize if left hand or pelvis is too close to or past Object1 (left wall's right edge)
    # Reward is 0 if outside clearance, negative if inside or too close
    # CRITICAL RULE: Rewards must be continuous and smooth, using torch.clamp for thresholds
    # The clamp logic was inverted in the original code. It should be `value - threshold` to get positive if past threshold.
    # Corrected: `torch.clamp(threshold - value, min=0.0)` for penalizing when `value` is less than `threshold`.
    # Corrected: `torch.clamp(value - threshold, min=0.0)` for penalizing when `value` is greater than `threshold`.
    reward_obj1_left_hand = torch.where(left_hand_pos[:, 0] < object1_right_edge_x + clearance,
                                        -torch.clamp(object1_right_edge_x + clearance - left_hand_pos[:, 0], min=0.0) * 5.0,
                                        0.0)
    reward_obj1_pelvis = torch.where(pelvis_pos[:, 0] < object1_right_edge_x + clearance,
                                     -torch.clamp(object1_right_edge_x + clearance - pelvis_pos[:, 0], min=0.0) * 5.0,
                                     0.0)

    # Penalize if right hand or pelvis is too close to or past Object2 (right wall's left edge)
    reward_obj2_right_hand = torch.where(right_hand_pos[:, 0] > object2_left_edge_x - clearance,
                                         -torch.clamp(right_hand_pos[:, 0] - (object2_left_edge_x - clearance), min=0.0) * 5.0,
                                         0.0)
    reward_obj2_pelvis = torch.where(pelvis_pos[:, 0] > object2_left_edge_x - clearance,
                                     -torch.clamp(pelvis_pos[:, 0] - (object2_left_edge_x - clearance), min=0.0) * 5.0,
                                     0.0)

    reward = reward_obj1_left_hand + reward_obj1_pelvis + reward_obj2_right_hand + reward_obj2_pelvis

    # CRITICAL RULE: Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def posture_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "posture_stability_reward") -> torch.Tensor:
    '''Shaping reward for maintaining a stable standing posture.

    This reward encourages the robot to keep its feet on the ground and its pelvis at a reasonable height,
    promoting stable walking and a good state for subsequent skills.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot part(s)
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    left_foot_pos_z = robot.data.body_pos_w[:, left_foot_idx, 2]
    right_foot_pos_z = robot.data.body_pos_w[:, right_foot_idx, 2]
    pelvis_pos_z = robot.data.body_pos_w[:, pelvis_idx, 2]

    # Target z-height for feet (on the ground)
    # CRITICAL RULE: Z-height can be an absolute target if it represents a stable ground contact
    target_foot_z = 0.0 # Assuming ground is at z=0

    # Target z-height for pelvis (stable standing)
    target_pelvis_z = 0.7

    # Reward for feet being close to the ground (but not below)
    # Penalize if feet are too high or below ground
    # CRITICAL RULE: Use torch.abs for distances, continuous rewards
    reward_left_foot_z = -torch.abs(left_foot_pos_z - target_foot_z)
    reward_right_foot_z = -torch.abs(right_foot_pos_z - target_foot_z)

    # Reward for pelvis being at target height
    reward_pelvis_stability_z = -torch.abs(pelvis_pos_z - target_pelvis_z)

    # Combine rewards with specified weights
    reward = (reward_left_foot_z + reward_right_foot_z) * 0.5 + reward_pelvis_stability_z * 0.5

    # CRITICAL RULE: Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

@configclass
class TaskRewardsCfg:
    # Primary reward for reaching the small block
    # CRITICAL RULE: Main reward typically has a weight of 1.0
    MainWalkToSmallBlockReward = RewTerm(func=main_walk_to_small_block_reward, weight=1.0,
                                         params={"normalise": True, "normaliser_name": "main_walk_to_small_block_reward"})

    # Shaping reward for avoiding collisions with doorway walls
    # CRITICAL RULE: Shaping rewards typically have lower weights (e.g., 0.1-0.5)
    DoorwayClearanceReward = RewTerm(func=doorway_clearance_reward, weight=0.6,
                                     params={"normalise": True, "normaliser_name": "doorway_clearance_reward"})

    # Shaping reward for maintaining stable posture
    PostureStabilityReward = RewTerm(func=posture_stability_reward, weight=0.3,
                                     params={"normalise": True, "normaliser_name": "posture_stability_reward"})