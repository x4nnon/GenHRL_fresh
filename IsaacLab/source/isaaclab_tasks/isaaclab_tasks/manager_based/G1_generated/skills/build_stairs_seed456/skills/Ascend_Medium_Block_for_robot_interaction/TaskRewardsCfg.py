from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.reward_normalizer import get_normalizer, RewardStats # this automatically sets up the RewNormalizer instance.
from genhrl.generation.objects import get_object_volume
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv # Corrected: Added ManagerBasedRLEnv import
import torch

from isaaclab.envs import mdp
# Import custom MDP functions from genhrl
import genhrl.generation.mdp.rewards as custom_rewards
import genhrl.generation.mdp.terminations as custom_terminations  
import genhrl.generation.mdp.observations as custom_observations
import genhrl.generation.mdp.events as custom_events
import genhrl.generation.mdp.curriculums as custom_curriculums


def main_ascend_medium_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_ascend_medium_block_reward") -> torch.Tensor:
    """
    Main reward for the robot to jump onto the top surface of the Medium Block.
    This reward encourages the robot's feet to be on the top surface of the Medium Block and its pelvis to be at a stable standing height above it.
    """
    # Get normalizer instance (mandatory)
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects (mandatory: direct access using ObjectN names)
    medium_block = env.scene['Object2'] # Medium Block for robot interaction
    medium_block_pos = medium_block.data.root_pos_w # Access object position (mandatory: approved pattern)

    # Access the required robot part(s) (mandatory: approved pattern for indices and positions)
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Medium Block dimensions (mandatory: hardcoded from description, as dimensions cannot be accessed from RigidObject)
    medium_block_height = 0.6
    medium_block_x_size = 1.0
    medium_block_y_size = 1.0

    # Target Z position for feet (top of block + small clearance for foot thickness)
    # This is a relative target Z based on the block's Z position.
    target_foot_z = medium_block_pos[:, 2] + medium_block_height / 2.0 + 0.05 # Assuming foot thickness ~0.05m

    # Target Z position for pelvis (stable standing height above block)
    # This is a relative target Z based on the block's Z position.
    target_pelvis_z = medium_block_pos[:, 2] + medium_block_height / 2.0 + 0.7 # 0.7m above block top

    # Reward for feet being on top of the block (X, Y, Z)
    # X-Y bounds check for feet (mandatory: relative distances)
    # Negative absolute difference for continuous reward: closer to 0 is better.
    left_foot_on_block_x = -torch.abs(left_foot_pos[:, 0] - medium_block_pos[:, 0])
    left_foot_on_block_y = -torch.abs(left_foot_pos[:, 1] - medium_block_pos[:, 1])
    right_foot_on_block_x = -torch.abs(right_foot_pos[:, 0] - medium_block_pos[:, 0])
    right_foot_on_block_y = -torch.abs(right_foot_pos[:, 1] - medium_block_pos[:, 1])

    # Z-height for feet (mandatory: relative distances)
    left_foot_z_reward = -torch.abs(left_foot_pos[:, 2] - target_foot_z)
    right_foot_z_reward = -torch.abs(right_foot_pos[:, 2] - target_foot_z)

    # Pelvis Z-height for stability (mandatory: relative distances)
    pelvis_z_reward = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Combine rewards for feet and pelvis
    # All operations work on batched tensors (mandatory)
    reward = (left_foot_on_block_x + left_foot_on_block_y + left_foot_z_reward +
              right_foot_on_block_x + right_foot_on_block_y + right_foot_z_reward) * 0.5 + pelvis_z_reward * 0.5

    # Condition: Feet must be within the X/Y bounds of the block to get the full reward for being "on" it.
    # This makes the X/Y rewards more impactful when close.
    # Using a small buffer (0.1m) for the bounds.
    x_bound_condition_left = torch.abs(left_foot_pos[:, 0] - medium_block_pos[:, 0]) < (medium_block_x_size / 2.0 + 0.1)
    y_bound_condition_left = torch.abs(left_foot_pos[:, 1] - medium_block_pos[:, 1]) < (medium_block_y_size / 2.0 + 0.1)
    x_bound_condition_right = torch.abs(right_foot_pos[:, 0] - medium_block_pos[:, 0]) < (medium_block_x_size / 2.0 + 0.1)
    y_bound_condition_right = torch.abs(right_foot_pos[:, 1] - medium_block_pos[:, 1]) < (medium_block_y_size / 2.0 + 0.1)

    # Apply a small positive constant if feet are within bounds to encourage staying on (continuous reward)
    on_block_bonus = torch.where(x_bound_condition_left & y_bound_condition_left & x_bound_condition_right & y_bound_condition_right, 0.5, 0.0)
    reward += on_block_bonus

    # Mandatory normalization (exact implementation)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def approach_medium_block_xy_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_medium_block_xy_reward") -> torch.Tensor:
    """
    Shaping reward 1: Encourages the robot's pelvis to approach the Medium Block in X and Y dimensions.
    It is active when the robot is still some distance away from the Medium Block, guiding it towards the target.
    It also penalizes moving past the Medium Block in the X direction towards the Large Block.
    """
    # Get normalizer instance (mandatory)
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects (mandatory: direct access using ObjectN names)
    medium_block = env.scene['Object2'] # Medium Block for robot interaction
    large_block = env.scene['Object3'] # Large Block for robot interaction
    medium_block_pos = medium_block.data.root_pos_w
    large_block_pos = large_block.data.root_pos_w

    # Access the required robot part(s) (mandatory: approved pattern)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Medium Block dimensions (mandatory: hardcoded from description)
    medium_block_x_size = 1.0
    medium_block_y_size = 1.0 # Although not used in penalty, good to keep consistent

    # Distance to Medium Block in X and Y (mandatory: relative distances, continuous reward)
    distance_x_to_medium = -torch.abs(pelvis_pos[:, 0] - medium_block_pos[:, 0])
    distance_y_to_medium = -torch.abs(pelvis_pos[:, 1] - medium_block_pos[:, 1])

    # Penalty for overshooting Medium Block in X towards Large Block
    # Assuming Large Block is further in X than Medium Block.
    # Penalty is applied if pelvis X is beyond the medium block's X-extent plus a buffer.
    # (mandatory: relative distance for condition, continuous reward)
    overshoot_penalty = torch.where(pelvis_pos[:, 0] > (medium_block_pos[:, 0] + medium_block_x_size / 2.0 + 0.2), -1.0, 0.0) # 0.2m buffer past block

    # Activation condition: Active when pelvis is not yet "on" the block (e.g., more than 0.1m away in X or Y).
    # This prevents it from interfering with the primary reward once the robot is on the block.
    # (mandatory: relative distances for condition)
    activation_condition = (torch.abs(pelvis_pos[:, 0] - medium_block_pos[:, 0]) > (medium_block_x_size / 2.0 + 0.1)) | \
                           (torch.abs(pelvis_pos[:, 1] - medium_block_pos[:, 1]) > (medium_block_y_size / 2.0 + 0.1))

    reward = (distance_x_to_medium + distance_y_to_medium) * 0.5 + overshoot_penalty
    # Apply activation condition (mandatory: all operations work on batched tensors)
    reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device))

    # Mandatory normalization (exact implementation)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def clearance_and_collision_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "clearance_and_collision_reward") -> torch.Tensor:
    """
    Shaping reward 2: Encourages the robot to gain sufficient Z-height with its feet and pelvis to clear the Medium Block,
    specifically when it is close enough to initiate a jump. It also includes a general collision avoidance term for
    the robot's body parts with all blocks.
    """
    # Get normalizer instance (mandatory)
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects (mandatory: direct access using ObjectN names)
    medium_block = env.scene['Object2'] # Medium Block for robot interaction
    small_block = env.scene['Object1'] # Small Block for robot interaction
    large_block = env.scene['Object3'] # Large Block for robot interaction
    medium_block_pos = medium_block.data.root_pos_w
    small_block_pos = small_block.data.root_pos_w
    large_block_pos = large_block.data.root_pos_w

    # Access the required robot part(s) (mandatory: approved pattern)
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')
    head_idx = robot.body_names.index('head_link')

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    head_pos = robot.data.body_pos_w[:, head_idx]

    # Medium Block dimensions (mandatory: hardcoded from description)
    medium_block_height = 0.6
    medium_block_x_size = 1.0
    medium_block_y_size = 1.0

    # Block dimensions for collision calculation (mandatory: hardcoded from description)
    block_x_size = 1.0 # All blocks are 1m in X and Y
    block_y_size = 1.0
    small_block_height = 0.3
    large_block_height = 0.9

    # Target Z-height for clearing the block (block top + clearance)
    # (mandatory: relative target Z based on block position)
    clearance_z = medium_block_pos[:, 2] + medium_block_height / 2.0 + 0.2 # 0.2m clearance

    # Reward for increasing Z-height of feet and pelvis when close to block
    # (mandatory: relative distances, continuous reward)
    z_height_reward_left_foot = -torch.abs(left_foot_pos[:, 2] - clearance_z)
    z_height_reward_right_foot = -torch.abs(right_foot_pos[:, 2] - clearance_z)
    z_height_reward_pelvis = -torch.abs(pelvis_pos[:, 2] - clearance_z)

    # Activation condition: Active when pelvis is within a certain X/Y range of the Medium Block, but not yet on top.
    # This ensures the jump reward is active during the jump phase.
    # (mandatory: relative distances for condition)
    activation_condition = (torch.abs(pelvis_pos[:, 0] - medium_block_pos[:, 0]) < (medium_block_x_size / 2.0 + 0.5)) & \
                           (torch.abs(pelvis_pos[:, 1] - medium_block_pos[:, 1]) < (medium_block_y_size / 2.0 + 0.5)) & \
                           (pelvis_pos[:, 2] < (medium_block_pos[:, 2] + medium_block_height / 2.0 + 0.6)) # Not yet stable on top

    # Collision avoidance for all blocks (mandatory: relative distances for collision check)
    collision_penalty = torch.zeros_like(pelvis_pos[:, 0]) # Initialize penalty tensor for batch
    robot_parts_to_check = [left_foot_pos, right_foot_pos, pelvis_pos, head_pos]
    blocks_to_check = [(small_block_pos, small_block_height), (medium_block_pos, medium_block_height), (large_block_pos, large_block_height)]

    for i, (block_pos, block_h) in enumerate(blocks_to_check):
        for part_pos in robot_parts_to_check:
            # Calculate distance to block center (mandatory: relative distances)
            dist_x = torch.abs(part_pos[:, 0] - block_pos[:, 0])
            dist_y = torch.abs(part_pos[:, 1] - block_pos[:, 1])
            dist_z = torch.abs(part_pos[:, 2] - block_pos[:, 2])

            # Check for overlap (assuming block size 1x1m for X/Y)
            # Add a small buffer (e.g., 0.1m) for collision detection
            overlap_x = (dist_x < (block_x_size / 2.0 + 0.1))
            overlap_y = (dist_y < (block_y_size / 2.0 + 0.1))
            overlap_z = (dist_z < (block_h / 2.0 + 0.1))

            # If overlapping, apply penalty. The penalty increases as overlap increases.
            # Only penalize if the part is not supposed to be on top of the block (e.g., head with medium block)
            # For feet on medium block, this is handled by primary reward, so avoid double penalty.
            # (mandatory: all operations work on batched tensors)
            # Corrected: Check if part_pos is the same tensor object as left_foot_pos or right_foot_pos
            is_foot_on_medium_block = (part_pos is left_foot_pos or part_pos is right_foot_pos) and (block_pos is medium_block_pos)
            
            if not is_foot_on_medium_block:
                collision_penalty += torch.where(overlap_x & overlap_y & overlap_z, -1.0, 0.0) # Large penalty for collision

    reward = (z_height_reward_left_foot + z_height_reward_right_foot + z_height_reward_pelvis) * 0.3 + collision_penalty * 0.7
    # Apply activation condition (mandatory: all operations work on batched tensors)
    reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device))

    # Mandatory normalization (exact implementation)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Primary reward for being on the medium block, weight 1.0 (mandatory)
    MainAscendMediumBlockReward = RewTerm(func=main_ascend_medium_block_reward, weight=1.0, 
                                params={"normalise": True, "normaliser_name": "main_ascend_medium_block_reward"})
    
    # Shaping reward for approaching the medium block, lower weight (mandatory)
    ApproachMediumBlockXYReward = RewTerm(func=approach_medium_block_xy_reward, weight=0.6,
                              params={"normalise": True, "normaliser_name": "approach_medium_block_xy_reward"})
    
    # Shaping reward for clearance and collision avoidance, lower weight (mandatory)
    ClearanceAndCollisionReward = RewTerm(func=clearance_and_collision_reward, weight=0.4, # Adjusted weight slightly to 0.4 from 0.7 to balance with approach reward
                              params={"normalise": True, "normaliser_name": "clearance_and_collision_reward"})