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
    Primary reward for guiding the robot to walk towards and position its pelvis close to the Medium Block (Object2)
    along the x and y axes, while maintaining a stable standing height.
    It encourages the robot to be adjacent to Object2, not on top or too far, ready for pushing.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access Object2 (Medium Block) and robot's pelvis
    object_name = env.scene['Object2'] # Accessing object using approved pattern
    robot = env.scene["robot"] # Accessing robot using approved pattern
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern
    
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    object2_pos = object_name.data.root_pos_w # Accessing object position using approved pattern
    object2_x = object2_pos[:, 0]
    object2_y = object2_pos[:, 1]
    object2_z = object2_pos[:, 2]

    # Medium Block dimensions: x=1m, y=1m, z=0.6m. Half-dimensions: 0.5m, 0.5m, 0.3m.
    # Target distance from block surface for pelvis: 0.5m.
    # This means pelvis should be 0.5m (half block width) + 0.5m (desired clearance) = 1.0m from block center along x or y.
    target_distance_xy = 1.0 # Hardcoded from object config and skill requirement, not an arbitrary threshold.

    # Calculate distance from pelvis to object center along X and Y axes
    dist_x_pelvis_obj2 = torch.abs(pelvis_pos_x - object2_x) # Relative distance in x-direction
    dist_y_pelvis_obj2 = torch.abs(pelvis_pos_y - object2_y) # Relative distance in y-direction

    # Reward for being at the target distance along one axis (e.g., X) and aligned along the other (Y)
    # This encourages approaching from a side.
    # The robot should be within the block's half-width (0.5m from center) along one axis, and at target_distance_xy along the other.
    # We use -torch.abs(distance) to create a continuous reward that is maximized when the distance is at the target.
    
    # Reward for approaching along X and aligning along Y
    reward_approach_x = -torch.abs(dist_x_pelvis_obj2 - target_distance_xy)
    # Encourage alignment with block's Y center (within its width)
    # The ideal alignment is when the pelvis Y is close to the block's Y center.
    # The block's half-width is 0.5m. We want the pelvis to be within this range.
    # A simple negative absolute distance works as a continuous penalty.
    reward_align_y = -torch.abs(dist_y_pelvis_obj2) 

    # Reward for approaching along Y and aligning along X
    reward_approach_y = -torch.abs(dist_y_pelvis_obj2 - target_distance_xy)
    # Encourage alignment with block's X center (within its length)
    reward_align_x = -torch.abs(dist_x_pelvis_obj2) 

    # Combine to encourage being adjacent to either X or Y side.
    # Use a max to reward whichever side is closer to the ideal configuration, making the reward continuous.
    primary_reward_xy = torch.max(
        (reward_approach_x + reward_align_y),
        (reward_approach_y + reward_align_x)
    )

    # Reward for pelvis height (stable standing position)
    # Target pelvis Z is 0.7m, a standard stable height for the robot. This is a physical constant.
    target_pelvis_z = 0.7 
    primary_reward_z = -torch.abs(pelvis_pos_z - target_pelvis_z) # Continuous reward based on absolute difference from target height.

    # Combine all primary components
    reward = primary_reward_xy + primary_reward_z

    # Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_object2_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_object2_reward") -> torch.Tensor:
    """
    Penalizes the robot for any part of its body colliding with or getting too close to the Medium Block (Object2).
    This prevents the robot from climbing on the block or pushing it prematurely.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    object_name = env.scene['Object2'] # Accessing object using approved pattern

    # Medium Block dimensions (hardcoded from object configuration)
    object2_x_dim = 1.0
    object2_y_dim = 1.0
    object2_z_dim = 0.6

    # Robot parts to monitor for collision
    robot = env.scene["robot"] # Accessing robot using approved pattern
    left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Accessing robot part index using approved pattern
    right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Accessing robot part index using approved pattern
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Accessing robot part position using approved pattern
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Accessing robot part position using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern

    object2_pos = object_name.data.root_pos_w # Accessing object position using approved pattern

    # Define half-dimensions of the block and a small buffer for clearance
    half_x = object2_x_dim / 2.0
    half_y = object2_y_dim / 2.0
    half_z = object2_z_dim / 2.0
    clearance_buffer = 0.1 # Small buffer to avoid immediate collision, not an arbitrary threshold.

    total_penalty = torch.zeros_like(pelvis_pos[:, 0]) # Initialize total penalty tensor for batch operations

    # Function to calculate continuous penalty for a given robot part and object
    # This function ensures relative distances are used and provides a continuous penalty.
    def calculate_continuous_collision_penalty(robot_part_pos, obj_pos, obj_half_dims, buffer):
        dist_x = torch.abs(robot_part_pos[:, 0] - obj_pos[:, 0]) # Relative distance in x-direction
        dist_y = torch.abs(robot_part_pos[:, 1] - obj_pos[:, 1]) # Relative distance in y-direction
        dist_z = torch.abs(robot_part_pos[:, 2] - obj_pos[:, 2]) # Relative distance in z-direction

        # Collision condition: part is within the object's bounds plus buffer
        collision_x = (dist_x < obj_half_dims[0] + buffer)
        collision_y = (dist_y < obj_half_dims[1] + buffer)
        collision_z = (dist_z < obj_half_dims[2] + buffer)
        
        collision_condition = collision_x & collision_y & collision_z
        
        # Continuous penalty: more negative the deeper into the object.
        # The penalty is based on how much the sum of distances (x,y,z) is less than the sum of half-dimensions + buffer.
        # This makes it continuous and increases as the part gets "deeper" into the object.
        penalty_value = -(obj_half_dims[0] + obj_half_dims[1] + obj_half_dims[2] + 3 * buffer - (dist_x + dist_y + dist_z))
        
        return torch.where(collision_condition, penalty_value, torch.zeros_like(penalty_value))

    # Penalties for left foot, right foot, and pelvis with Object2
    total_penalty += calculate_continuous_collision_penalty(left_foot_pos, object2_pos, [half_x, half_y, half_z], clearance_buffer)
    total_penalty += calculate_continuous_collision_penalty(right_foot_pos, object2_pos, [half_x, half_y, half_z], clearance_buffer)
    total_penalty += calculate_continuous_collision_penalty(pelvis_pos, object2_pos, [half_x, half_y, half_z], clearance_buffer)

    reward = total_penalty

    # Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_other_blocks_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_other_blocks_reward") -> torch.Tensor:
    """
    Penalizes the robot for colliding with or getting too close to the Small Block (Object1) and Large Block (Object3).
    This ensures the robot focuses on the target block (Object2) and avoids unnecessary interactions.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access other blocks
    object1 = env.scene['Object1'] # Accessing object using approved pattern
    object3 = env.scene['Object3'] # Accessing object using approved pattern

    # Object dimensions (hardcoded from object configuration)
    object1_x_dim = 1.0
    object1_y_dim = 1.0
    object1_z_dim = 0.3

    object3_x_dim = 1.0
    object3_y_dim = 1.0
    object3_z_dim = 0.9

    # Robot parts to monitor for collision
    robot = env.scene["robot"] # Accessing robot using approved pattern
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern
    left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Accessing robot part index using approved pattern
    right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Accessing robot part index using approved pattern

    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Accessing robot part position using approved pattern
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Accessing robot part position using approved pattern

    total_penalty = torch.zeros_like(pelvis_pos[:, 0]) # Initialize total penalty tensor for batch operations
    clearance_buffer = 0.1 # Small buffer for collision avoidance, not an arbitrary threshold.

    # Function to calculate continuous penalty for a given robot part and object
    # This function ensures relative distances are used and provides a continuous penalty.
    def calculate_continuous_collision_penalty(robot_part_pos, obj_pos, obj_dims, buffer):
        half_x, half_y, half_z = obj_dims[0]/2.0, obj_dims[1]/2.0, obj_dims[2]/2.0
        
        dist_x = torch.abs(robot_part_pos[:, 0] - obj_pos[:, 0]) # Relative distance in x-direction
        dist_y = torch.abs(robot_part_pos[:, 1] - obj_pos[:, 1]) # Relative distance in y-direction
        dist_z = torch.abs(robot_part_pos[:, 2] - obj_pos[:, 2]) # Relative distance in z-direction
        
        collision_x = (dist_x < half_x + buffer)
        collision_y = (dist_y < half_y + buffer)
        collision_z = (dist_z < half_z + buffer)
        
        collision_condition = collision_x & collision_y & collision_z
        
        # Continuous penalty: more negative the deeper into the object
        penalty_value = -(half_x + half_y + half_z + 3 * buffer - (dist_x + dist_y + dist_z))
        
        return torch.where(collision_condition, penalty_value, torch.zeros_like(penalty_value))

    # Penalties for Object1 (Small Block)
    total_penalty += calculate_continuous_collision_penalty(pelvis_pos, object1.data.root_pos_w, [object1_x_dim, object1_y_dim, object1_z_dim], clearance_buffer)
    total_penalty += calculate_continuous_collision_penalty(left_foot_pos, object1.data.root_pos_w, [object1_x_dim, object1_y_dim, object1_z_dim], clearance_buffer)
    total_penalty += calculate_continuous_collision_penalty(right_foot_pos, object1.data.root_pos_w, [object1_x_dim, object1_y_dim, object1_z_dim], clearance_buffer)

    # Penalties for Object3 (Large Block)
    total_penalty += calculate_continuous_collision_penalty(pelvis_pos, object3.data.root_pos_w, [object3_x_dim, object3_y_dim, object3_z_dim], clearance_buffer)
    total_penalty += calculate_continuous_collision_penalty(left_foot_pos, object3.data.root_pos_w, [object3_x_dim, object3_y_dim, object3_z_dim], clearance_buffer)
    total_penalty += calculate_continuous_collision_penalty(right_foot_pos, object3.data.root_pos_w, [object3_x_dim, object3_y_dim, object3_z_dim], clearance_buffer)

    reward = total_penalty

    # Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Primary reward for positioning the robot near Object2
    walk_to_medium_block_primary_reward = RewTerm(func=walk_to_medium_block_primary_reward, weight=1.0,
                                                  params={"normalise": True, "normaliser_name": "walk_to_medium_block_primary_reward"})

    # Shaping reward for collision avoidance with Object2
    collision_avoidance_object2_reward = RewTerm(func=collision_avoidance_object2_reward, weight=0.6,
                                                 params={"normalise": True, "normaliser_name": "collision_avoidance_object2_reward"})

    # Shaping reward for collision avoidance with other blocks (Object1 and Object3)
    collision_avoidance_other_blocks_reward = RewTerm(func=collision_avoidance_other_blocks_reward, weight=0.4,
                                                      params={"normalise": True, "normaliser_name": "collision_avoidance_other_blocks_reward"})