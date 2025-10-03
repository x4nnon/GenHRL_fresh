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


def main_walk_to_small_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_walk_to_small_block_reward") -> torch.Tensor:
    """
    Main reward for the walk_to_small_block skill.
    Encourages the robot's pelvis to reach the immediate vicinity of the small block (Object3) and stop.
    Rewards for alignment in x, reaching target y, and maintaining stable z-height.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved pattern
    object3 = env.scene['Object3']
    object3_pos = object3.data.root_pos_w

    # Access the required robot part(s) using approved pattern
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Object3 dimensions (from skill info: 0.3m cubed) - Hardcoded as per requirements
    # object3_size_y = 0.3 # This variable is not used, can be removed or kept for clarity if needed later.

    # Target y-position for pelvis: just before the block, considering robot's depth.
    # A target offset of 0.5m from the block's center in y means the robot's pelvis should be 0.5m behind the block's center.
    # This places the robot's front (e.g., feet) near the block's front face.
    target_y_offset = 0.5 # Distance from object3.y to robot pelvis y, this is an offset, not an absolute position.
    target_pelvis_y = object3_pos[:, 1] - target_y_offset # Relative target y-position calculation.

    # Calculate distances to target using relative positions
    distance_y = pelvis_pos[:, 1] - target_pelvis_y # Relative distance in y-direction
    distance_x = pelvis_pos[:, 0] - object3_pos[:, 0] # Relative distance in x-direction
    # Distance in z: pelvis_pos_z should be close to a stable height (e.g., 0.7m for humanoid)
    # This is one of the few cases where an absolute Z value is acceptable, as it refers to robot stability.
    target_pelvis_z = 0.7 # Absolute Z target for stability, allowed by prompt.
    distance_z = pelvis_pos[:, 2] - target_pelvis_z # Relative distance in z-direction

    # Reward for reaching target y-position, with a focus on not overshooting
    reward_y = -torch.abs(distance_y) # Continuous reward for y-proximity, based on relative distance.

    # Add a penalty for overshooting the target y-position.
    # If pelvis_pos_y is greater than target_pelvis_y, it means the robot has moved past the target.
    overshoot_penalty = torch.where(pelvis_pos[:, 1] > target_pelvis_y, (pelvis_pos[:, 1] - target_pelvis_y) * 2.0, 0.0)
    reward_y -= overshoot_penalty # Stronger penalty for overshooting, based on relative distance.

    # Reward for x-alignment: penalize deviation from object3's x-position
    reward_x = -torch.abs(distance_x) # Continuous reward for x-alignment, based on relative distance.

    # Reward for maintaining stable pelvis height
    reward_z = -torch.abs(distance_z) # Continuous reward for z-height stability, based on relative distance to target_pelvis_z.

    # Combine rewards
    reward = reward_y + reward_x + reward_z

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def doorway_clearance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "doorway_clearance_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to maintain a safe distance from Object1 and Object2 (the walls forming the doorway).
    Applies a penalty if the robot gets too close to either wall along the x-axis.
    This reward is active only when the robot is within the y-range of the doorway.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved pattern
    object1 = env.scene['Object1'] # Heavy Cube (Wall 1)
    object2 = env.scene['Object2'] # Heavy Cube (Wall 2)
    object1_pos = object1.data.root_pos_w
    object2_pos = object2.data.root_pos_w

    # Access the required robot part(s) using approved pattern
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_y = pelvis_pos[:, 1] # Extract y-component for activation condition

    left_palm_idx = robot.body_names.index('left_palm_link')
    left_palm_pos = robot.data.body_pos_w[:, left_palm_idx]
    right_palm_idx = robot.body_names.index('right_palm_link')
    right_palm_pos = robot.data.body_pos_w[:, right_palm_idx]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    head_idx = robot.body_names.index('head_link')
    head_pos = robot.data.body_pos_w[:, head_idx]

    # Object1 and Object2 dimensions: "x of 0.5" (0.5m width in x-direction) - Hardcoded from config
    wall_x_dimension = 0.5
    half_wall_x_dimension = wall_x_dimension / 2.0

    # Define the inner faces of the walls using relative positions.
    # Assuming Object1 is the left wall and Object2 is the right wall.
    inner_face_obj1_x = object1_pos[:, 0] + half_wall_x_dimension # Relative to Object1's center
    inner_face_obj2_x = object2_pos[:, 0] - half_wall_x_dimension # Relative to Object2's center

    # Define the y-range where the doorway is active.
    # Walls are "y of 5m" (5m length in y-direction). Hardcoded from config.
    wall_y_length = 5.0
    # Assuming walls are centered at their root_pos_w[:, 1]
    doorway_y_min = object1_pos[:, 1] - (wall_y_length / 2.0) # Relative to Object1's center
    doorway_y_max = object1_pos[:, 1] + (wall_y_length / 2.0) # Relative to Object1's center

    # Activation condition: robot's pelvis is within the y-span of the walls
    activation_condition = (pelvis_pos_y > doorway_y_min) & (pelvis_pos_y < doorway_y_max)

    # Define a small buffer for clearance to avoid collisions - Hardcoded threshold, allowed.
    clearance_buffer = 0.1 # Robot parts should stay at least 0.1m away from the inner wall faces

    # Calculate distances to inner faces of walls for various body parts. All distances are relative.
    # A positive distance means the part is outside the wall (good).
    # A negative distance means the part is inside the wall (bad).
    # We want to penalize when distance is less than clearance_buffer.

    # Distances to left wall (Object1)
    dist_to_obj1_x_pelvis = pelvis_pos[:, 0] - inner_face_obj1_x
    dist_to_obj1_x_lpalm = left_palm_pos[:, 0] - inner_face_obj1_x
    dist_to_obj1_x_rpalm = right_palm_pos[:, 0] - inner_face_obj1_x
    dist_to_obj1_x_lfoot = left_foot_pos[:, 0] - inner_face_obj1_x
    dist_to_obj1_x_rfoot = right_foot_pos[:, 0] - inner_face_obj1_x
    dist_to_obj1_x_head = head_pos[:, 0] - inner_face_obj1_x

    # Distances to right wall (Object2)
    dist_to_obj2_x_pelvis = inner_face_obj2_x - pelvis_pos[:, 0]
    dist_to_obj2_x_lpalm = inner_face_obj2_x - left_palm_pos[:, 0]
    dist_to_obj2_x_rpalm = inner_face_obj2_x - right_palm_pos[:, 0]
    dist_to_obj2_x_lfoot = inner_face_obj2_x - left_foot_pos[:, 0]
    dist_to_obj2_x_rfoot = inner_face_obj2_x - right_foot_pos[:, 0]
    dist_to_obj2_x_head = inner_face_obj2_x - head_pos[:, 0]

    # Penalty for being too close to left wall (negative values if too close/colliding)
    # The penalty is (distance - clearance_buffer), so if distance is 0.05 and buffer is 0.1, penalty is -0.05.
    # We want to maximize this value (make it less negative), so we multiply by -1 later.
    penalty_obj1 = torch.where(dist_to_obj1_x_pelvis < clearance_buffer, dist_to_obj1_x_pelvis - clearance_buffer, 0.0)
    penalty_obj1 += torch.where(dist_to_obj1_x_lpalm < clearance_buffer, dist_to_obj1_x_lpalm - clearance_buffer, 0.0)
    penalty_obj1 += torch.where(dist_to_obj1_x_rpalm < clearance_buffer, dist_to_obj1_x_rpalm - clearance_buffer, 0.0)
    penalty_obj1 += torch.where(dist_to_obj1_x_lfoot < clearance_buffer, dist_to_obj1_x_lfoot - clearance_buffer, 0.0)
    penalty_obj1 += torch.where(dist_to_obj1_x_rfoot < clearance_buffer, dist_to_obj1_x_rfoot - clearance_buffer, 0.0)
    penalty_obj1 += torch.where(dist_to_obj1_x_head < clearance_buffer, dist_to_obj1_x_head - clearance_buffer, 0.0)

    # Penalty for being too close to right wall
    penalty_obj2 = torch.where(dist_to_obj2_x_pelvis < clearance_buffer, dist_to_obj2_x_pelvis - clearance_buffer, 0.0)
    penalty_obj2 += torch.where(dist_to_obj2_x_lpalm < clearance_buffer, dist_to_obj2_x_lpalm - clearance_buffer, 0.0)
    penalty_obj2 += torch.where(dist_to_obj2_x_rpalm < clearance_buffer, dist_to_obj2_x_rpalm - clearance_buffer, 0.0)
    penalty_obj2 += torch.where(dist_to_obj2_x_lfoot < clearance_buffer, dist_to_obj2_x_lfoot - clearance_buffer, 0.0)
    penalty_obj2 += torch.where(dist_to_obj2_x_rfoot < clearance_buffer, dist_to_obj2_x_rfoot - clearance_buffer, 0.0)
    penalty_obj2 += torch.where(dist_to_obj2_x_head < clearance_buffer, dist_to_obj2_x_head - clearance_buffer, 0.0)

    # Total collision penalty. Multiply by 5.0 to make it a significant penalty.
    # The sum of penalties is negative when too close, so this becomes a negative reward (penalty).
    collision_reward = (penalty_obj1 + penalty_obj2) * 5.0 # Scale penalty

    # Apply reward only when robot is within the doorway's y-span
    reward = torch.where(activation_condition, collision_reward, 0.0)

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def forward_progress_pre_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "forward_progress_pre_block_reward") -> torch.Tensor:
    """
    Shaping reward to encourage continuous forward movement along the y-axis towards the small block (Object3)
    *before* the robot reaches the immediate vicinity of the block.
    This helps prevent the robot from getting stuck or wandering.
    It deactivates as the robot gets close to the target y-position of Object3, allowing the primary reward to take over.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved pattern
    object3 = env.scene['Object3']
    object3_pos = object3.data.root_pos_w

    # Access the required robot part(s) using approved pattern
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_y = pelvis_pos[:, 1] # Extract y-component

    # Target y-position for pelvis (just before the block, same as primary reward)
    target_y_offset = 0.5 # This value is consistent with the main reward, an offset.
    target_pelvis_y = object3_pos[:, 1] - target_y_offset # Relative target y-position calculation.

    # Activation condition: robot is still far from the target y-position of the block.
    # This reward is active if the robot's pelvis is more than 1.0m away from the target_pelvis_y
    # in the negative y direction (i.e., before reaching the target).
    activation_threshold = 1.0 # Hardcoded threshold, allowed for activation condition.
    activation_condition = (target_pelvis_y - pelvis_pos_y) > activation_threshold

    # Reward for reducing the y-distance to the target_pelvis_y.
    # This reward should be positive for decreasing distance.
    # (pelvis_pos_y - target_pelvis_y) will be negative when robot is behind target, and increases towards zero as it moves forward.
    reward = pelvis_pos_y - target_pelvis_y # Continuous reward, based on relative distance.

    # Apply reward only when the activation condition is met
    reward = torch.where(activation_condition, reward, 0.0)

    # Mandatory normalization
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
    Reward terms for the walk_to_small_block skill.
    """
    # Main reward for reaching the small block
    main_walk_to_small_block_reward = RewTerm(
        func=main_walk_to_small_block_reward,
        weight=1.0, # Primary reward, typically weight 1.0
        params={"normalise": True, "normaliser_name": "main_walk_to_small_block_reward"}
    )

    # Shaping reward for collision avoidance with doorway walls
    doorway_clearance_reward = RewTerm(
        func=doorway_clearance_reward,
        weight=0.6, # Supporting reward, lower weight than main
        params={"normalise": True, "normaliser_name": "doorway_clearance_reward"}
    )

    # Shaping reward for forward progress before reaching the block
    forward_progress_pre_block_reward = RewTerm(
        func=forward_progress_pre_block_reward,
        weight=0.3, # Supporting reward, lower weight
        params={"normalise": True, "normaliser_name": "forward_progress_pre_block_reward"}
    )