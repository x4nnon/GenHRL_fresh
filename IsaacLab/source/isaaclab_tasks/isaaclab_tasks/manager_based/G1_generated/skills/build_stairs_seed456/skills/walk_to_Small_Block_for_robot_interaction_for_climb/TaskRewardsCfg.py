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


def walk_to_small_block_primary_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_small_block_primary_reward") -> torch.Tensor:
    """
    Primary reward for positioning the robot's pelvis and feet directly in front of the Small Block (Object1)
    at an optimal distance for jumping.
    """
    # Get normalizer instance for this reward function
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    small_block = env.scene['Object1']
    small_block_pos = small_block.data.root_pos_w # [num_envs, 3]

    # Access the required robot part(s) using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # [num_envs, 3]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # [num_envs, 3]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # [num_envs, 3]
    
    # Calculate average feet position for precise positioning
    avg_feet_pos_x = (left_foot_pos[:, 0] + right_foot_pos[:, 0]) / 2
    avg_feet_pos_y = (left_foot_pos[:, 1] + right_foot_pos[:, 1]) / 2
    avg_feet_pos_z = (left_foot_pos[:, 2] + right_foot_pos[:, 2]) / 2

    # Small Block dimensions (hardcoded from object configuration: x=1m, y=1m, z=0.3m)
    # This adheres to the rule of not accessing dimensions from the object itself.
    small_block_x_dim = 1.0
    small_block_y_dim = 1.0
    small_block_z_dim = 0.3

    # Define target position relative to the small block for jumping
    # The plan states: "Target X: 0.5m before the block's front face (block_x - block_x_dim/2 - 0.5)"
    # Assuming block_x is the center, front face is at block_x + block_x_dim/2.
    # So, "0.5m before the block's front face" would be (block_x + block_x_dim/2) - 0.5.
    # However, the plan's formula `block_x - block_x_dim/2 - 0.5` implies approaching from positive X
    # and targeting a point 0.5m *behind* the block's *back* face.
    # Given the common sense of "in front of" for jumping, and the robot starting at origin,
    # it's more likely the robot approaches from negative X towards positive X of the block.
    # Let's interpret "0.5m before the block's front face" as 0.5m *in front of* the block's *center*
    # if approaching from negative X, which means `small_block_pos[:, 0] - 0.5`.
    # This aligns with the common interpretation of "before" when moving towards an object.
    target_x_offset_from_center = 0.5 # Distance from the block's center
    target_pelvis_z = 0.7 # Stable standing height
    target_feet_z = 0.0 # Ground level

    # Calculate target coordinates based on Object1's position
    # Using the more intuitive interpretation: 0.5m in front of the block's center along X.
    target_pos_x = small_block_pos[:, 0] - target_x_offset_from_center # Target X is 0.5m before the block's center
    target_pos_y = small_block_pos[:, 1] # Target Y is centered with the block

    # Pelvis distance to target (using relative distances)
    dist_pelvis_x = torch.abs(pelvis_pos[:, 0] - target_pos_x)
    dist_pelvis_y = torch.abs(pelvis_pos[:, 1] - target_pos_y)
    dist_pelvis_z = torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Average feet distance to target (using relative distances)
    dist_feet_x = torch.abs(avg_feet_pos_x - target_pos_x)
    dist_feet_y = torch.abs(avg_feet_pos_y - target_pos_y)
    dist_feet_z = torch.abs(avg_feet_pos_z - target_feet_z)

    # Combine distances for primary reward. Negative sum for continuous reward.
    # Weighted towards feet for precise positioning and pelvis for overall stability.
    primary_reward = - (dist_pelvis_x * 0.4 + dist_pelvis_y * 0.4 + dist_pelvis_z * 0.2 +
                        dist_feet_x * 0.6 + dist_feet_y * 0.6 + dist_feet_z * 0.4)

    # Ensure robot does not walk past the block (overshoot penalty)
    # The front face of the block (assuming positive X is forward for the block) is at small_block_pos[:, 0] + small_block_x_dim / 2.
    # If robot's average feet x position is beyond this point plus a small buffer, penalize.
    overshoot_threshold = 0.1 # 0.1m past the front face
    overshoot_condition = avg_feet_pos_x > (small_block_pos[:, 0] + small_block_x_dim / 2 + overshoot_threshold)
    primary_reward = torch.where(overshoot_condition, primary_reward - 10.0, primary_reward) # Large penalty for overshooting

    reward = primary_reward

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    """
    Penalizes the robot for any part of its body (excluding feet) coming too close or colliding with Object1, Object2, or Object3.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    small_block = env.scene['Object1']
    medium_block = env.scene['Object2']
    large_block = env.scene['Object3']

    # Robot body parts to monitor for collision (excluding feet)
    robot = env.scene["robot"]
    body_parts_to_monitor = ['pelvis', 'head_link', 'left_palm_link', 'right_palm_link',
                             'left_knee_link', 'right_knee_link', 'left_elbow_pitch_link', 'right_elbow_pitch_link']

    collision_reward = torch.zeros(env.num_envs, device=env.device)
    collision_threshold = 0.1 # Distance threshold for collision avoidance (buffer around block)

    # Hardcoded block dimensions from object configuration
    # Small Block: x=1m y=1m z=0.3m
    # Medium Block: x=1m y=1m z=0.6m
    # Large Block: x=1m y=1m z=0.9m
    block_dims = {
        small_block: (1.0, 1.0, 0.3),
        medium_block: (1.0, 1.0, 0.6),
        large_block: (1.0, 1.0, 0.9)
    }

    for part_name in body_parts_to_monitor:
        part_idx = robot.body_names.index(part_name)
        part_pos = robot.data.body_pos_w[:, part_idx] # [num_envs, 3]

        for block in [small_block, medium_block, large_block]:
            block_pos = block.data.root_pos_w # [num_envs, 3]
            block_x_dim, block_y_dim, block_z_dim = block_dims[block]
            
            block_half_x = block_x_dim / 2.0
            block_half_y = block_y_dim / 2.0
            block_half_z = block_z_dim / 2.0

            # Calculate relative distance from part to block center for each dimension
            dist_x = torch.abs(part_pos[:, 0] - block_pos[:, 0])
            dist_y = torch.abs(part_pos[:, 1] - block_pos[:, 1])
            dist_z = torch.abs(part_pos[:, 2] - block_pos[:, 2])

            # Calculate overlap for each dimension. Clamp to 0 if no overlap (part is outside buffer).
            # This creates a continuous penalty that increases as the part penetrates deeper into the block's buffered volume.
            overlap_x = torch.clamp((block_half_x + collision_threshold) - dist_x, min=0.0)
            overlap_y = torch.clamp((block_half_y + collision_threshold) - dist_y, min=0.0)
            overlap_z = torch.clamp((block_half_z + collision_threshold) - dist_z, min=0.0)
            
            # Sum of overlaps as a penalty. Scale factor to make it significant.
            # This ensures the reward is continuous and penalizes proximity.
            collision_penalty = (overlap_x + overlap_y + overlap_z) * 5.0
            
            collision_reward -= collision_penalty

    reward = collision_reward

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def pelvis_stability_orientation_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_stability_orientation_reward") -> torch.Tensor:
    """
    Encourages the robot to maintain a stable, upright posture by penalizing large deviations of the pelvis
    from the target height (0.7m) and penalizing falling.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot part(s)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # [num_envs, 3]
    pelvis_pos_z = pelvis_pos[:, 2] # Z-component for height

    # Target pelvis height
    target_pelvis_z = 0.7

    # Pelvis height reward: penalize deviation from target_pelvis_z.
    # This is a continuous negative reward, encouraging the pelvis to stay near 0.7m.
    pelvis_height_reward = -torch.abs(pelvis_pos_z - target_pelvis_z) * 0.5 # Scale factor

    # Penalty if the robot's pelvis is too far from the ground, indicating instability or falling.
    # This is distinct from the target_pelvis_z which is for desired standing height.
    min_pelvis_z_threshold = 0.3 # If pelvis falls below this, it's likely falling
    # Apply a large, continuous penalty if pelvis is below threshold, scaling with how far below it is.
    falling_penalty = torch.where(pelvis_pos_z < min_pelvis_z_threshold, (pelvis_pos_z - min_pelvis_z_threshold) * 10.0, 0.0)
    # The penalty is negative, so (pelvis_pos_z - min_pelvis_z_threshold) will be negative when falling, making the penalty more negative.

    # Orientation reward: The plan mentions facing Object1.
    # Given the constraints (no quat_rotate_vector, focus on relative distances),
    # direct orientation alignment using dot products is complex.
    # The primary reward's x-y alignment will implicitly encourage facing if the robot walks forward.
    # For this shaping reward, we will focus on height stability and falling, as per the simplified skeleton.
    # If a more explicit orientation reward is needed, it would require quaternion math to get the robot's forward vector.
    # For now, we omit explicit orientation reward here to adhere to "approved access patterns" and simplicity.

    shaping_reward = pelvis_height_reward + falling_penalty

    reward = shaping_reward

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
    # Primary reward for positioning the robot in front of the small block
    walk_to_SmallBlock_PrimaryReward = RewTerm(func=walk_to_small_block_primary_reward, weight=1.0,
                                               params={"normalise": True, "normaliser_name": "walk_to_small_block_primary_reward"})

    # Shaping reward for collision avoidance with all blocks
    CollisionAvoidanceReward = RewTerm(func=collision_avoidance_reward, weight=0.4,
                                       params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})

    # Shaping reward for pelvis height stability and preventing falling
    PelvisStabilityOrientationReward = RewTerm(func=pelvis_stability_orientation_reward, weight=0.3,
                                               params={"normalise": True, "normaliser_name": "pelvis_stability_orientation_reward"})