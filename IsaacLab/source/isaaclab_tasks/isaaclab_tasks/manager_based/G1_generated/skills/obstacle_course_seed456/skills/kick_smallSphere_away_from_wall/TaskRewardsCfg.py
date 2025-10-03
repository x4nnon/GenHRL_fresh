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


def main_kick_smallSphere_away_from_wall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the 'kick_smallSphere_away_from_wall' skill.
    This reward encourages the small sphere (Object2) to move a significant distance away from its
    initial position (relative to the high wall, Object4) and towards the block (Object5),
    without going past the block. The goal is to move it approximately 2 meters.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects using approved patterns
    small_sphere = env.scene['Object2']
    high_wall = env.scene['Object4']
    block_cube = env.scene['Object5']

    # Access object positions using approved patterns
    small_sphere_pos = small_sphere.data.root_pos_w
    high_wall_pos = high_wall.data.root_pos_w
    block_cube_pos = block_cube.data.root_pos_w

    # Hardcoded object dimensions from task description (CRITICAL RULE: NO ACCESSING DIMENSIONS FROM OBJECTS)
    small_sphere_radius = 0.2
    block_cube_x_dim = 0.5

    # Calculate the initial x-position of the small sphere relative to the high wall.
    # The task description states the small sphere is 3m after the high wall.
    # This is a relative calculation based on object positions, not a hardcoded absolute position.
    initial_sphere_x_relative_to_wall = high_wall_pos[:, 0] + 3.0

    # Define the target x-position for the sphere.
    # The sphere should move towards the block but stop just before it.
    # The block's front face is at block_cube_pos[:, 0] - block_cube_x_dim / 2.
    # The sphere's center should be slightly before this point, accounting for its radius.
    # This is a relative calculation based on object positions and hardcoded dimensions.
    target_sphere_x_pos = block_cube_pos[:, 0] - block_cube_x_dim / 2 - small_sphere_radius - 0.1 # 0.1m buffer

    # Calculate the sphere's current x-position relative to its initial position.
    # This measures how much it has moved in the positive x-direction.
    sphere_movement_x = small_sphere_pos[:, 0] - initial_sphere_x_relative_to_wall

    # Reward for moving the sphere in the positive x-direction.
    # We want to reward movement up to a certain point (e.g., 2m past initial, or just before the block).
    # The maximum desired movement is approximately 2m, but also limited by the block's position.
    # We'll use the target_sphere_x_pos as the ultimate goal for the sphere's x-position.
    # A simple negative absolute difference from the target, scaled, provides a continuous reward.
    # We want to maximize sphere_movement_x, but not exceed target_sphere_x_pos.
    # Let's define a desired movement range.
    max_desired_movement_from_initial = 2.0 # Target movement distance from description

    # Calculate the maximum beneficial movement based on the target position and initial position.
    # This ensures the reward doesn't encourage moving past the block.
    max_beneficial_movement = target_sphere_x_pos - initial_sphere_x_relative_to_wall
    # Clamp this to be at least 0.1 to avoid division by zero or negative values if target is behind initial.
    max_beneficial_movement = torch.max(max_beneficial_movement, torch.tensor(0.1, device=env.device))
    # Also cap it at the explicitly stated max desired movement of 2.0m.
    max_beneficial_movement = torch.min(max_beneficial_movement, torch.tensor(max_desired_movement_from_initial, device=env.device))

    # Reward for moving past the initial position, clamped to be non-negative.
    reward_movement_from_start = torch.clamp(sphere_movement_x, min=0.0)
    
    # The reward is the clamped movement, normalized by the maximum beneficial movement.
    # Provide both min and max as tensors to match PyTorch's clamp signature expectations when one is a tensor.
    min_zero = torch.zeros_like(max_beneficial_movement)
    sphere_x_reward = torch.clamp(reward_movement_from_start, min=min_zero, max=max_beneficial_movement) / max_beneficial_movement

    # The final reward is simply this sphere movement reward.
    reward = sphere_x_reward

    # Mandatory normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def robot_stability_and_positioning_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward_1") -> torch.Tensor:
    """
    Shaping reward 1: Encourages the robot's pelvis to maintain a stable height (around 0.7m)
    and to be in a good x-position relative to the block (Object5) for the next skill.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts using approved patterns
    robot = env.scene["robot"]
    block_cube = env.scene['Object5']

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_z = pelvis_pos[:, 2] # Z-position is allowed as an absolute measure for height/stability

    block_cube_pos = block_cube.data.root_pos_w

    # Hardcoded object dimensions from task description
    block_cube_x_dim = 0.5

    # Reward for pelvis stability (z-height)
    # Target pelvis height is 0.7m. Reward is negative absolute difference, making it continuous.
    target_pelvis_z = 0.7
    pelvis_z_reward = -torch.abs(pelvis_pos_z - target_pelvis_z)
    # Normalize this component to be within a reasonable range, e.g., -1 to 0.
    # Max deviation could be 0.7m, so divide by 0.7 to scale.
    pelvis_z_reward = pelvis_z_reward / target_pelvis_z # Scales to approx -1 to 0, continuous.

    # Reward for robot's x-position relative to the block for the next skill.
    # The robot should approach the block, ideally stopping slightly before its front face.
    # Target robot x-position: block's front face minus a buffer (e.g., 0.5m).
    # This is a relative target based on object position and hardcoded dimension.
    target_robot_x_pos = block_cube_pos[:, 0] - block_cube_x_dim / 2 - 0.5
    robot_x_pos_reward = -torch.abs(pelvis_pos_x - target_robot_x_pos) # Continuous reward.
    # Normalize this component. A typical distance could be 1-2m, so divide by 1.0 or 2.0.
    robot_x_pos_reward = robot_x_pos_reward / 1.0 # Scales to approx -2 to 0, continuous.

    # Combine these two components for shaping reward 1
    reward = pelvis_z_reward + robot_x_pos_reward

    # Mandatory normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def foot_sphere_interaction_and_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward_2") -> torch.Tensor:
    """
    Shaping reward 2: Encourages the robot's kicking foot to make contact with the small sphere (Object2)
    and then move away from it after the kick. It also includes collision avoidance for the robot's
    body parts (pelvis, feet) with the high wall (Object4) and the small sphere (Object2).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts using approved patterns
    robot = env.scene["robot"]
    small_sphere = env.scene['Object2']
    high_wall = env.scene['Object4']

    right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Assuming right foot for kicking
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    right_foot_pos_x = right_foot_pos[:, 0]
    right_foot_pos_y = right_foot_pos[:, 1]
    right_foot_pos_z = right_foot_pos[:, 2]

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    small_sphere_pos = small_sphere.data.root_pos_w
    high_wall_pos = high_wall.data.root_pos_w

    # Hardcoded object dimensions from task description
    small_sphere_radius = 0.2
    high_wall_x_dim = 0.3
    high_wall_y_dim = 5.0
    high_wall_z_dim = 1.0

    # Reward for foot proximity to sphere (for kicking)
    # This reward encourages the foot to get close to the sphere for a kick.
    # Distance is calculated using torch.norm for continuous reward.
    distance_foot_sphere = torch.norm(right_foot_pos - small_sphere_pos, dim=1)

    # Reward for getting foot close to sphere. Use an exponential decay.
    # Reward is higher when distance is smaller, max 1.0 when distance is 0. Continuous.
    foot_sphere_proximity_reward = torch.exp(-5.0 * distance_foot_sphere)

    # Condition this reward to be active only when the robot is generally in the kicking phase.
    # Robot's pelvis should be roughly aligned with the sphere in y, and before/at it in x.
    # These are relative distance checks.
    condition_approach_sphere = (torch.abs(pelvis_pos_y - small_sphere_pos[:, 1]) < 0.5) & \
                                (pelvis_pos_x < small_sphere_pos[:, 0] + 0.5) & \
                                (pelvis_pos_x > small_sphere_pos[:, 0] - 1.5) # Robot is in front of sphere

    # Apply the condition: reward is 0 if condition is not met.
    foot_sphere_proximity_reward = torch.where(condition_approach_sphere, foot_sphere_proximity_reward, torch.tensor(0.0, device=env.device))

    # Collision avoidance with high wall (Object4)
    # Penalize if pelvis or right foot gets too close to the high wall.
    # High wall is a box. Its center is at high_wall_pos.
    # Define half-dimensions for collision checks using hardcoded dimensions.
    half_wall_x = high_wall_x_dim / 2.0
    half_wall_y = high_wall_y_dim / 2.0
    half_wall_z = high_wall_z_dim / 2.0

    # Pelvis collision avoidance
    # Check if pelvis is within the wall's y and z extent (plus a small buffer for robot size)
    # These are relative distance checks.
    pelvis_in_wall_y_range = torch.abs(pelvis_pos_y - high_wall_pos[:, 1]) < (half_wall_y + 0.3) # 0.3m buffer for robot width
    pelvis_in_wall_z_range = torch.abs(pelvis_pos_z - high_wall_pos[:, 2]) < (half_wall_z + 0.3) # 0.3m buffer for robot height

    # If within y and z range, penalize based on x-distance to wall's x-faces.
    pelvis_dist_to_wall_x_face = torch.abs(pelvis_pos_x - high_wall_pos[:, 0])
    
    # Collision penalty if pelvis is too close in x, and within y/z ranges.
    # Use a negative exponential for continuous penalty, becoming more negative as it gets closer.
    collision_threshold_x_pelvis = half_wall_x + 0.2 # 0.2m buffer for robot depth
    
    pelvis_collision_penalty = torch.where(
        pelvis_in_wall_y_range & pelvis_in_wall_z_range & (pelvis_dist_to_wall_x_face < collision_threshold_x_pelvis),
        -torch.exp(-5.0 * (pelvis_dist_to_wall_x_face - collision_threshold_x_pelvis)), # Penalize more as it gets closer
        torch.tensor(0.0, device=env.device)
    )
    # Ensure penalty is always negative or zero.
    pelvis_collision_penalty = torch.clamp(pelvis_collision_penalty, max=0.0)


    # Right foot collision avoidance (similar logic)
    foot_in_wall_y_range = torch.abs(right_foot_pos_y - high_wall_pos[:, 1]) < (half_wall_y + 0.2) # Smaller buffer for foot
    foot_in_wall_z_range = torch.abs(right_foot_pos_z - high_wall_pos[:, 2]) < (half_wall_z + 0.2)

    foot_dist_to_wall_x_face = torch.abs(right_foot_pos_x - high_wall_pos[:, 0])
    
    collision_threshold_x_foot = half_wall_x + 0.1 # Smaller buffer for foot
    
    foot_collision_penalty = torch.where(
        foot_in_wall_y_range & foot_in_wall_z_range & (foot_dist_to_wall_x_face < collision_threshold_x_foot),
        -torch.exp(-5.0 * (foot_dist_to_wall_x_face - collision_threshold_x_foot)),
        torch.tensor(0.0, device=env.device)
    )
    foot_collision_penalty = torch.clamp(foot_collision_penalty, max=0.0)

    # Combine collision penalties
    collision_avoidance_reward = pelvis_collision_penalty + foot_collision_penalty

    # Combine all components for shaping reward 2
    reward = foot_sphere_proximity_reward + collision_avoidance_reward

    # Mandatory normalization implementation
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
    Reward terms for the 'kick_smallSphere_away_from_wall' skill.
    """
    # Main reward for moving the small sphere away from the wall
    main_kick_smallSphere_away_from_wall_reward = RewTerm(
        func=main_kick_smallSphere_away_from_wall_reward,
        weight=1.0, # Main reward, typically weight 1.0
        params={"normalise": True, "normaliser_name": "main_kick_smallSphere_away_from_wall_reward"}
    )

    # Shaping reward for robot stability and positioning for the next skill
    robot_stability_and_positioning_reward = RewTerm(
        func=robot_stability_and_positioning_reward,
        weight=0.6, # Moderate weight to encourage stability and preparation
        params={"normalise": True, "normaliser_name": "robot_stability_and_positioning_reward"}
    )

    # Shaping reward for foot-sphere interaction and collision avoidance
    foot_sphere_interaction_and_collision_avoidance_reward = RewTerm(
        func=foot_sphere_interaction_and_collision_avoidance_reward,
        weight=0.4, # Lower weight as it's a more detailed shaping reward
        params={"normalise": True, "normaliser_name": "foot_sphere_interaction_and_collision_avoidance_reward"}
    )