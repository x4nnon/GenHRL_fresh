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


def kick_small_sphere_away_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "kick_small_sphere_away_reward") -> torch.Tensor:
    """
    Primary reward for the 'Kick_SmallSphere_Away_From_Wall' skill.
    This reward guides the robot through approaching the small sphere (Object2), positioning for the kick,
    successfully kicking the sphere away from its initial position and the high wall (Object4),
    and finally, penalizes overshooting the next target (Object5).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot and objects using approved patterns
    robot = env.scene["robot"]
    object_small_sphere = env.scene['Object2'] # Object2 is the small sphere for robot to kick
    object_high_wall = env.scene['Object4']   # Object4 is the high wall for large sphere to push over
    object_block = env.scene['Object5']       # Object5 is the block cube for robot to jump on top of

    # Access robot part positions using approved patterns
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    right_foot_pos_x = right_foot_pos[:, 0]
    right_foot_pos_y = right_foot_pos[:, 1]

    # Hardcoded object dimensions from the object configuration (as per requirements)
    small_sphere_radius = 0.2
    high_wall_x_dim = 0.3
    block_x_dim = 0.5

    # Phase 1: Approach small sphere (Object2)
    # Reward for pelvis moving towards the small sphere in x, and staying aligned in y.
    # This uses relative distances between pelvis and small sphere.
    dist_pelvis_to_sphere_x = object_small_sphere.data.root_pos_w[:, 0] - pelvis_pos_x
    dist_pelvis_to_sphere_y = object_small_sphere.data.root_pos_w[:, 1] - pelvis_pos_y
    approach_reward = -torch.abs(dist_pelvis_to_sphere_x) - torch.abs(dist_pelvis_to_sphere_y)

    # Phase 2: Positioning kicking foot (right foot) near the sphere
    # Condition: Pelvis is close to the sphere (e.g., within 1.0m in x) to activate this phase.
    # This ensures the robot is in the general vicinity before fine-tuning foot position.
    pelvis_near_sphere_condition = (pelvis_pos_x < object_small_sphere.data.root_pos_w[:, 0] + 1.0) & \
                                   (pelvis_pos_x > object_small_sphere.data.root_pos_w[:, 0] - 1.0)

    # Reward for right foot being close to the sphere, slightly behind in x, and aligned in y.
    # Target position for kicking foot: sphere_x - sphere_radius - small_offset, sphere_y.
    # All positions are relative to the sphere's current position.
    target_foot_x = object_small_sphere.data.root_pos_w[:, 0] - small_sphere_radius - 0.1 # small offset behind sphere
    target_foot_y = object_small_sphere.data.root_pos_w[:, 1]

    dist_foot_to_kick_pos_x = target_foot_x - right_foot_pos_x
    dist_foot_to_kick_pos_y = target_foot_y - right_foot_pos_y
    positioning_reward = -torch.abs(dist_foot_to_kick_pos_x) - torch.abs(dist_foot_to_kick_pos_y)
    # Only apply positioning reward if pelvis is near the sphere.
    positioning_reward = torch.where(pelvis_near_sphere_condition, positioning_reward, 0.0)

    # Phase 3: Kicking the sphere away
    # Condition: Right foot is very close to the sphere to indicate a potential kick.
    # This uses relative distances between right foot and small sphere.
    foot_at_sphere_condition = (torch.abs(right_foot_pos_x - object_small_sphere.data.root_pos_w[:, 0]) < small_sphere_radius + 0.1) & \
                               (torch.abs(right_foot_pos_y - object_small_sphere.data.root_pos_w[:, 1]) < small_sphere_radius + 0.1)

    # Reward for sphere moving away from its initial position (relative to high wall).
    # The initial x-position of the small sphere is assumed to be 3m after the high wall's x-end.
    # This provides a continuous reward for the sphere's positive x-displacement.
    initial_sphere_x_ref = object_high_wall.data.root_pos_w[:, 0] + high_wall_x_dim / 2.0 + 3.0
    sphere_moved_x_reward = object_small_sphere.data.root_pos_w[:, 0] - initial_sphere_x_ref
    # Apply higher weight to this reward when the foot is in kicking proximity.
    sphere_moved_x_reward = torch.where(foot_at_sphere_condition, sphere_moved_x_reward * 2.0, 0.0)

    # Phase 4: Ensure robot doesn't overshoot the next skill's target (Object5)
    # The robot's pelvis should not go past the block's starting x-position.
    # This uses relative distance between pelvis and block.
    block_start_x = object_block.data.root_pos_w[:, 0] - block_x_dim / 2.0
    # Penalize if pelvis x-position is greater than the block's start x-position.
    overshoot_penalty = torch.where(pelvis_pos_x > block_start_x, -torch.abs(pelvis_pos_x - block_start_x) * 5.0, 0.0)

    # Combine all reward components
    reward = approach_reward + positioning_reward + sphere_moved_x_reward + overshoot_penalty

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def posture_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "posture_stability_reward") -> torch.Tensor:
    """
    Shaping reward 1: Encourages the robot to maintain an upright and stable posture.
    Rewards pelvis staying at a target height (0.7m) and penalizes large deviations in y-axis from the center.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot part positions using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Reward for pelvis height: continuous penalty for deviation from target_pelvis_z.
    # This is one of the few cases where an absolute position (z-height) is allowed.
    target_pelvis_z = 0.7
    pelvis_height_reward = -torch.abs(pelvis_pos_z - target_pelvis_z)

    # Penalty for large y-deviation: continuous penalty for deviation from y=0 (central path).
    # This uses relative distance from y=0.
    pelvis_y_deviation_penalty = -torch.abs(pelvis_pos_y) * 0.5

    reward = pelvis_height_reward + pelvis_y_deviation_penalty

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
    Shaping reward 2: Penalizes unnecessary collisions of robot body parts with the small sphere (Object2)
    or the high wall (Object4). This ensures a clean approach and kick.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access objects using approved patterns
    object_small_sphere = env.scene['Object2']
    object_high_wall = env.scene['Object4']

    # Hardcoded object dimensions from the object configuration (as per requirements)
    small_sphere_radius = 0.2
    high_wall_x_dim = 0.3
    high_wall_y_dim = 5.0
    high_wall_z_dim = 1.0

    # Define robot parts to check for collisions
    robot = env.scene["robot"]
    robot_parts_to_check = ['pelvis', 'left_ankle_roll_link', 'right_ankle_roll_link', 'left_palm_link', 'right_palm_link']
    collision_penalty = torch.zeros(env.num_envs, device=env.device) # Initialize penalty tensor

    for part_name in robot_parts_to_check:
        part_idx = robot.body_names.index(part_name)
        part_pos = robot.data.body_pos_w[:, part_idx]
        part_pos_x = part_pos[:, 0]
        part_pos_y = part_pos[:, 1]
        part_pos_z = part_pos[:, 2]

        # Collision with small sphere (Object2)
        # Penalize if any part other than the right foot (kicking foot) is too close.
        # For the right foot, we assume its contact is handled by the primary reward (kicking phase).
        is_right_foot = (part_name == 'right_ankle_roll_link')
        
        # Calculate relative distances to the sphere's center
        dist_to_sphere_x = torch.abs(part_pos_x - object_small_sphere.data.root_pos_w[:, 0])
        dist_to_sphere_y = torch.abs(part_pos_y - object_small_sphere.data.root_pos_w[:, 1])
        dist_to_sphere_z = torch.abs(part_pos_z - object_small_sphere.data.root_pos_w[:, 2])
        
        # Define a collision threshold slightly larger than the sphere's radius
        sphere_collision_threshold = small_sphere_radius + 0.05
        
        # Check if any dimension is within the threshold, indicating proximity
        # Using max of distances to ensure it's "inside" a conceptual collision box
        # A more precise check would be Euclidean distance, but this is simpler and often sufficient for penalty.
        # The prompt's skeleton used individual axis checks, so we'll stick to that for consistency.
        sphere_collision_condition = (dist_to_sphere_x < sphere_collision_threshold) & \
                                     (dist_to_sphere_y < sphere_collision_threshold) & \
                                     (dist_to_sphere_z < sphere_collision_threshold)
        
        # Apply penalty if collision condition met AND it's not the right foot.
        # Using a continuous penalty based on proximity, rather than binary, for smoother learning.
        # The original skeleton used a continuous penalty for sphere collision, but then a binary for wall.
        # Let's make it consistent with the prompt's skeleton for sphere:
        # Ensure the mask is a boolean tensor by combining with a tensor mask
        not_right_foot_mask = torch.full((env.num_envs,), (part_name != 'right_ankle_roll_link'), dtype=torch.bool, device=env.device)
        collision_penalty_sphere_part = torch.where(sphere_collision_condition & not_right_foot_mask,
                                                    -1.0 * (sphere_collision_threshold - torch.max(torch.max(dist_to_sphere_x, dist_to_sphere_y), dist_to_sphere_z)),
                                                    0.0)
        collision_penalty += collision_penalty_sphere_part

        # Collision with high wall (Object4)
        # Penalize any contact with the high wall.
        # Calculate wall boundaries relative to its root position and dimensions.
        wall_min_x = object_high_wall.data.root_pos_w[:, 0] - high_wall_x_dim / 2.0
        wall_max_x = object_high_wall.data.root_pos_w[:, 0] + high_wall_x_dim / 2.0
        wall_min_y = object_high_wall.data.root_pos_w[:, 1] - high_wall_y_dim / 2.0
        wall_max_y = object_high_wall.data.root_pos_w[:, 1] + high_wall_y_dim / 2.0
        wall_min_z = object_high_wall.data.root_pos_w[:, 2] - high_wall_z_dim / 2.0
        wall_max_z = object_high_wall.data.root_pos_w[:, 2] + high_wall_z_dim / 2.0

        # Check if part position is within wall boundaries
        wall_collision_condition = (part_pos_x > wall_min_x) & (part_pos_x < wall_max_x) & \
                                   (part_pos_y > wall_min_y) & (part_pos_y < wall_max_y) & \
                                   (part_pos_z > wall_min_z) & (part_pos_z < wall_max_z)
        
        # Apply a continuous penalty for being inside the wall.
        # The prompt's skeleton specified a binary penalty for the wall, so adhering to that.
        collision_penalty_wall_part = torch.where(wall_collision_condition, -1.0, 0.0) # Binary penalty for simplicity as per plan
        collision_penalty += collision_penalty_wall_part

    reward = collision_penalty

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
    # Primary reward for the main task objective: approaching, positioning, kicking, and avoiding overshoot.
    KickSmallSphereAwayReward = RewTerm(func=kick_small_sphere_away_reward, weight=1.0,
                                        params={"normalise": True, "normaliser_name": "kick_small_sphere_away_reward"})

    # Shaping reward 1: Encourages stable and upright posture.
    PostureStabilityReward = RewTerm(func=posture_stability_reward, weight=0.4,
                                     params={"normalise": True, "normaliser_name": "posture_stability_reward"})

    # Shaping reward 2: Penalizes unwanted collisions with the sphere or wall.
    CollisionAvoidanceReward = RewTerm(func=collision_avoidance_reward, weight=0.6,
                                       params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})