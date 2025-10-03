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

def walk_to_largeSphere_primary_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_largeSphere_primary_reward") -> torch.Tensor:
    """
    Primary reward for the walk_to_largeSphere skill.
    Guides the robot's pelvis towards the desired x, y, and z position relative to the large sphere (Object1)
    for pushing. It encourages the robot to be stable and within a specific range for interaction.
    The target x-position is slightly before the sphere's center to allow for pushing.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot and object data using approved patterns
    robot = env.scene["robot"]
    object_large_sphere = env.scene['Object1'] # Object1 is the large sphere for robot to push

    # Get robot pelvis position
    robot_pelvis_idx = robot.body_names.index('pelvis') # Access robot part index using approved pattern
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx] # Access robot part position using approved pattern

    # Get large sphere position
    large_sphere_pos = object_large_sphere.data.root_pos_w # Access object position using approved pattern

    # Hardcode large sphere radius from object configuration (1m radius)
    large_sphere_radius = 1.0
    # Define a small offset to position the pelvis slightly before the sphere for pushing
    target_x_offset = 0.2

    # Calculate target pelvis positions relative to the large sphere
    # Target x: slightly before the sphere's center
    target_pelvis_x = large_sphere_pos[:, 0] - large_sphere_radius - target_x_offset
    # Target y: aligned with the sphere's y-center
    target_pelvis_y = large_sphere_pos[:, 1]
    # Target z: stable standing height (absolute height from ground)
    target_pelvis_z = 0.7

    # Calculate rewards based on relative distances
    # Reward for x-positioning relative to the sphere (negative absolute difference)
    reward_x = -torch.abs(robot_pelvis_pos[:, 0] - target_pelvis_x)
    # Reward for y-alignment with the sphere (negative absolute difference)
    reward_y = -torch.abs(robot_pelvis_pos[:, 1] - target_pelvis_y)
    # Reward for stable pelvis height (negative absolute difference)
    reward_z = -torch.abs(robot_pelvis_pos[:, 2] - target_pelvis_z)

    # Combine rewards
    reward = reward_x + reward_y + reward_z

    # Normalization implementation (mandatory)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def walk_to_largeSphere_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_largeSphere_collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward for collision avoidance.
    Penalizes the robot for getting too close to the large sphere (Object1) and the high wall (Object4).
    Ensures safe navigation and prevents unwanted collisions.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot and object data
    robot = env.scene["robot"]
    object_large_sphere = env.scene['Object1'] # Object1 is the large sphere
    object_high_wall = env.scene['Object4'] # Object4 is the high wall

    # Get robot pelvis position (used for general proximity checks)
    robot_pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]

    # Get object positions
    large_sphere_pos = object_large_sphere.data.root_pos_w
    high_wall_pos = object_high_wall.data.root_pos_w

    # Hardcode object dimensions from object configuration
    large_sphere_radius = 1.0 # From object configuration
    high_wall_x_dim = 0.3 # From object configuration
    high_wall_y_dim = 5.0 # From object configuration
    high_wall_z_dim = 1.0 # From object configuration

    # Define collision thresholds (relative to object dimensions + a small buffer)
    # For sphere, using a single threshold for all dimensions for simplicity, as it's a sphere.
    collision_threshold_sphere = large_sphere_radius + 0.1 # Sphere radius + small buffer
    # For wall, considering dimensions separately for a box-like object.
    collision_threshold_wall_x = high_wall_x_dim / 2 + 0.1 # Half wall thickness + small buffer
    collision_threshold_wall_y = high_wall_y_dim / 2 + 0.1 # Half wall width + small buffer
    collision_threshold_wall_z = high_wall_z_dim / 2 + 0.1 # Half wall height + small buffer

    reward_collision = torch.zeros_like(robot_pelvis_pos[:, 0]) # Initialize reward tensor

    # Check proximity/collision with large sphere (Object1) using pelvis position
    # Calculate absolute distances in each dimension
    dist_to_sphere_x = torch.abs(robot_pelvis_pos[:, 0] - large_sphere_pos[:, 0])
    dist_to_sphere_y = torch.abs(robot_pelvis_pos[:, 1] - large_sphere_pos[:, 1])
    dist_to_sphere_z = torch.abs(robot_pelvis_pos[:, 2] - large_sphere_pos[:, 2])

    # Check if any dimension is within the collision threshold
    # For a sphere, a simple check if the robot's pelvis is within the sphere's bounding box + buffer.
    # This is a simplified collision check, not a precise one.
    is_colliding_sphere = (dist_to_sphere_x < collision_threshold_sphere) & \
                          (dist_to_sphere_y < collision_threshold_sphere) & \
                          (dist_to_sphere_z < collision_threshold_sphere)

    # Apply a negative reward if colliding with the sphere
    reward_collision += torch.where(is_colliding_sphere, -1.0, 0.0)

    # Check proximity/collision with high wall (Object4) using pelvis position
    # Calculate absolute distances in each dimension
    dist_to_wall_x = torch.abs(robot_pelvis_pos[:, 0] - high_wall_pos[:, 0])
    dist_to_wall_y = torch.abs(robot_pelvis_pos[:, 1] - high_wall_pos[:, 1])
    dist_to_wall_z = torch.abs(robot_pelvis_pos[:, 2] - high_wall_pos[:, 2])

    # Check if any dimension is within the collision threshold
    # For a box, check if the robot's pelvis is within the wall's bounding box + buffer.
    is_colliding_wall = (dist_to_wall_x < collision_threshold_wall_x) & \
                        (dist_to_wall_y < collision_threshold_wall_y) & \
                        (dist_to_wall_z < collision_threshold_wall_z)

    # Apply a negative reward if colliding with the wall
    reward_collision += torch.where(is_colliding_wall, -1.0, 0.0)

    reward = reward_collision

    # Normalization implementation (mandatory)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def walk_to_largeSphere_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_largeSphere_stability_reward") -> torch.Tensor:
    """
    Shaping reward for maintaining stability and alignment.
    Encourages the robot to maintain a stable, upright posture by penalizing large deviations of the pelvis
    from the target z-height (0.7m) and large y-deviations relative to the large sphere,
    which could indicate falling or instability.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot and object data
    robot = env.scene["robot"]
    object_large_sphere = env.scene['Object1'] # Object1 is the large sphere

    # Get robot pelvis position
    robot_pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]

    # Get large sphere position
    large_sphere_pos = object_large_sphere.data.root_pos_w

    # Define target stable pelvis height (absolute height from ground)
    target_pelvis_z = 0.7
    # Define target y-alignment with the sphere's y-center
    target_pelvis_y = large_sphere_pos[:, 1]

    # Penalize large deviations from target pelvis height (relative distance)
    # Using a continuous negative reward based on absolute difference
    reward_pelvis_z_stability = -torch.abs(robot_pelvis_pos[:, 2] - target_pelvis_z) * 0.5

    # Penalize large deviations from target y-alignment with the sphere (relative distance)
    # Using a continuous negative reward based on absolute difference
    reward_pelvis_y_alignment = -torch.abs(robot_pelvis_pos[:, 1] - target_pelvis_y) * 0.5

    # Combine rewards
    reward = reward_pelvis_z_stability + reward_pelvis_y_alignment

    # Normalization implementation (mandatory)
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
    Reward terms for the walk_to_largeSphere skill.
    Defines the primary and shaping rewards with their respective weights and normalization settings.
    """
    # Primary reward for positioning the robot near the large sphere
    WalkToLargeSpherePrimaryReward = RewTerm(
        func=walk_to_largeSphere_primary_reward,
        weight=1.0, # Main reward, typically weight 1.0
        params={"normalise": True, "normaliser_name": "walk_to_largeSphere_primary_reward"}
    )

    # Shaping reward for avoiding collisions with objects
    WalkToLargeSphereCollisionAvoidanceReward = RewTerm(
        func=walk_to_largeSphere_collision_avoidance_reward,
        weight=0.4, # Shaping reward, lower weight than primary
        params={"normalise": True, "normaliser_name": "walk_to_largeSphere_collision_avoidance_reward"}
    )

    # Shaping reward for maintaining robot stability and alignment
    WalkToLargeSphereStabilityReward = RewTerm(
        func=walk_to_largeSphere_stability_reward,
        weight=0.3, # Shaping reward, lower weight than primary
        params={"normalise": True, "normaliser_name": "walk_to_largeSphere_stability_reward"}
    )