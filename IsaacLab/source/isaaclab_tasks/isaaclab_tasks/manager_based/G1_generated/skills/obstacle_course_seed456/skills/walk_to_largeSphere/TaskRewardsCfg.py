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


def walk_to_largeSphere_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_largeSphere_main_reward") -> torch.Tensor:
    """
    Main reward for the walk_to_largeSphere skill.
    Encourages the robot's pelvis to be positioned at a specific x-distance from the large sphere (Object1)
    to be ready to push it towards the high wall (Object4), while also aligning in y and maintaining a stable height.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts
    robot = env.scene["robot"] # Accessing robot using approved pattern
    large_sphere = env.scene['Object1'] # Accessing Object1 (large sphere) using approved pattern
    high_wall = env.scene['Object4'] # Accessing Object4 (high wall) for contextual direction

    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern

    large_sphere_pos = large_sphere.data.root_pos_w # Accessing object position using approved pattern

    # Object dimensions (hardcoded from description as per rules)
    large_sphere_radius = 1.0 # Object1 is 1m radius

    # Calculate target x-position for pelvis relative to large sphere
    # The robot should be positioned to push the sphere towards the high wall (positive x-direction).
    # This means the robot's pelvis should be on the side of the sphere with a smaller x-coordinate.
    # Target x-position for pelvis: large_sphere.x - (large_sphere_radius + desired_clearance)
    # Desired clearance for pushing, e.g., 0.3m from sphere surface.
    # So, pelvis should be 1.3m from the sphere's center in the negative x direction.
    target_pelvis_x_offset_from_sphere_center = - (large_sphere_radius + 0.3) # Pelvis should be 1.3m behind sphere center

    # 1. Reward for reaching the target x-position relative to the sphere
    # The robot should be at large_sphere.x + target_pelvis_x_offset_from_sphere_center
    # Error is (pelvis_pos_x - (large_sphere_pos_x + target_pelvis_x_offset_from_sphere_center))
    # Using torch.abs for continuous, negative reward based on distance from target.
    reward_x_positioning = -torch.abs(pelvis_pos[:, 0] - (large_sphere_pos[:, 0] + target_pelvis_x_offset_from_sphere_center))

    # 2. Reward for y-alignment with the sphere
    # The robot should be aligned with the sphere's y-position.
    reward_y_alignment = -torch.abs(pelvis_pos[:, 1] - large_sphere_pos[:, 1])

    # 3. Reward for maintaining stable pelvis height
    # Target pelvis z is 0.7m for stable standing.
    target_pelvis_z = 0.7
    reward_pelvis_z = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Combine primary rewards with weights.
    # X-positioning is most critical for "walk_to" and "position_to_push".
    # Y-alignment and pelvis_z are for fine-tuning and stability.
    reward = (reward_x_positioning * 0.6) + (reward_y_alignment * 0.2) + (reward_pelvis_z * 0.2)

    # Normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def walk_to_largeSphere_collision_avoidance_low_wall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_largeSphere_collision_avoidance_low_wall_reward") -> torch.Tensor:
    """
    Penalizes the robot if its pelvis gets too close to the low wall (Object3) after it has passed it.
    This ensures the robot doesn't backtrack or collide with the wall it just jumped over.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts
    robot = env.scene["robot"] # Accessing robot using approved pattern
    low_wall = env.scene['Object3'] # Accessing Object3 (low wall) using approved pattern

    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern

    low_wall_pos = low_wall.data.root_pos_w # Accessing object position using approved pattern

    # Object dimensions (hardcoded from description as per rules)
    low_wall_x_dim = 0.3 # Object3 is 0.3m in x axis

    # Calculate the x-distance from the pelvis to the low wall's far side (in positive x direction)
    # Assuming low_wall_pos[:, 0] is the center of the wall.
    low_wall_far_x = low_wall_pos[:, 0] + (low_wall_x_dim / 2.0)

    # The robot should be moving away from the low wall.
    # We want to penalize if the robot's pelvis x-position is less than a certain threshold relative to the wall.
    # This threshold should be slightly past the wall.
    # Let's say the robot should maintain an x-position greater than low_wall_far_x + 0.2m.

    # Calculate the difference from the desired minimum x-position
    x_diff = (low_wall_far_x + 0.2) - pelvis_pos[:, 0]

    # Apply penalty only if x_diff is positive (i.e., pelvis_x is less than the threshold)
    # Use torch.relu to ensure penalty is 0 or positive, then negate for reward.
    reward = -torch.relu(x_diff) * 2.0 # Multiply by a factor to make it more impactful

    # Normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def walk_to_largeSphere_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_largeSphere_stability_reward") -> torch.Tensor:
    """
    Rewards the robot for maintaining a stable base (pelvis) orientation.
    This encourages stable walking and standing.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot object
    robot = env.scene["robot"] # Accessing robot using approved pattern

    # Get pelvis orientation (quaternion)
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern
    pelvis_quat = robot.data.body_quat_w[:, pelvis_idx] # Accessing robot part orientation using approved pattern

    # For stability, we want roll and pitch to be close to zero.
    # A common way to penalize deviation from upright is to use the dot product of the up-vector
    # of the body with the world's up-vector (0,0,1).
    # The z-axis of the body frame (0,0,1) rotated by pelvis_quat gives its world-frame up-vector.
    # The z-component of the rotated z-axis (R_zz) is 1 - 2*(x^2 + y^2) for a unit quaternion [w, x, y, z].
    pelvis_up_alignment = 1.0 - 2.0 * (pelvis_quat[:, 1]**2 + pelvis_quat[:, 2]**2) # 1 - 2*(qx^2 + qy^2) for R_zz

    # Reward for being upright (closer to 1.0)
    # We want pelvis_up_alignment to be close to 1.0.
    # Reward is negative of the deviation from 1.0.
    reward = -torch.abs(pelvis_up_alignment - 1.0)

    # Normalization implementation
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
    """
    # Main reward for positioning the robot's pelvis relative to the large sphere.
    WalkToLargeSphereMainReward = RewTerm(
        func=walk_to_largeSphere_main_reward,
        weight=1.0, # Primary reward, highest weight
        params={"normalise": True, "normaliser_name": "walk_to_largeSphere_main_reward"}
    )

    # Supporting reward to penalize moving back towards the low wall after passing it.
    WalkToLargeSphereCollisionAvoidanceLowWallReward = RewTerm(
        func=walk_to_largeSphere_collision_avoidance_low_wall_reward,
        weight=0.2, # Lower weight as a shaping reward
        params={"normalise": True, "normaliser_name": "walk_to_largeSphere_collision_avoidance_low_wall_reward"}
    )

    # Supporting reward for maintaining pelvis stability (upright orientation).
    WalkToLargeSphereStabilityReward = RewTerm(
        func=walk_to_largeSphere_stability_reward,
        weight=0.1, # Lower weight for general stability
        params={"normalise": True, "normaliser_name": "walk_to_largeSphere_stability_reward"}
    )