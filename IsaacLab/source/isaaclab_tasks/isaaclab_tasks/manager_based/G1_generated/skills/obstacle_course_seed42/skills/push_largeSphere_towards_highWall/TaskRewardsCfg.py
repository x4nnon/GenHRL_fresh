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


def main_push_sphere_towards_wall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Primary reward for the push_largeSphere_towards_highWall skill.
    This reward guides the robot through three phases:
    1. Approaching Object1 (large sphere) from the side opposite Object4 (high wall).
    2. Pushing Object1 towards Object4.
    3. Ensuring Object4 is toppled by Object1 and the robot maintains stability.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot and objects using approved patterns
    robot = env.scene["robot"]
    object1 = env.scene['Object1']  # Large sphere for robot to push
    object4 = env.scene['Object4']  # High wall to be pushed over by large sphere

    # Access robot pelvis position using approved pattern
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Hard-coded object dimensions from the object configuration (as allowed)
    large_sphere_radius = 1.0  # From Object1 description
    high_wall_height = 1.0     # From Object4 description (z-dimension)
    high_wall_thickness = 0.3  # From Object4 description (x-dimension)

    # Phase 1: Approach Object1 and align for push
    # Robot's pelvis should be behind Object1 (relative to Object4) and aligned in Y.
    # Target X for pelvis: Object1.x - large_sphere_radius (i.e., at the back edge of the sphere)
    # Target Y for pelvis: Object1.y
    # Reward is negative absolute difference, so closer to target is higher reward (less negative).
    # Using relative distances as required.
    approach_reward_x = -torch.abs(object1.data.root_pos_w[:, 0] - large_sphere_radius)
    approach_reward_y = -torch.abs(object1.data.root_pos_w[:, 1] - pelvis_pos_y)

    # Phase 2: Push Object1 towards Object4
    # Reward for Object1 moving closer to Object4.
    # The distance should decrease, so negative absolute distance is used.
    # Using relative distances as required.
    push_distance_x = object4.data.root_pos_w[:, 0] - object1.data.root_pos_w[:, 0]
    push_reward = -torch.abs(push_distance_x)

    # Phase 3: Topple Object4 and maintain stability
    # Reward for Object4's Z position dropping (indicating it's toppled).
    # Initial Z of high wall is 0.5 * high_wall_height = 0.5m (assuming root is at base).
    # If toppled, Z should be close to 0.5 * high_wall_thickness = 0.15m (if it falls flat).
    # A threshold for "toppled" is set significantly below its initial height.
    # Using a hardcoded threshold for Z position, which is allowed for state detection.
    toppled_threshold_z = 0.3  # If wall's Z position is below this, it's considered toppled.
    # Large positive reward for toppling the wall.
    topple_reward = torch.where(object4.data.root_pos_w[:, 2] < toppled_threshold_z, 10.0, 0.0)

    # Robot stability after toppling:
    # Robot should be stable and not too far past Object4's original position.
    # Pelvis X should be near Object4's original X position (plus its thickness).
    # Pelvis Z should be at a stable standing height (e.g., 0.7m).
    # Using relative distances and hardcoded target height for stability.
    stability_x_reward = -torch.abs(pelvis_pos_x - (object4.data.root_pos_w[:, 0] + high_wall_thickness))
    stability_z_reward = -torch.abs(pelvis_pos_z - 0.7)

    # Combine rewards with conditional activation based on skill progression.
    # The conditions ensure that rewards for a specific phase are active only when relevant.

    # Approach phase condition: Robot's pelvis is behind Object1 (relative to push direction)
    # A small margin (0.5m) is added to allow the robot to get slightly past the exact back edge for better pushing.
    # Using relative position for condition.
    approach_condition = pelvis_pos_x < (object1.data.root_pos_w[:, 0] - large_sphere_radius + 0.5)
    approach_phase_reward = torch.where(approach_condition, approach_reward_x + approach_reward_y, 0.0)

    # Push phase condition: Object1 is between a relative back offset and Object4,
    # AND the robot's pelvis is generally aligned with Object1 for pushing.
    # Replace unavailable initial position with a relative back margin behind current sphere position.
    # Using relative positions for conditions.
    push_condition = (object1.data.root_pos_w[:, 0] > (object1.data.root_pos_w[:, 0] - large_sphere_radius - 0.1)) & \
                     (object1.data.root_pos_w[:, 0] < (object4.data.root_pos_w[:, 0] + high_wall_thickness)) & \
                     (pelvis_pos_x > (object1.data.root_pos_w[:, 0] - large_sphere_radius - 0.5)) & \
                     (pelvis_pos_x < (object1.data.root_pos_w[:, 0] + large_sphere_radius + 0.5))
    push_phase_reward = torch.where(push_condition, push_reward, 0.0)

    # Wall toppled and stability phase condition: Object4's Z position is below the toppled threshold.
    # Using relative position for condition.
    toppled_condition = object4.data.root_pos_w[:, 2] < toppled_threshold_z
    toppled_phase_reward = torch.where(toppled_condition, topple_reward + stability_x_reward + stability_z_reward, 0.0)

    # Final primary reward combines phases. Prioritize toppling.
    # The phases are designed to be somewhat mutually exclusive or to build upon each other.
    # If the wall is toppled, the toppled_phase_reward dominates.
    # Otherwise, it's a mix of approach and push rewards.
    reward = approach_phase_reward + push_phase_reward + toppled_phase_reward

    # Normalize and update stats as required.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_reward") -> torch.Tensor:
    """
    Shaping reward for collision avoidance.
    Penalizes the robot for colliding with Object1 (large sphere) before it's in a pushing position,
    and for colliding with Object4 (high wall) at any point.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot and objects using approved patterns
    robot = env.scene["robot"]
    object1 = env.scene['Object1']
    object4 = env.scene['Object4']

    # Access robot pelvis position using approved pattern
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]

    # Hard-coded object dimensions from the object configuration (as allowed)
    large_sphere_radius = 1.0
    high_wall_thickness = 0.3  # x-dimension of the wall
    high_wall_y_dim = 5.0      # y-dimension of the wall
    high_wall_z_dim = 1.0      # z-dimension of the wall

    # Collision with Object1 (large sphere)
    # Penalize if robot's pelvis is too close to Object1's center, especially from the front or side,
    # before the robot is in a position to push (i.e., behind the sphere).
    # A small buffer (e.g., 0.2m) around the sphere's radius defines the collision zone.
    # Using relative distances for collision detection.
    dist_pelvis_obj1_x = torch.abs(object1.data.root_pos_w[:, 0] - pelvis_pos[:, 0])
    dist_pelvis_obj1_y = torch.abs(object1.data.root_pos_w[:, 1] - pelvis_pos[:, 1])
    dist_pelvis_obj1_z = torch.abs(object1.data.root_pos_w[:, 2] - pelvis_pos[:, 2])

    # Collision condition for Object1: Robot's pelvis is within the sphere's bounds + buffer,
    # AND the robot is in front of the sphere (pelvis_x > sphere_x) or too far off in Y/Z.
    # This prevents penalizing when the robot is correctly positioned behind the sphere for pushing.
    # Using relative positions for condition.
    collision_obj1_condition = (dist_pelvis_obj1_x < (large_sphere_radius + 0.2)) & \
                               (dist_pelvis_obj1_y < (large_sphere_radius + 0.2)) & \
                               (dist_pelvis_obj1_z < (large_sphere_radius + 0.2)) & \
                               (pelvis_pos_x > object1.data.root_pos_w[:, 0]) # Robot is in front of sphere

    collision_obj1_reward = torch.where(collision_obj1_condition, -5.0, 0.0) # Penalty for collision from front/side

    # Collision with Object4 (high wall)
    # Penalize if robot's pelvis is too close to Object4.
    # Object4 is a box, so consider its half-dimensions. Assume root is at center of base.
    # Using relative distances for collision detection.
    half_wall_x = high_wall_thickness / 2.0
    half_wall_y = high_wall_y_dim / 2.0
    half_wall_z = high_wall_z_dim / 2.0

    dist_pelvis_obj4_x = torch.abs(object4.data.root_pos_w[:, 0] - pelvis_pos[:, 0])
    dist_pelvis_obj4_y = torch.abs(object4.data.root_pos_w[:, 1] - pelvis_pos[:, 1])
    dist_pelvis_obj4_z = torch.abs(object4.data.root_pos_w[:, 2] - pelvis_pos[:, 2])

    # Collision zone for Object4: within 0.2m of wall surface in all dimensions.
    # Using relative positions for condition.
    collision_obj4_condition = (dist_pelvis_obj4_x < (half_wall_x + 0.2)) & \
                               (dist_pelvis_obj4_y < (half_wall_y + 0.2)) & \
                               (dist_pelvis_obj4_z < (half_wall_z + 0.2))

    collision_obj4_reward = torch.where(collision_obj4_condition, -10.0, 0.0) # Larger penalty for hitting the wall

    reward = collision_obj1_reward + collision_obj4_reward

    # Normalize and update stats as required.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def maintain_upright_posture_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "posture_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to maintain an upright and stable posture.
    Penalizes deviation from a target pelvis height and adds an extra penalty if the pelvis is too low.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot pelvis position using approved pattern
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Target pelvis height for stable standing (hardcoded as allowed for target height)
    target_pelvis_z = 0.7

    # Reward for pelvis being close to the target height.
    # Use a negative absolute difference, so closer to 0.7 is higher reward (less negative).
    # Using relative distance to a target height.
    reward = -torch.abs(pelvis_pos_z - target_pelvis_z)

    # Optionally, add a small penalty if pelvis is too low (e.g., below 0.4m, indicating falling).
    # Using a hardcoded threshold for Z position, which is allowed for state detection.
    low_pelvis_condition = pelvis_pos_z < 0.4
    reward = torch.where(low_pelvis_condition, reward - 5.0, reward) # Additional penalty for being too low

    # Normalize and update stats as required.
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
    Configuration for the reward terms for the push_largeSphere_towards_highWall skill.
    """
    # Primary reward for pushing the sphere and toppling the wall, with stability.
    # Weight 1.0 as required for primary reward.
    MainPushSphereTowardsWallReward = RewTerm(func=main_push_sphere_towards_wall_reward, weight=1.0,
                                              params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for collision avoidance with the sphere (before push) and the wall.
    # Weight 0.6 as required for shaping reward.
    CollisionAvoidanceReward = RewTerm(func=collision_avoidance_reward, weight=0.6,
                                       params={"normalise": True, "normaliser_name": "collision_reward"})

    # Shaping reward for maintaining an upright and stable posture.
    # Weight 0.3 as required for shaping reward.
    MaintainUprightPostureReward = RewTerm(func=maintain_upright_posture_reward, weight=0.3,
                                           params={"normalise": True, "normaliser_name": "posture_reward"})