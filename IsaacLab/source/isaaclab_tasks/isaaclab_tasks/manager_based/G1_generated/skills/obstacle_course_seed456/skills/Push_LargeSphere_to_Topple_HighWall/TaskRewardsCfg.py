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


def main_push_sphere_to_topple_wall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the Push_LargeSphere_to_Topple_HighWall skill.
    This reward guides the robot through phases: approaching the large sphere, pushing it towards the high wall,
    toppling the high wall, and finally positioning itself for the next skill.
    """
    # Get normalizer instance (MANDATORY REWARD NORMALIZATION)
    RewNormalizer = get_normalizer(env.device)

    # Access required objects using approved patterns (USE ONLY APPROVED ACCESS PATTERNS)
    large_sphere = env.scene['Object1']
    high_wall = env.scene['Object4']
    small_sphere = env.scene['Object2']

    # Access required robot part (pelvis) using approved patterns (USE ONLY APPROVED ACCESS PATTERNS)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Hardcoded object dimensions from task description (CRITICAL RULE: NO ACCESSING .size or .radius from object)
    large_sphere_radius = 1.0  # Object1 radius
    high_wall_x_dim = 0.3      # Object4 x-dimension
    high_wall_z_dim = 1.0      # Object4 z-dimension

    # Phase 1: Approach large sphere (Object1)
    # Calculate relative distances between robot pelvis and large sphere (USE RELATIVE DISTANCES)
    dist_pelvis_sphere_x = large_sphere.data.root_pos_w[:, 0] - pelvis_pos[:, 0]
    dist_pelvis_sphere_y = large_sphere.data.root_pos_w[:, 1] - pelvis_pos[:, 1]

    # Reward for approaching and aligning with the large sphere (continuous, negative for distance) (CONTINUOUS REWARDS)
    # Encourages robot to get close to the sphere in both x and y
    reward_approach_sphere = -torch.abs(dist_pelvis_sphere_x) - torch.abs(dist_pelvis_sphere_y)

    # Condition for transitioning to pushing phase: pelvis is close to sphere
    # Pelvis x-position should be slightly behind or at the sphere's front edge to push it
    # Pelvis y-position should be aligned with the sphere
    approach_condition = (pelvis_pos[:, 0] < large_sphere.data.root_pos_w[:, 0] + large_sphere_radius + 0.5) & \
                         (pelvis_pos[:, 0] > large_sphere.data.root_pos_w[:, 0] - large_sphere_radius - 0.5) & \
                         (torch.abs(dist_pelvis_sphere_y) < 1.0) # Within reasonable y-alignment

    # Phase 2 & 3: Push large sphere (Object1) towards high wall (Object4) and topple
    # Calculate relative distances between large sphere and high wall (USE RELATIVE DISTANCES)
    dist_sphere_wall_x = high_wall.data.root_pos_w[:, 0] - large_sphere.data.root_pos_w[:, 0]
    dist_sphere_wall_y = high_wall.data.root_pos_w[:, 1] - large_sphere.data.root_pos_w[:, 1]

    # Reward for moving sphere towards wall and aligning (continuous, negative for distance) (CONTINUOUS REWARDS)
    # Encourages sphere to get close to the wall in both x and y
    reward_push_sphere = -torch.abs(dist_sphere_wall_x) - torch.abs(dist_sphere_wall_y)

    # Check if high wall (Object4) is toppled (its z-position is significantly lower than its initial height)
    # Assuming initial z-pos of wall is around high_wall_z_dim / 2 + ground_level, and it falls to near 0
    # A simple check for toppled: z-position of wall's root is below a threshold (e.g., 0.2m, assuming it falls flat)
    wall_toppled_threshold = 0.2 # If wall's root z-pos is below this, it's considered toppled
    wall_toppled_condition = high_wall.data.root_pos_w[:, 2] < wall_toppled_threshold

    # Reward for toppling the wall (binary for simplicity, but can be shaped)
    # Provides a large positive reward upon successful toppling
    reward_topple_wall = torch.where(wall_toppled_condition, 10.0, 0.0)

    # Phase 4: Final positioning for next skill (avoid overshooting Object2)
    # The robot should ideally be between the high wall and the small sphere, or just past the high wall.
    # Approximate initial x-positions: High wall at ~9m, Small sphere at ~12m.
    # Target x-range for pelvis: Just past the wall to before the small sphere.
    # Target x-min: high_wall_x_pos + high_wall_x_dim/2 + 0.5 (0.5m clearance past wall)
    # Target x-max: small_sphere_x_pos - 1.0 (1.0m before small sphere)
    # Using current object positions for relative target zone (NO HARD-CODED POSITIONS for object locations)
    target_x_min = high_wall.data.root_pos_w[:, 0] + high_wall_x_dim / 2 + 0.5
    target_x_max = small_sphere.data.root_pos_w[:, 0] - 1.0
    target_x_center = (target_x_min + target_x_max) / 2.0

    # Reward for being in the middle of the target zone after the wall is toppled (CONTINUOUS REWARDS)
    # Continuous reward, negative for deviation from target center
    reward_final_pos = -torch.abs(pelvis_pos[:, 0] - target_x_center)
    # This reward is only active once the wall has been toppled
    reward_final_pos = torch.where(wall_toppled_condition, reward_final_pos, 0.0)

    # Combine rewards based on phases using torch.where for smooth transitions (CONTINUOUS REWARDS)
    # If wall is toppled, focus on toppling reward and final positioning.
    # Otherwise, if sphere is very close to wall, increase push reward.
    # Otherwise, if approached, focus on pushing.
    # Otherwise, focus on approaching.
    primary_reward = torch.where(
        wall_toppled_condition,
        reward_topple_wall + reward_final_pos,
        torch.where(
            # Condition: large sphere is very close to the high wall (within 0.1m of contact)
            (large_sphere.data.root_pos_w[:, 0] > high_wall.data.root_pos_w[:, 0] - large_sphere_radius - high_wall_x_dim/2 - 0.1),
            reward_push_sphere * 2.0, # Increase push reward as it gets very close
            torch.where(
                approach_condition,
                reward_push_sphere, # Once approached, focus on pushing sphere
                reward_approach_sphere # Initial phase: approach sphere
            )
        )
    )

    # Normalize and update stats (MANDATORY REWARD NORMALIZATION)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, primary_reward)
        RewNormalizer.update_stats(normaliser_name, primary_reward)
        return scaled_reward
    return primary_reward


def shaping_right_hand_push_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_right_hand_push_reward") -> torch.Tensor:
    """
    Shaping reward 1: Encourages the robot to use its right hand (right_palm_link) to make contact with and push the large sphere (Object1).
    It rewards proximity and alignment of the hand with the sphere.
    """
    # Get normalizer instance (MANDATORY REWARD NORMALIZATION)
    RewNormalizer = get_normalizer(env.device)

    # Access required objects (USE ONLY APPROVED ACCESS PATTERNS)
    large_sphere = env.scene['Object1']
    high_wall = env.scene['Object4'] # Needed for activation condition

    # Access required robot part (right_palm_link) (USE ONLY APPROVED ACCESS PATTERNS)
    robot = env.scene["robot"]
    right_palm_idx = robot.body_names.index('right_palm_link')
    right_palm_pos = robot.data.body_pos_w[:, right_palm_idx]

    # Hardcoded object dimensions (CRITICAL RULE: NO ACCESSING .size or .radius from object)
    large_sphere_radius = 1.0 # Object1 radius
    high_wall_x_dim = 0.3     # Object4 x-dimension

    # Calculate relative distances between right palm and large sphere (USE RELATIVE DISTANCES)
    dist_palm_sphere_x = large_sphere.data.root_pos_w[:, 0] - right_palm_pos[:, 0]
    dist_palm_sphere_y = large_sphere.data.root_pos_w[:, 1] - right_palm_pos[:, 1]
    dist_palm_sphere_z = large_sphere.data.root_pos_w[:, 2] - right_palm_pos[:, 2]

    # Reward for getting hand close to the sphere's surface in x-direction (CONTINUOUS REWARDS)
    # The hand should be slightly behind the sphere's center to push it forward.
    # Target x for palm: sphere_x - sphere_radius (to be at the surface)
    target_palm_x = large_sphere.data.root_pos_w[:, 0] - large_sphere_radius
    reward_palm_x = -torch.abs(right_palm_pos[:, 0] - target_palm_x)

    # Reward for y and z alignment with the sphere's center (CONTINUOUS REWARDS)
    reward_palm_yz = -torch.abs(dist_palm_sphere_y) - torch.abs(dist_palm_sphere_z)

    # Condition: Only active when the robot is generally in the "pushing" phase
    # This means the large sphere has not yet reached the high wall.
    # The sphere should be at least 0.1m away from the wall's front face.
    sphere_not_at_wall_condition = large_sphere.data.root_pos_w[:, 0] < high_wall.data.root_pos_w[:, 0] - high_wall_x_dim/2 - large_sphere_radius - 0.1

    # Apply the reward only when the sphere is not yet at the wall
    shaping_reward_1 = torch.where(sphere_not_at_wall_condition, reward_palm_x + reward_palm_yz, 0.0)

    # Normalize and update stats (MANDATORY REWARD NORMALIZATION)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward_1)
        RewNormalizer.update_stats(normaliser_name, shaping_reward_1)
        return scaled_reward
    return shaping_reward_1


def shaping_pelvis_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_pelvis_stability_reward") -> torch.Tensor:
    """
    Shaping reward 2: Encourages the robot to maintain a stable, upright posture (pelvis_z around 0.7m)
    throughout the skill, and specifically after the wall has been toppled, to ensure it is ready for the next skill.
    It also includes a small penalty for excessive lateral movement (y-axis) to keep the robot on track.
    """
    # Get normalizer instance (MANDATORY REWARD NORMALIZATION)
    RewNormalizer = get_normalizer(env.device)

    # Access required robot part (pelvis) (USE ONLY APPROVED ACCESS PATTERNS)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Target pelvis height for stability (hardcoded, as it's a desired posture)
    target_pelvis_z = 0.7

    # Reward for maintaining target pelvis height (continuous, negative for deviation) (CONTINUOUS REWARDS)
    reward_pelvis_z = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Reward for staying centered on the y-axis (assuming task is primarily along x-axis, target y is 0) (CONTINUOUS REWARDS)
    # Continuous, negative for deviation from y=0
    reward_pelvis_y = -torch.abs(pelvis_pos[:, 1])

    # Check if high wall (Object4) is toppled (same condition as primary reward)
    high_wall = env.scene['Object4']
    wall_toppled_threshold = 0.2
    wall_toppled_condition = high_wall.data.root_pos_w[:, 2] < wall_toppled_threshold

    # Combine stability rewards
    shaping_reward_2 = reward_pelvis_z + reward_pelvis_y
    # Apply higher weight to stability after wall is toppled to emphasize readiness for next skill
    shaping_reward_2 = torch.where(wall_toppled_condition, shaping_reward_2 * 1.5, shaping_reward_2)

    # Normalize and update stats (MANDATORY REWARD NORMALIZATION)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward_2)
        RewNormalizer.update_stats(normaliser_name, shaping_reward_2)
        return scaled_reward
    return shaping_reward_2


@configclass
class TaskRewardsCfg:
    # Primary reward for the main objective: pushing the sphere and toppling the wall (PROPER WEIGHTS)
    MainPushSphereToToppleWallReward = RewTerm(func=main_push_sphere_to_topple_wall_reward, weight=1.0,
                                               params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward 1: Encourages active pushing with the right hand (PROPER WEIGHTS)
    ShapingRightHandPushReward = RewTerm(func=shaping_right_hand_push_reward, weight=0.4,
                                         params={"normalise": True, "normaliser_name": "shaping_right_hand_push_reward"})

    # Shaping reward 2: Encourages stable posture and staying on track (PROPER WEIGHTS)
    ShapingPelvisStabilityReward = RewTerm(func=shaping_pelvis_stability_reward, weight=0.2,
                                           params={"normalise": True, "normaliser_name": "shaping_pelvis_stability_reward"})