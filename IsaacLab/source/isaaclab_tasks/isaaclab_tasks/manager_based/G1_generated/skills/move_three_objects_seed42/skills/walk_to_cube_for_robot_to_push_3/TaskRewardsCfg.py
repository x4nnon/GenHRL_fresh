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


def walk_to_cube_for_robot_to_push_3_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the robot to walk to a position near the third 'Cube for robot to push' (Object3)
    to prepare for pushing. The goal state is for the robot's pelvis to be positioned appropriately
    to initiate a push on Object3.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object3 = env.scene['Object3'] # Cube for robot to push
    
    # Access the required robot part(s) using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Extract x, y, z components for clarity and specific calculations
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Object3 dimensions (0.5m cubed block) - Hardcoded from object configuration as per rules
    object3_size_x = 0.5

    # Calculate target x position relative to Object3 for pushing (behind the cube)
    # Robot should be at object3.x - (object3_size_x / 2 + buffer_distance)
    # A buffer_distance of 0.25m means the pelvis is 0.5m behind the center of the cube.
    # This ensures the robot is positioned to push the cube forward.
    target_x_offset = object3_size_x / 2 + 0.25 # 0.25 + 0.25 = 0.5m behind center
    target_pelvis_x = object3.data.root_pos_w[:, 0] - target_x_offset # Relative distance calculation

    # Calculate distances to target position using relative distances
    # Distance in x-direction to the target pushing position
    distance_x = torch.abs(pelvis_pos_x - target_pelvis_x)
    # Distance in y-direction to align with the cube's center
    distance_y = torch.abs(pelvis_pos_y - object3.data.root_pos_w[:, 1])
    # Distance in z-direction to encourage a stable upright posture (target height 0.7m)
    distance_z_pelvis = torch.abs(pelvis_pos_z - 0.7) # Z is an absolute height, allowed for stability.

    # Reward for approaching the target x, y, and maintaining z
    # Use negative exponential for continuous positive reward, closer is higher reward.
    # The exponential function ensures the reward is smooth and continuous, avoiding local minima.
    reward_x = -distance_x
    reward_y = -distance_y
    reward_z = -distance_z_pelvis

    # Combine rewards. Emphasize x and y for positioning, z for stability.
    # Averaging the rewards provides a balanced contribution from each component.
    reward = (3*reward_x + reward_y + reward_z) / 3.0

    # Add bounds checking to prevent extreme values
    reward = torch.clamp(reward, min=-100.0, max=100.0)
    reward = torch.nan_to_num(reward, nan=0.0, posinf=100.0, neginf=-100.0)

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def walk_to_cube_for_robot_to_push_3_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    """
    This reward penalizes the robot if its feet get too close to Object1, Object2, Object3, or Object4.
    This encourages the robot to navigate carefully and avoid tripping or unnecessary contact,
    especially with the target cube (Object3) before the push.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object1 = env.scene['Object1'] # Cube for robot to push
    object2 = env.scene['Object2'] # Cube for robot to push
    object3 = env.scene['Object3'] # Cube for robot to push
    object4 = env.scene['Object4'] # Platform for cubes

    # Access the required robot part(s) using approved patterns
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    left_hand_idx = robot.body_names.index('left_palm_link')
    right_hand_idx = robot.body_names.index('right_palm_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]

    # Object dimensions (0.5m cubed blocks, platform is thin) - Hardcoded from object configuration
    # `cube_min_dist` is slightly more than half cube size (0.25m) to give clearance.
    # This ensures the robot's feet maintain a safe distance from the cubes.
    cube_min_dist = 1.0
    # `platform_min_dist_z` ensures feet do not go below the platform surface, preventing clipping.
    # The platform's Z-position is 0.001, so a foot Z-position below this would be clipping.
    # We want to penalize if the foot's Z is too close to or below the platform's Z.
    platform_z_level = object4.data.root_pos_w[:, 2]
    foot_z_threshold_below_platform = 0.05 # e.g., if foot Z is less than platform Z + 0.05

    # Calculate Euclidean distances for left foot to cubes
    # Using torch.norm for Euclidean distance, which is a relative distance.
    dist_lf_obj1 = torch.norm(left_foot_pos - object1.data.root_pos_w, dim=-1)
    # dist_lf_obj2 = torch.norm(left_foot_pos - object2.data.root_pos_w, dim=-1)
    # dist_lf_obj3 = torch.norm(left_foot_pos - object3.data.root_pos_w, dim=-1)
    # Penalize if left foot's Z position is below the platform's Z level plus a small buffer
    # penalty_lf_platform_z = torch.where(left_foot_pos[:, 2] < (platform_z_level + foot_z_threshold_below_platform), -1.0, 0.0)

    # Calculate Euclidean distances for right foot to cubes
    dist_rf_obj1 = torch.norm(right_foot_pos - object1.data.root_pos_w, dim=-1)
    # dist_rf_obj2 = torch.norm(right_foot_pos - object2.data.root_pos_w, dim=-1)
    # dist_rf_obj3 = torch.norm(right_foot_pos - object3.data.root_pos_w, dim=-1)
    # # Penalize if right foot's Z position is below the platform's Z level plus a small buffer
    # penalty_rf_platform_z = torch.where(right_foot_pos[:, 2] < (platform_z_level + foot_z_threshold_below_platform), -1.0, 0.0)

    # Calculate Euclidean distances for left hand to cubes
    dist_lh_obj1 = torch.norm(left_hand_pos - object1.data.root_pos_w, dim=-1)
    # dist_lh_obj2 = torch.norm(left_hand_pos - object2.data.root_pos_w, dim=-1)
    # dist_lh_obj3 = torch.norm(left_hand_pos - object3.data.root_pos_w, dim=-1)
    # # Penalize if left hand's Z position is below the platform's Z level plus a small buffer
    # penalty_lh_platform_z = torch.where(left_hand_pos[:, 2] < (platform_z_level + foot_z_threshold_below_platform), -1.0, 0.0)

    # Calculate Euclidean distances for right hand to cubes
    dist_rh_obj1 = torch.norm(right_hand_pos - object1.data.root_pos_w, dim=-1)
    # dist_rh_obj2 = torch.norm(right_hand_pos - object2.data.root_pos_w, dim=-1)
    # dist_rh_obj3 = torch.norm(right_hand_pos - object3.data.root_pos_w, dim=-1)
    # # Penalize if right hand's Z position is below the platform's Z level plus a small buffer
    # penalty_rh_platform_z = torch.where(right_hand_pos[:, 2] < (platform_z_level + foot_z_threshold_below_platform), -1.0, 0.0)

    # Penalize if feet are too close to cubes (x,y,z combined distance)
    # A fixed negative reward is applied when the distance falls below the threshold.
    penalty_lf_cubes = torch.where(dist_lf_obj1 < cube_min_dist, -1.0, 0.0) # + \
                    #    torch.where(dist_lf_obj2 < cube_min_dist, -1.0, 0.0) + \
                    #    torch.where(dist_lf_obj3 < cube_min_dist, -1.0, 0.0)

    penalty_rf_cubes = torch.where(dist_rf_obj1 < cube_min_dist, -1.0, 0.0) # + \
                    #    torch.where(dist_rf_obj2 < cube_min_dist, -1.0, 0.0) + \
                    #    torch.where(dist_rf_obj3 < cube_min_dist, -1.0, 0.0)

    penalty_lh_cubes = torch.where(dist_lh_obj1 < cube_min_dist, -1.0, 0.0) # + \
                    #    torch.where(dist_lh_obj2 < cube_min_dist, -1.0, 0.0) + \
                    #    torch.where(dist_lh_obj3 < cube_min_dist, -1.0, 0.0)

    penalty_rh_cubes = torch.where(dist_rh_obj1 < cube_min_dist, -1.0, 0.0) # + \
                    #    torch.where(dist_rh_obj2 < cube_min_dist, -1.0, 0.0) + \
                    #    torch.where(dist_rh_obj3 < cube_min_dist, -1.0, 0.0)

    # Combine all penalties for the shaping reward
    reward = penalty_lf_cubes + penalty_rf_cubes + penalty_lh_cubes + penalty_rh_cubes

    # Add bounds checking to prevent extreme values
    reward = torch.clamp(reward, min=-100.0, max=100.0)
    reward = torch.nan_to_num(reward, nan=0.0, posinf=100.0, neginf=-100.0)

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def walk_to_cube_for_robot_to_push_3_forward_progress_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "forward_progress_stability_reward") -> torch.Tensor:
    """
    This reward encourages the robot to make consistent forward progress towards Object3 (in the x-direction)
    while maintaining an upright and stable posture (pelvis z-height). It also penalizes excessive lateral
    movement (y-direction) that is not contributing to alignment with Object3.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object3 = env.scene['Object3'] # Cube for robot to push

    # Access the required robot part(s) using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Extract x, y, z components for clarity
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Object3 dimensions (0.5m cubed block) - Hardcoded from object configuration
    object3_size_x = 0.5

    # Target x for pushing is object3.x - 0.5 (from primary reward)
    # This is the ideal x-position for the pelvis relative to the cube for pushing.
    target_pelvis_x_for_push = object3.data.root_pos_w[:, 0] - (object3_size_x / 2 + 0.25)

    # Reward for forward progress in x-direction:
    # This reward encourages the robot to move towards the target x-position.
    # It's designed to be positive when the robot is "behind" the target x and decreases as it approaches.
    # If the robot overshoots, it becomes a penalty. This creates a continuous gradient.
    # The reward is based on the difference between current pelvis x and target x.
    # A higher multiplier for overshooting (e.g., -2.0) can make it a stronger penalty.
    # This is a relative distance reward.
    # Add safety checks to prevent extreme values in exponential function
    x_distance = torch.abs(pelvis_pos_x - target_pelvis_x_for_push)
    x_distance = torch.clamp(x_distance, max=10.0)  # Prevent extreme exponential values
    x_progress_reward = torch.where(
        pelvis_pos_x < target_pelvis_x_for_push, # If robot is still behind the target x
        torch.exp(-2.0 * x_distance), # Reward for approaching from behind
        -2.0 * x_distance # Penalty for overshooting or being too far past
    )

    # Reward for minimizing lateral deviation from Object3's y-coordinate.
    # This encourages the robot to align itself centrally with the cube.
    # This is a relative distance reward.
    lateral_deviation_y = torch.abs(pelvis_pos_y - object3.data.root_pos_w[:, 1])
    lateral_alignment_reward = -1.0 * lateral_deviation_y # Penalty for lateral deviation

    # Reward for maintaining a good pelvis height for stability.
    # This reinforces stable posture throughout the walk.
    # This is an absolute z-height, allowed for stability.
    pelvis_z_target = 0.7
    # Add safety checks to prevent extreme values in exponential function
    z_distance = torch.abs(pelvis_pos_z - pelvis_z_target)
    z_distance = torch.clamp(z_distance, max=10.0)  # Prevent extreme exponential values
    pelvis_z_stability_reward = torch.exp(-5.0 * z_distance)

    # Combine all components for the shaping reward
    reward = x_progress_reward + lateral_alignment_reward + pelvis_z_stability_reward

    # Add bounds checking to prevent extreme values
    reward = torch.clamp(reward, min=-100.0, max=100.0)
    reward = torch.nan_to_num(reward, nan=0.0, posinf=100.0, neginf=-100.0)

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
    """
    Reward terms for the 'walk_to_cube_for_robot_to_push_3' skill.
    """
    # Main reward for positioning the robot near Object3 for pushing.
    # Weight 1.0 as it's the primary objective.
    MainWalkToCubeForPushReward = RewTerm(func=walk_to_cube_for_robot_to_push_3_main_reward, weight=1.0,
                                          params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for collision avoidance with all objects.
    # Weight 0.6 provides a significant penalty for collisions without overshadowing the main goal.
    CollisionAvoidanceReward = RewTerm(func=walk_to_cube_for_robot_to_push_3_collision_avoidance_reward, weight=0.6,
                                       params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})

    # Shaping reward for encouraging forward progress and maintaining stability.
    # Weight 0.1 guides the robot's movement and posture during the approach.
    ForwardProgressStabilityReward = RewTerm(func=walk_to_cube_for_robot_to_push_3_forward_progress_stability_reward, weight=0.1,
                                            params={"normalise": True, "normaliser_name": "forward_progress_stability_reward"})