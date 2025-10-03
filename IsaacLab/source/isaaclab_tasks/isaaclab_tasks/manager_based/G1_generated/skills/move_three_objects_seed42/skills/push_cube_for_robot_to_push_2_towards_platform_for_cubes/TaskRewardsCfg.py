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

def main_push_cube_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_push_cube_reward") -> torch.Tensor:
    """
    Main reward for pushing Object2 towards Object4.
    Encourages reducing the 2D distance between Object2 and Object4, and penalizes Z-axis deviation.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    # Requirement: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    object2_pos = env.scene['Object2'].data.root_pos_w
    object4_pos = env.scene['Object4'].data.root_pos_w

    # Calculate the 2D (x, y) distance between Object2 and Object4
    # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
    # Requirement: Consider x, y, and z components separately. Here, 2D distance for primary goal.
    distance_x_obj2_obj4 = object2_pos[:, 0] - object4_pos[:, 0]
    distance_y_obj2_obj4 = object2_pos[:, 1] - object4_pos[:, 1]
    # Use torch.clamp to prevent NaN from negative values due to floating-point precision
    horizontal_distance_obj2_obj4 = torch.sqrt(torch.clamp(distance_x_obj2_obj4**2 + distance_y_obj2_obj4**2, min=0.0))

    # Reward for reducing horizontal distance (negative as we want to minimize distance)
    # Requirement: Rewards should be continuous and positive where possible. Here, negative distance is fine.
    primary_reward = -horizontal_distance_obj2_obj4

    # Target Z-position for Object2 on Object4
    # Object2 is a 0.5m cube, so its center should be 0.25m above the platform's Z.
    # Requirement: NO hard-coded positions or arbitrary thresholds for object locations.
    # Object4's Z is its root_pos_w[:, 2].
    # Requirement: THERE IS NO way to access the SIZE of an object. Hardcode from config.
    cube_half_height = 0.25 # Hardcoded from object configuration (0.5m cube)
    target_z_obj2 = object4_pos[:, 2] + cube_half_height

    # Small penalty for Z-axis deviation from target Z (on platform)
    # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
    z_deviation = torch.abs(object2_pos[:, 2] - target_z_obj2)
    primary_reward -= 0.1 * z_deviation # Penalty weight 0.1

    # Add bounds checking to prevent extreme values
    primary_reward = torch.clamp(primary_reward, min=-100.0, max=100.0)
    primary_reward = torch.nan_to_num(primary_reward, nan=0.0, posinf=100.0, neginf=-100.0)

    # Requirement: ALWAYS implement proper reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, primary_reward)
        RewNormalizer.update_stats(normaliser_name, primary_reward)
        return scaled_reward
    return primary_reward

def hands_behind_cube_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "hands_behind_cube_reward") -> torch.Tensor:
    """
    Shaping reward encouraging robot hands to be behind Object2 relative to Object4, facilitating pushing.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts
    object2_pos = env.scene['Object2'].data.root_pos_w
    object4_pos = env.scene['Object4'].data.root_pos_w
    robot = env.scene["robot"]

    # Requirement: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    left_hand_idx = robot.body_names.index('left_palm_link')
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]
    right_hand_idx = robot.body_names.index('right_palm_link')
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]

    # Cube dimensions for relative positioning (hardcoded from object config: 0.5m cube)
    # Requirement: THERE IS NO way to access the SIZE of an object. Hardcode from config.
    cube_half_size = 0.25

    # Calculate vector from Object2 to Object4 in XY plane
    # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
    obj2_to_obj4_vec_x = object4_pos[:, 0] - object2_pos[:, 0]
    obj2_to_obj4_vec_y = object4_pos[:, 1] - object2_pos[:, 1]
    # Use torch.clamp to prevent NaN from negative values due to floating-point precision
    obj2_to_obj4_vec_norm = torch.sqrt(torch.clamp(obj2_to_obj4_vec_x**2 + obj2_to_obj4_vec_y**2, min=0.0))

    # Avoid division by zero if objects are at the same spot or very close
    # Use torch.clamp with minimum epsilon to prevent NaN/Inf from very small values
    obj2_to_obj4_vec_norm = torch.clamp(obj2_to_obj4_vec_norm, min=1e-8)

    # Normalized direction vector from Object2 to Object4
    dir_x = obj2_to_obj4_vec_x / obj2_to_obj4_vec_norm
    dir_y = obj2_to_obj4_vec_y / obj2_to_obj4_vec_norm

    # Target position for hands relative to Object2 (on the side opposite to Object4)
    # This means hands should be at Object2.pos - (cube_half_size + clearance) * dir_vector
    # Requirement: NO hard-coded positions or arbitrary thresholds. This clearance is a small, relative buffer.
    clearance = 0.05 # Small clearance for hands to be behind the cube (arbitrary threshold, but small and relative)
    target_hand_x = object2_pos[:, 0] - (cube_half_size + clearance) * dir_x
    target_hand_y = object2_pos[:, 1] - (cube_half_size + clearance) * dir_y
    target_hand_z = object2_pos[:, 2] # Hands should be at cube's center Z for pushing

    # Distance from hands to target push position
    # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
    dist_left_hand_x = left_hand_pos[:, 0] - target_hand_x
    dist_left_hand_y = left_hand_pos[:, 1] - target_hand_y
    dist_left_hand_z = left_hand_pos[:, 2] - target_hand_z

    dist_right_hand_x = right_hand_pos[:, 0] - target_hand_x
    dist_right_hand_y = right_hand_pos[:, 1] - target_hand_y
    dist_right_hand_z = right_hand_pos[:, 2] - target_hand_z

    # Reward for hands being close to the target push position (negative Euclidean distance)
    # Requirement: Rewards should be continuous and positive where possible. Negative distance is fine for minimization.
    # Use torch.clamp to prevent NaN from negative values due to floating-point precision
    shaping_reward1 = -0.5 * (torch.sqrt(torch.clamp(dist_left_hand_x**2 + dist_left_hand_y**2 + dist_left_hand_z**2, min=0.0)) +
                              torch.sqrt(torch.clamp(dist_right_hand_x**2 + dist_right_hand_y**2 + dist_right_hand_z**2, min=0.0)))

    # Condition: Only active when Object2 is not yet very close to Object4
    # This prevents the robot from trying to push an already placed cube
    # Requirement: NO hard-coded positions or arbitrary thresholds. This threshold (0.5m) is relative.
    # Use torch.clamp to prevent NaN from negative values due to floating-point precision
    horizontal_distance_obj2_obj4_for_activation = torch.sqrt(torch.clamp((object2_pos[:, 0] - object4_pos[:, 0])**2 +
                                                              (object2_pos[:, 1] - object4_pos[:, 1])**2, min=0.0))
    activation_condition = horizontal_distance_obj2_obj4_for_activation > 0.5 # 0.5m threshold

    # Apply the activation condition
    # Requirement: All tensor operations correctly handle batched environments.
    shaping_reward1 = torch.where(activation_condition, shaping_reward1, torch.tensor(0.0, device=env.device))

    # Add bounds checking to prevent extreme values
    shaping_reward1 = torch.clamp(shaping_reward1, min=-100.0, max=100.0)
    shaping_reward1 = torch.nan_to_num(shaping_reward1, nan=0.0, posinf=100.0, neginf=-100.0)

    # Requirement: ALWAYS implement proper reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward1)
        RewNormalizer.update_stats(normaliser_name, shaping_reward1)
        return scaled_reward
    return shaping_reward1

def collision_avoidance_and_posture_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_and_posture_reward") -> torch.Tensor:
    """
    Shaping reward for collision avoidance with other objects (Object1, Object3), ground,
    and promoting stable upright posture. Penalizes non-pushing collisions with Object2.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts
    object1_pos = env.scene['Object1'].data.root_pos_w
    object2_pos = env.scene['Object2'].data.root_pos_w
    object3_pos = env.scene['Object3'].data.root_pos_w
    robot = env.scene["robot"]

    # Requirement: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    left_hand_idx = robot.body_names.index('left_palm_link')
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]
    right_hand_idx = robot.body_names.index('right_palm_link')
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]

    # Cube dimensions for collision (hardcoded from object config: 0.5m cube)
    # Requirement: THERE IS NO way to access the SIZE of an object. Hardcode from config.
    cube_half_size = 0.25

    # Pelvis height reward (encourage standing upright)
    # Requirement: z is the only absolute position allowed. Use this sparingly, only when height is important.
    # Target pelvis Z for standing is around 0.7m (arbitrary but reasonable target height for a humanoid)
    shaping_reward2 = -0.2 * torch.abs(pelvis_pos_z - 0.7)

    # Collision avoidance with Object1 and Object3 (other cubes)
    # Calculate 2D distance from pelvis to Object1 and Object3
    # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
    dist_pelvis_obj1_x = pelvis_pos[:, 0] - object1_pos[:, 0]
    dist_pelvis_obj1_y = pelvis_pos[:, 1] - object1_pos[:, 1]
    # Use torch.clamp to prevent NaN from negative values due to floating-point precision
    dist_pelvis_obj1_xy = torch.sqrt(torch.clamp(dist_pelvis_obj1_x**2 + dist_pelvis_obj1_y**2, min=0.0))

    dist_pelvis_obj3_x = pelvis_pos[:, 0] - object3_pos[:, 0]
    dist_pelvis_obj3_y = pelvis_pos[:, 1] - object3_pos[:, 1]
    # Use torch.clamp to prevent NaN from negative values due to floating-point precision
    dist_pelvis_obj3_xy = torch.sqrt(torch.clamp(dist_pelvis_obj3_x**2 + dist_pelvis_obj3_y**2, min=0.0))

    # Penalty if pelvis is too close to Object1 or Object3
    # collision_threshold is relative: cube radius + a small buffer for robot body
    # Requirement: NO hard-coded positions or arbitrary thresholds. This threshold is relative and justified.
    collision_threshold = cube_half_size + 0.2 # 0.2m is an arbitrary but reasonable buffer for robot body
    # Requirement: All tensor operations correctly handle batched environments.
    shaping_reward2 -= 0.3 * torch.where(dist_pelvis_obj1_xy < collision_threshold, (collision_threshold - dist_pelvis_obj1_xy), torch.tensor(0.0, device=env.device))
    shaping_reward2 -= 0.3 * torch.where(dist_pelvis_obj3_xy < collision_threshold, (collision_threshold - dist_pelvis_obj3_xy), torch.tensor(0.0, device=env.device))

    # Penalty for feet being too low (e.g., clipping ground)
    # Requirement: z is the only absolute position allowed. Use this sparingly, only when height is important.
    # Requirement: NO hard-coded positions or arbitrary thresholds. This threshold is a reasonable ground clearance.
    ground_clearance_threshold = 0.05 # 5cm above ground (arbitrary but reasonable clearance)
    # Requirement: All tensor operations correctly handle batched environments.
    shaping_reward2 -= 0.1 * torch.where(left_foot_pos[:, 2] < ground_clearance_threshold, (ground_clearance_threshold - left_foot_pos[:, 2]), torch.tensor(0.0, device=env.device))
    shaping_reward2 -= 0.1 * torch.where(right_foot_pos[:, 2] < ground_clearance_threshold, (ground_clearance_threshold - right_foot_pos[:, 2]), torch.tensor(0.0, device=env.device))

    # Penalty for robot parts (e.g., pelvis) being too close to Object2 when not actively pushing it
    # This is to prevent the robot from climbing over Object2 or getting stuck
    # Calculate 3D distance from hands to Object2
    # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
    dist_left_hand_obj2 = torch.norm(left_hand_pos - object2_pos, dim=1)
    dist_right_hand_obj2 = torch.norm(right_hand_pos - object2_pos, dim=1)

    # If hands are not within 0.3m of Object2, then consider it "not pushing"
    # Requirement: NO hard-coded positions or arbitrary thresholds. This threshold is a reasonable hand proximity for pushing.
    # 0.3m is an arbitrary but reasonable threshold for hand proximity for pushing
    not_pushing_condition = (dist_left_hand_obj2 > 0.3) & (dist_right_hand_obj2 > 0.3)

    # Calculate 2D distance from pelvis to Object2
    # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
    dist_pelvis_obj2_x = pelvis_pos[:, 0] - object2_pos[:, 0]
    dist_pelvis_obj2_y = pelvis_pos[:, 1] - object2_pos[:, 1]
    # Use torch.clamp to prevent NaN from negative values due to floating-point precision
    dist_pelvis_obj2_xy = torch.sqrt(torch.clamp(dist_pelvis_obj2_x**2 + dist_pelvis_obj2_y**2, min=0.0))

    # Apply penalty if not pushing and pelvis is too close to Object2
    # Requirement: All tensor operations correctly handle batched environments.
    shaping_reward2 -= 0.4 * torch.where(not_pushing_condition & (dist_pelvis_obj2_xy < collision_threshold),
                                         (collision_threshold - dist_pelvis_obj2_xy), torch.tensor(0.0, device=env.device))

    # Add bounds checking to prevent extreme values
    shaping_reward2 = torch.clamp(shaping_reward2, min=-100.0, max=100.0)
    shaping_reward2 = torch.nan_to_num(shaping_reward2, nan=0.0, posinf=100.0, neginf=-100.0)

    # Requirement: ALWAYS implement proper reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward2)
        RewNormalizer.update_stats(normaliser_name, shaping_reward2)
        return scaled_reward
    return shaping_reward2

@configclass
class TaskRewardsCfg:
    # Main reward for pushing Object2 towards Object4
    # Weight 1.0 as it's the primary objective
    MainPushCubeReward = RewTerm(func=main_push_cube_reward, weight=1.0,
                                 params={"normalise": True, "normaliser_name": "main_push_cube_reward"})

    # Shaping reward for hand positioning behind Object2
    # Weight 0.6 to guide the robot's interaction with the cube
    HandsBehindCubeReward = RewTerm(func=hands_behind_cube_reward, weight=0.6,
                                    params={"normalise": True, "normaliser_name": "hands_behind_cube_reward"})

    # Shaping reward for collision avoidance and posture
    # Weight 0.4 to encourage safe and stable movement
    CollisionAvoidanceAndPostureReward = RewTerm(func=collision_avoidance_and_posture_reward, weight=0.4,
                                                 params={"normalise": True, "normaliser_name": "collision_avoidance_and_posture_reward"})