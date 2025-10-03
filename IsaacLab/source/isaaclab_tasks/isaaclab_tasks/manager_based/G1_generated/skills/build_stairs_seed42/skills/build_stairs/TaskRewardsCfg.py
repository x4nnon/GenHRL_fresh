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


def main_build_stairs_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_build_stairs_reward") -> torch.Tensor:
    """
    Main reward for the build_stairs skill.
    Rewards the robot for pushing the small, medium, and large blocks into a stair-like configuration.
    The reward is structured sequentially: Object1 first, then Object2 relative to Object1, then Object3 relative to Object2.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access objects using approved patterns
    object1 = env.scene['Object1'] # Small Block
    object2 = env.scene['Object2'] # Medium Block
    object3 = env.scene['Object3'] # Large Block

    # Access robot parts using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]

    # Hardcode block dimensions from the skill description (x=1m y=1m)
    # Object1: z=0.3m, Object2: z=0.6m, Object3: z=0.9m
    obj1_z_dim = 0.3
    obj2_z_dim = 0.6
    obj3_z_dim = 0.9
    obj_xy_dim = 1.0 # All blocks have 1m x 1m base

    # Define target positions for blocks using relative distances
    # Target for Object1 (Small Block): 1.5m in front of the robot's pelvis (along y-axis), centered on x-axis.
    # Z-target is half its height, assuming ground is at z=0.
    target_obj1_x_rel_pelvis = 0.0
    target_obj1_y_rel_pelvis = 1.5
    target_obj1_z_abs = obj1_z_dim / 2.0

    # Target for Object2 (Medium Block): Relative to Object1.
    # Offset in x and y for stair step, and stacked on top of Object1.
    # A reasonable step offset could be half the block's side length.
    step_offset_xy = obj_xy_dim / 2.0
    target_obj2_x_rel_obj1 = step_offset_xy
    target_obj2_y_rel_obj1 = step_offset_xy
    target_obj2_z_rel_obj1 = obj1_z_dim + (obj2_z_dim / 2.0) # Stacked on Object1

    # Target for Object3 (Large Block): Relative to Object2.
    # Offset in x and y for stair step, and stacked on top of Object2.
    target_obj3_x_rel_obj2 = step_offset_xy
    target_obj3_y_rel_obj2 = step_offset_xy
    target_obj3_z_rel_obj2 = obj2_z_dim + (obj3_z_dim / 2.0) # Stacked on Object2

    # Calculate current distances for each block to its target
    # Reward for Object1: Distance to its target relative to pelvis
    dist_obj1_x = object1.data.root_pos_w[:, 0] - (pelvis_pos[:, 0] + target_obj1_x_rel_pelvis)
    dist_obj1_y = object1.data.root_pos_w[:, 1] - (pelvis_pos[:, 1] + target_obj1_y_rel_pelvis)
    dist_obj1_z = object1.data.root_pos_w[:, 2] - target_obj1_z_abs # Z target is absolute ground level for first block

    # Reward for Object2: Distance to its target relative to Object1
    dist_obj2_x = object2.data.root_pos_w[:, 0] - (object1.data.root_pos_w[:, 0] + target_obj2_x_rel_obj1)
    dist_obj2_y = object2.data.root_pos_w[:, 1] - (object1.data.root_pos_w[:, 1] + target_obj2_y_rel_obj1)
    dist_obj2_z = object2.data.root_pos_w[:, 2] - target_obj2_z_rel_obj1

    # Reward for Object3: Distance to its target relative to Object2
    dist_obj3_x = object3.data.root_pos_w[:, 0] - (object2.data.root_pos_w[:, 0] + target_obj3_x_rel_obj2)
    dist_obj3_y = object3.data.root_pos_w[:, 1] - (object2.data.root_pos_w[:, 1] + target_obj3_y_rel_obj2)
    dist_obj3_z = object3.data.root_pos_w[:, 2] - target_obj3_z_rel_obj2

    # Define tolerances for "in place" conditions
    # X/Y tolerance can be larger as blocks are 1m wide, Z tolerance smaller for stacking accuracy
    xy_tolerance = 0.2 # 20cm tolerance for horizontal position
    z_tolerance = 0.1 # 10cm tolerance for vertical position

    # Reward for pushing Object1 towards its target
    # Using negative absolute distance for continuous reward, closer to 0 is better
    reward_obj1_push = -torch.abs(dist_obj1_x) - torch.abs(dist_obj1_y) - torch.abs(dist_obj1_z)

    # Condition for Object1 being "in place"
    obj1_in_place_condition = (torch.abs(dist_obj1_x) < xy_tolerance) & \
                              (torch.abs(dist_obj1_y) < xy_tolerance) & \
                              (torch.abs(dist_obj1_z) < z_tolerance)

    # Reward for pushing Object2, active only if Object1 is reasonably in place
    reward_obj2_push = -torch.abs(dist_obj2_x) - torch.abs(dist_obj2_y) - torch.abs(dist_obj2_z)
    reward_obj2_push = torch.where(obj1_in_place_condition, reward_obj2_push, torch.tensor(0.0, device=env.device))

    # Condition for Object2 being "in place" (relative to Object1)
    obj2_in_place_condition = (torch.abs(dist_obj2_x) < xy_tolerance) & \
                              (torch.abs(dist_obj2_y) < xy_tolerance) & \
                              (torch.abs(dist_obj2_z) < z_tolerance)

    # Reward for pushing Object3, active only if Object2 is reasonably in place
    reward_obj3_push = -torch.abs(dist_obj3_x) - torch.abs(dist_obj3_y) - torch.abs(dist_obj3_z)
    reward_obj3_push = torch.where(obj2_in_place_condition, reward_obj3_push, torch.tensor(0.0, device=env.device))

    # Combine rewards. The sequential activation ensures the robot focuses on one block at a time.
    # A small bonus for completing each stage can be added for stronger signal.
    reward = reward_obj1_push + reward_obj2_push + reward_obj3_push

    # Add completion bonuses for reaching each stage
    reward += torch.where(obj1_in_place_condition, torch.tensor(1.0, device=env.device), torch.tensor(0.0, device=env.device))
    reward += torch.where(obj2_in_place_condition, torch.tensor(2.0, device=env.device), torch.tensor(0.0, device=env.device))
    # Define obj3_in_place_condition analogous to obj1/obj2
    obj3_in_place_condition = (torch.abs(dist_obj3_x) < xy_tolerance) & \
                              (torch.abs(dist_obj3_y) < xy_tolerance) & \
                              (torch.abs(dist_obj3_z) < z_tolerance)
    reward += torch.where(obj3_in_place_condition, torch.tensor(3.0, device=env.device), torch.tensor(0.0, device=env.device))


    # Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_approach_and_hand_proximity_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_approach_hands") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to approach the current target block and position its hands close for pushing.
    Activates sequentially for each block.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access objects
    object1 = env.scene['Object1'] # Small Block
    object2 = env.scene['Object2'] # Medium Block
    object3 = env.scene['Object3'] # Large Block

    # Access robot parts
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_palm_idx = robot.body_names.index('left_palm_link')
    left_palm_pos = robot.data.body_pos_w[:, left_palm_idx]
    right_palm_idx = robot.body_names.index('right_palm_link')
    right_palm_pos = robot.data.body_pos_w[:, right_palm_idx]

    # Define tolerances for "in place" conditions (re-using from main reward for consistency)
    xy_tolerance = 0.2
    z_tolerance = 0.1

    # Re-calculate conditions for sequential activation
    # Object1 target (relative to pelvis)
    obj1_z_dim = 0.3
    target_obj1_x_rel_pelvis = 0.0
    target_obj1_y_rel_pelvis = 1.5
    target_obj1_z_abs = obj1_z_dim / 2.0
    dist_obj1_x = object1.data.root_pos_w[:, 0] - (pelvis_pos[:, 0] + target_obj1_x_rel_pelvis)
    dist_obj1_y = object1.data.root_pos_w[:, 1] - (pelvis_pos[:, 1] + target_obj1_y_rel_pelvis)
    dist_obj1_z = object1.data.root_pos_w[:, 2] - target_obj1_z_abs
    obj1_in_place_condition = (torch.abs(dist_obj1_x) < xy_tolerance) & \
                              (torch.abs(dist_obj1_y) < xy_tolerance) & \
                              (torch.abs(dist_obj1_z) < z_tolerance)

    # Object2 target (relative to Object1)
    obj_xy_dim = 1.0
    obj2_z_dim = 0.6
    step_offset_xy = obj_xy_dim / 2.0
    target_obj2_x_rel_obj1 = step_offset_xy
    target_obj2_y_rel_obj1 = step_offset_xy
    target_obj2_z_rel_obj1 = obj1_z_dim + (obj2_z_dim / 2.0)
    dist_obj2_x = object2.data.root_pos_w[:, 0] - (object1.data.root_pos_w[:, 0] + target_obj2_x_rel_obj1)
    dist_obj2_y = object2.data.root_pos_w[:, 1] - (object1.data.root_pos_w[:, 1] + target_obj2_y_rel_obj1)
    dist_obj2_z = object2.data.root_pos_w[:, 2] - target_obj2_z_rel_obj1
    obj2_in_place_condition = (torch.abs(dist_obj2_x) < xy_tolerance) & \
                              (torch.abs(dist_obj2_y) < xy_tolerance) & \
                              (torch.abs(dist_obj2_z) < z_tolerance)

    # Object3 target (relative to Object2)
    obj3_z_dim = 0.9
    target_obj3_x_rel_obj2 = step_offset_xy
    target_obj3_y_rel_obj2 = step_offset_xy
    target_obj3_z_rel_obj2 = obj2_z_dim + (obj3_z_dim / 2.0)
    dist_obj3_x = object3.data.root_pos_w[:, 0] - (object2.data.root_pos_w[:, 0] + target_obj3_x_rel_obj2)
    dist_obj3_y = object3.data.root_pos_w[:, 1] - (object2.data.root_pos_w[:, 1] + target_obj3_y_rel_obj2)
    dist_obj3_z = object3.data.root_pos_w[:, 2] - target_obj3_z_rel_obj2
    obj3_in_place_condition = (torch.abs(dist_obj3_x) < xy_tolerance) & \
                              (torch.abs(dist_obj3_y) < xy_tolerance) & \
                              (torch.abs(dist_obj3_z) < z_tolerance)

    # Phase 1: Approach Object1 and get hands close
    # Active when Object1 is NOT yet in place
    obj1_not_in_place_condition = ~obj1_in_place_condition

    # Distance from pelvis to Object1 (XY plane for approach)
    dist_pelvis_obj1_xy = torch.norm(object1.data.root_pos_w[:, :2] - pelvis_pos[:, :2], dim=-1)
    reward_approach_obj1 = -dist_pelvis_obj1_xy

    # Distance from hands to Object1 (3D distance for proximity)
    dist_left_hand_obj1 = torch.norm(object1.data.root_pos_w - left_palm_pos, dim=-1)
    dist_right_hand_obj1 = torch.norm(object1.data.root_pos_w - right_palm_pos, dim=-1)
    reward_hands_obj1 = -torch.min(dist_left_hand_obj1, dist_right_hand_obj1)
    # Cap the negative reward to prevent it from dominating when hands are very far
    reward_hands_obj1 = torch.where(reward_hands_obj1 > -0.5, reward_hands_obj1, torch.tensor(-0.5, device=env.device))

    reward_phase1 = (reward_approach_obj1 * 0.5 + reward_hands_obj1 * 0.5)
    reward_phase1 = torch.where(obj1_not_in_place_condition, reward_phase1, torch.tensor(0.0, device=env.device))

    # Phase 2: Approach Object2 and get hands close
    # Active when Object1 is in place AND Object2 is NOT yet in place
    obj2_not_in_place_condition = ~obj2_in_place_condition
    condition_phase2 = obj1_in_place_condition & obj2_not_in_place_condition

    dist_pelvis_obj2_xy = torch.norm(object2.data.root_pos_w[:, :2] - pelvis_pos[:, :2], dim=-1)
    reward_approach_obj2 = -dist_pelvis_obj2_xy

    dist_left_hand_obj2 = torch.norm(object2.data.root_pos_w - left_palm_pos, dim=-1)
    dist_right_hand_obj2 = torch.norm(object2.data.root_pos_w - right_palm_pos, dim=-1)
    reward_hands_obj2 = -torch.min(dist_left_hand_obj2, dist_right_hand_obj2)
    reward_hands_obj2 = torch.where(reward_hands_obj2 > -0.5, reward_hands_obj2, torch.tensor(-0.5, device=env.device))

    reward_phase2 = (reward_approach_obj2 * 0.5 + reward_hands_obj2 * 0.5)
    reward_phase2 = torch.where(condition_phase2, reward_phase2, torch.tensor(0.0, device=env.device))

    # Phase 3: Approach Object3 and get hands close
    # Active when Object2 is in place AND Object3 is NOT yet in place
    obj3_not_in_place_condition = ~obj3_in_place_condition
    condition_phase3 = obj2_in_place_condition & obj3_not_in_place_condition

    dist_pelvis_obj3_xy = torch.norm(object3.data.root_pos_w[:, :2] - pelvis_pos[:, :2], dim=-1)
    reward_approach_obj3 = -dist_pelvis_obj3_xy

    dist_left_hand_obj3 = torch.norm(object3.data.root_pos_w - left_palm_pos, dim=-1)
    dist_right_hand_obj3 = torch.norm(object3.data.root_pos_w - right_palm_pos, dim=-1)
    reward_hands_obj3 = -torch.min(dist_left_hand_obj3, dist_right_hand_obj3)
    reward_hands_obj3 = torch.where(reward_hands_obj3 > -0.5, reward_hands_obj3, torch.tensor(-0.5, device=env.device))

    reward_phase3 = (reward_approach_obj3 * 0.5 + reward_hands_obj3 * 0.5)
    reward_phase3 = torch.where(condition_phase3, reward_phase3, torch.tensor(0.0, device=env.device))

    reward = reward_phase1 + reward_phase2 + reward_phase3

    # Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_stability_and_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_stability_collision") -> torch.Tensor:
    """
    Shaping reward for collision avoidance (robot body with blocks) and robot stability (pelvis height, feet on ground).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access objects
    object1 = env.scene['Object1'] # Small Block
    object2 = env.scene['Object2'] # Medium Block
    object3 = env.scene['Object3'] # Large Block

    # Access robot parts
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos_z = robot.data.body_pos_w[:, left_foot_idx, 2]
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos_z = robot.data.body_pos_w[:, right_foot_idx, 2]

    # Collision avoidance: Robot body parts vs. Blocks
    # Penalize if pelvis is too close to any block, outside of a reasonable pushing distance.
    # Blocks are 1m x 1m base. Half diagonal is sqrt(1^2+1^2)/2 = 0.707m.
    # A safe distance for pelvis not to be "colliding" could be 0.5m from block center.
    collision_threshold = 0.5 # meters from block center to pelvis center

    # Distance from pelvis to each block's center
    dist_pelvis_obj1 = torch.norm(pelvis_pos - object1.data.root_pos_w, dim=-1)
    dist_pelvis_obj2 = torch.norm(pelvis_pos - object2.data.root_pos_w, dim=-1)
    dist_pelvis_obj3 = torch.norm(pelvis_pos - object3.data.root_pos_w, dim=-1)

    # Penalize if distance is less than threshold, using a continuous penalty
    # Reward is 0 if distance > threshold, becomes negative as distance decreases below threshold
    # Initialize as per-environment tensor to avoid broadcasting issues
    collision_pelvis_reward = torch.zeros(env.num_envs, device=env.device)
    collision_pelvis_reward -= torch.where(dist_pelvis_obj1 < collision_threshold, (collision_threshold - dist_pelvis_obj1) / collision_threshold, torch.tensor(0.0, device=env.device)) * 0.5
    collision_pelvis_reward -= torch.where(dist_pelvis_obj2 < collision_threshold, (collision_threshold - dist_pelvis_obj2) / collision_threshold, torch.tensor(0.0, device=env.device)) * 0.5
    collision_pelvis_reward -= torch.where(dist_pelvis_obj3 < collision_threshold, (collision_threshold - dist_pelvis_obj3) / collision_threshold, torch.tensor(0.0, device=env.device)) * 0.5

    # Stability: Pelvis height
    # Encourage pelvis to be at a stable standing height (e.g., 0.7m for a human-like robot)
    pelvis_target_z = 0.7
    reward_pelvis_height = -torch.abs(pelvis_pos_z - pelvis_target_z)

    # Stability: Feet on ground
    # Reward for feet being near ground level (z=0). A small offset (0.05m) for foot thickness.
    # Penalize if feet are too high off the ground, indicating instability or falling.
    foot_ground_target_z = 0.05 # Approximate height of foot bottom from ground
    max_foot_lift_for_penalty = 0.2 # If foot is lifted more than 20cm, penalize heavily

    reward_feet_on_ground = -(torch.abs(left_foot_pos_z - foot_ground_target_z) + torch.abs(right_foot_pos_z - foot_ground_target_z))
    # Apply a stronger penalty if feet are significantly off the ground (e.g., robot is falling)
    reward_feet_on_ground = torch.where(left_foot_pos_z > max_foot_lift_for_penalty, torch.tensor(-1.0, device=env.device), reward_feet_on_ground)
    reward_feet_on_ground = torch.where(right_foot_pos_z > max_foot_lift_for_penalty, torch.tensor(-1.0, device=env.device), reward_feet_on_ground)

    reward = collision_pelvis_reward + reward_pelvis_height + reward_feet_on_ground

    # Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

@configclass
class TaskRewardsCfg:
    # Main reward for building the stairs
    main_build_stairs_reward = RewTerm(func=main_build_stairs_reward, weight=1.0,
                                       params={"normalise": True, "normaliser_name": "main_build_stairs_reward"})

    # Shaping reward for approaching blocks and hand proximity
    shaping_approach_and_hand_proximity_reward = RewTerm(func=shaping_approach_and_hand_proximity_reward, weight=0.4,
                                                         params={"normalise": True, "normaliser_name": "shaping_approach_hands"})

    # Shaping reward for robot stability and collision avoidance
    shaping_stability_and_collision_avoidance_reward = RewTerm(func=shaping_stability_and_collision_avoidance_reward, weight=0.2,
                                                                params={"normalise": True, "normaliser_name": "shaping_stability_collision"})