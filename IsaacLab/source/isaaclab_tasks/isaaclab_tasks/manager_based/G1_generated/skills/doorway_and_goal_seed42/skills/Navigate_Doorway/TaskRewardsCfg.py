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


def main_navigate_doorway_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_navigate_doorway_reward") -> torch.Tensor:
    """
    Main reward for the Navigate_Doorway skill.
    Encourages the robot to pass through the doorway and position its pelvis just past the doorway's x-extent,
    aligned with the y-position of Object3 (Small Block).
    """
    # Get normalizer instance (mandatory)
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts (mandatory: approved patterns)
    robot = env.scene["robot"]
    object1 = env.scene['Object1']  # Heavy Cube (Wall 1)
    object2 = env.scene['Object2']  # Heavy Cube (Wall 2)
    object3 = env.scene['Object3']  # Small Block

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Hardcoded object dimensions from task description (mandatory: no dynamic access)
    # Wall dimensions: x=0.5m, y=5m, z=1.5m
    wall_x_dim = 0.5  # x-dimension of the walls
    wall_y_dim = 5.0  # y-dimension of the walls

    # Calculate doorway x-extent based on Object1's position (assuming Object1 defines the doorway's x-position)
    # The walls are 0.5m in x. Assuming root_pos_w is the center of the wall.
    # The robot needs to pass through the x-depth of the walls.
    # doorway_x_front is the x-coordinate of the "front" face of the walls.
    # doorway_x_back is the x-coordinate of the "back" face of the walls.
    doorway_x_front = object1.data.root_pos_w[:, 0] - wall_x_dim / 2
    doorway_x_back = object1.data.root_pos_w[:, 0] + wall_x_dim / 2

    # Calculate doorway y-center (gap between walls)
    # Object1 (Wall 1) is the left wall, Object2 (Wall 2) is the right wall.
    # The gap is between Object1's right edge and Object2's left edge.
    # Object1's y-center is object1.data.root_pos_w[:, 1]
    # Object2's y-center is object2.data.root_pos_w[:, 1]
    # The doorway gap is 0.5m, and wall y-dimension is 5m.
    # The y-center of the doorway is the midpoint between the inner edges of the walls.
    # This calculation is not directly used for the target, but for understanding the setup.
    # doorway_y_inner_left = object1.data.root_pos_w[:, 1] + wall_y_dim / 2
    # doorway_y_inner_right = object2.data.root_pos_w[:, 1] - wall_y_dim / 2
    # doorway_y_center = (doorway_y_inner_left + doorway_y_inner_right) / 2

    # Target position for the pelvis after clearing the doorway
    # The robot should be slightly past the doorway's x-extent.
    # The robot should align with Object3's y-position.
    # The robot should maintain a stable pelvis height.
    target_x_pos = doorway_x_back + 0.5  # 0.5m past the back of the doorway
    target_y_pos = object3.data.root_pos_w[:, 1]  # Align with Object3's y-position
    pelvis_target_z = 0.7  # Target stable pelvis height

    # Calculate relative distances to the target point (mandatory: relative distances)
    # Reward is negative absolute distance, so smaller distance yields higher reward.
    dist_x = torch.abs(pelvis_pos_x - target_x_pos)
    dist_y = torch.abs(pelvis_pos_y - target_y_pos)
    dist_z = torch.abs(pelvis_pos_z - pelvis_target_z)

    # Reward components (continuous and positive, closer is better)
    reward_x = -dist_x
    reward_y = -dist_y
    reward_z = -dist_z

    # Combine rewards
    # The primary reward is a sum of these components, encouraging the robot to reach the target point.
    reward = reward_x + reward_y + reward_z

    # Mandatory: Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_walls_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_walls_reward") -> torch.Tensor:
    """
    Shaping reward that penalizes the robot for getting too close to or colliding with Object1 or Object2 (the walls).
    """
    # Get normalizer instance (mandatory)
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts (mandatory: approved patterns)
    robot = env.scene["robot"]
    object1 = env.scene['Object1']  # Heavy Cube (Wall 1)
    object2 = env.scene['Object2']  # Heavy Cube (Wall 2)

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Hardcoded object dimensions from task description (mandatory: no dynamic access)
    wall_x_dim = 0.5  # x-dimension of the walls
    wall_y_dim = 5.0  # y-dimension of the walls
    wall_z_dim = 1.5  # z-dimension of the walls

    # Define a safety margin for collision detection
    # This margin defines how close the robot can get before a penalty starts.
    safety_margin = 0.15  # meters

    robot_parts_to_check = [pelvis_pos, left_foot_pos, right_foot_pos]
    collision_penalty = torch.zeros_like(pelvis_pos[:, 0])

    # Iterate over each robot part to check proximity to both walls
    for part_pos in robot_parts_to_check:
        part_x, part_y, part_z = part_pos[:, 0], part_pos[:, 1], part_pos[:, 2]

        # Calculate bounding box min/max for Object1 (Wall 1)
        # Object positions are accessed using approved patterns.
        obj1_x_min = object1.data.root_pos_w[:, 0] - wall_x_dim / 2
        obj1_x_max = object1.data.root_pos_w[:, 0] + wall_x_dim / 2
        obj1_y_min = object1.data.root_pos_w[:, 1] - wall_y_dim / 2
        obj1_y_max = object1.data.root_pos_w[:, 1] + wall_y_dim / 2
        obj1_z_min = object1.data.root_pos_w[:, 2] - wall_z_dim / 2
        obj1_z_max = object1.data.root_pos_w[:, 2] + wall_z_dim / 2

        # Calculate bounding box min/max for Object2 (Wall 2)
        obj2_x_min = object2.data.root_pos_w[:, 0] - wall_x_dim / 2
        obj2_x_max = object2.data.root_pos_w[:, 0] + wall_x_dim / 2
        obj2_y_min = object2.data.root_pos_w[:, 1] - wall_y_dim / 2
        obj2_y_max = object2.data.root_pos_w[:, 1] + wall_y_dim / 2
        obj2_z_min = object2.data.root_pos_w[:, 2] - wall_z_dim / 2
        obj2_z_max = object2.data.root_pos_w[:, 2] + wall_z_dim / 2

        # Calculate closest distance from robot part to Object1's bounding box
        # This calculates the distance to the closest point on the box surface.
        # All distances are relative.
        dist_x_obj1 = torch.max(torch.zeros_like(part_x), torch.max(obj1_x_min - part_x, part_x - obj1_x_max))
        dist_y_obj1 = torch.max(torch.zeros_like(part_y), torch.max(obj1_y_min - part_y, part_y - obj1_y_max))
        dist_z_obj1 = torch.max(torch.zeros_like(part_z), torch.max(obj1_z_min - part_z, part_z - obj1_z_max))
        dist_to_obj1 = torch.sqrt(dist_x_obj1**2 + dist_y_obj1**2 + dist_z_obj1**2)

        # Calculate closest distance from robot part to Object2's bounding box
        dist_x_obj2 = torch.max(torch.zeros_like(part_x), torch.max(obj2_x_min - part_x, part_x - obj2_x_max))
        dist_y_obj2 = torch.max(torch.zeros_like(part_y), torch.max(obj2_y_min - part_y, part_y - obj2_y_max))
        dist_z_obj2 = torch.max(torch.zeros_like(part_z), torch.max(obj2_z_min - part_z, part_z - obj2_z_max))
        dist_to_obj2 = torch.sqrt(dist_x_obj2**2 + dist_y_obj2**2 + dist_z_obj2**2)

        # Take the minimum distance to either wall for the current robot part
        min_dist_to_wall_for_part = torch.min(dist_to_obj1, dist_to_obj2)

        # Apply a continuous penalty if the distance is less than the safety margin
        # The penalty increases (becomes more negative) as the robot part gets closer to the wall.
        penalty_for_part = torch.where(
            min_dist_to_wall_for_part < safety_margin,
            -(safety_margin - min_dist_to_wall_for_part),  # Negative reward, increases as distance decreases
            torch.zeros_like(min_dist_to_wall_for_part)
        )
        collision_penalty += penalty_for_part

    # The reward is the accumulated penalty (negative value)
    reward = collision_penalty

    # Mandatory: Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def upright_posture_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "upright_posture_stability_reward") -> torch.Tensor:
    """
    Shaping reward that encourages the robot to maintain an upright and stable posture
    by penalizing deviations of the pelvis's z-position from a target height.
    """
    # Get normalizer instance (mandatory)
    RewNormalizer = get_normalizer(env.device)

    # Access required robot part (mandatory: approved patterns)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Target pelvis height for stable standing (hardcoded, based on typical robot height)
    pelvis_target_z = 0.7  # meters

    # Reward for maintaining pelvis height close to target (mandatory: relative distance, continuous)
    # Penalize deviation from target height. Negative absolute difference.
    # This is a relative distance to a target height.
    reward = -torch.abs(pelvis_pos_z - pelvis_target_z)

    # Mandatory: Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Main reward for navigating through the doorway and reaching the target position
    MainNavigateDoorwayReward = RewTerm(func=main_navigate_doorway_reward, weight=1.0,
                                        params={"normalise": True, "normaliser_name": "main_navigate_doorway_reward"})

    # Shaping reward for avoiding collisions with the walls
    CollisionAvoidanceWallsReward = RewTerm(func=collision_avoidance_walls_reward, weight=0.6,
                                            params={"normalise": True, "normaliser_name": "collision_avoidance_walls_reward"})

    # Shaping reward for maintaining an upright and stable posture
    UprightPostureStabilityReward = RewTerm(func=upright_posture_stability_reward, weight=0.3,
                                            params={"normalise": True, "normaliser_name": "upright_posture_stability_reward"})