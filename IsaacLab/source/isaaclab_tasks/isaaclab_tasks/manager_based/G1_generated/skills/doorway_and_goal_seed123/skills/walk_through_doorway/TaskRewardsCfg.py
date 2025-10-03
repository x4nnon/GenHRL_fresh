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


def walk_through_doorway_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the walk_through_doorway skill.
    Rewards the robot for moving its pelvis past the doorway plane and reaching a target y-position
    slightly past the doorway, preparing for the next skill.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    # Object1: Heavy Cube (Wall 1) forming the left wall of the doorway
    object1 = env.scene['Object1']
    # Object3: Small Block for the robot to walk to (used for context of target y-position)
    object3 = env.scene['Object3']

    # Access the required robot part(s) using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_y = pelvis_pos[:, 1] # Extract y-component of pelvis position

    # Doorway y-plane reference: Use the y-position of Object1 (or Object2, they are aligned in y)
    # The walls are 5m long in y. Assuming their root_pos_w is the center.
    # The "doorway plane" for passing through is the front edge of the walls.
    # Wall y-dimension is 5m, so half-dimension is 2.5m.
    # Front edge of the wall (relative to robot moving forward in positive y): wall_center_y - 2.5m
    # This uses a hardcoded dimension from the object configuration and a relative calculation.
    doorway_front_y_plane = object1.data.root_pos_w[:, 1] - 2.5

    # Target y-position for the pelvis: 1.0m past the doorway's front plane.
    # This ensures the robot is clearly past the doorway but not too far, as Object3 is 2m past.
    # This is a relative target position based on the doorway's front plane.
    target_y_past_doorway = doorway_front_y_plane + 1.0

    # Reward for moving pelvis_pos_y towards target_y_past_doorway.
    # This is a negative absolute distance, so it's a positive reward as distance decreases.
    # This encourages the robot to reach the target y-position. This is a continuous reward.
    reward = -torch.abs(pelvis_pos_y - target_y_past_doorway)

    # Add a small bonus for being past the doorway's front plane.
    # This provides an initial positive signal for making progress through the doorway.
    # This is a continuous bonus (0 or 0.1) based on a relative position.
    reward += 0.1 * (pelvis_pos_y > doorway_front_y_plane).float()

    # Normalization using the mandatory pattern
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def doorway_collision_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_reward") -> torch.Tensor:
    """
    Penalizes the robot for getting too close to Object1 (left wall) and Object2 (right wall)
    with its body parts (pelvis, hands, feet). Also penalizes being outside the doorway's
    x-range when passing through the doorway's y-range.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    # Object1: Heavy Cube (Wall 1) forming the left wall of the doorway
    object1 = env.scene['Object1']
    # Object2: Heavy Cube (Wall 2) forming the right wall of the doorway
    object2 = env.scene['Object2']

    # Access the required robot part(s) using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    left_hand_idx = robot.body_names.index('left_palm_link')
    right_hand_idx = robot.body_names.index('right_palm_link')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')

    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Wall dimensions (from skill info/object config): x=0.5m, y=5m, z=1.5m
    # These are hardcoded from the object configuration, as per requirements.
    wall_x_dim = 0.5
    wall_y_dim = 5.0
    wall_z_dim = 1.5

    # Define a safety margin for collision avoidance. These are relative distances.
    safety_margin_x = 0.1 # meters
    safety_margin_y = 0.1 # meters
    safety_margin_z = 0.1 # meters

    # Collision penalty function: large negative reward if distance is below threshold.
    # This creates a continuous penalty that increases as the distance decreases below the threshold.
    def collision_penalty(distance, threshold):
        # Reward is 0 if distance >= threshold, otherwise it's a negative value proportional to (threshold - distance).
        return torch.where(distance < threshold, -10.0 * (threshold - distance), 0.0)

    collision_reward = 0.0

    # Calculate collision penalties for each relevant robot part with both walls.
    for part_pos in [pelvis_pos, left_hand_pos, right_hand_pos, left_foot_pos, right_foot_pos]:
        # Collision with Object1 (left wall)
        wall1_center_x = object1.data.root_pos_w[:, 0]
        wall1_center_y = object1.data.root_pos_w[:, 1]
        wall1_center_z = object1.data.root_pos_w[:, 2]

        # Calculate relative distances to the wall center for each dimension.
        dist_x_wall1 = torch.abs(part_pos[:, 0] - wall1_center_x)
        dist_y_wall1 = torch.abs(part_pos[:, 1] - wall1_center_y)
        dist_z_wall1 = torch.abs(part_pos[:, 2] - wall1_center_z)

        # Penalize if any dimension is too close to Object1.
        # Thresholds are based on half-dimensions of the wall plus a safety margin.
        collision_reward += collision_penalty(dist_x_wall1, wall_x_dim / 2.0 + safety_margin_x)
        collision_reward += collision_penalty(dist_y_wall1, wall_y_dim / 2.0 + safety_margin_y)
        collision_reward += collision_penalty(dist_z_wall1, wall_z_dim / 2.0 + safety_margin_z)

        # Collision with Object2 (right wall)
        wall2_center_x = object2.data.root_pos_w[:, 0]
        wall2_center_y = object2.data.root_pos_w[:, 1]
        wall2_center_z = object2.data.root_pos_w[:, 2]

        # Calculate relative distances to the wall center for each dimension.
        dist_x_wall2 = torch.abs(part_pos[:, 0] - wall2_center_x)
        dist_y_wall2 = torch.abs(part_pos[:, 1] - wall2_center_y)
        dist_z_wall2 = torch.abs(part_pos[:, 2] - wall2_center_z)

        # Penalize if any dimension is too close to Object2.
        collision_reward += collision_penalty(dist_x_wall2, wall_x_dim / 2.0 + safety_margin_x)
        collision_reward += collision_penalty(dist_y_wall2, wall_y_dim / 2.0 + safety_margin_y)
        collision_reward += collision_penalty(dist_z_wall2, wall_z_dim / 2.0 + safety_margin_z)

    # Add a specific penalty for being outside the doorway x-range when passing through.
    # This encourages the robot to stay within the gap.
    # Object1 is the left wall, Object2 is the right wall.
    # Object1's right edge (inner x-boundary): Object1.x + half_x_dim (0.25m)
    # Object2's left edge (inner x-boundary): Object2.x - half_x_dim (0.25m)
    # These are relative boundaries derived from object positions and hardcoded dimensions.
    object1_inner_x = object1.data.root_pos_w[:, 0] + (wall_x_dim / 2.0)
    object2_inner_x = object2.data.root_pos_w[:, 0] - (wall_x_dim / 2.0)

    # Doorway y-range (front to back of walls)
    # Wall center y is object1.data.root_pos_w[:, 1]. Wall y-dimension is 5m.
    # These are relative boundaries derived from object positions and hardcoded dimensions.
    doorway_front_y = object1.data.root_pos_w[:, 1] - (wall_y_dim / 2.0)
    doorway_back_y = object1.data.root_pos_w[:, 1] + (wall_y_dim / 2.0)

    # Condition: Pelvis is within the y-range of the doorway walls.
    # This ensures the x-centering penalty only applies when the robot is actively passing through.
    pelvis_in_doorway_y_condition = (pelvis_pos[:, 1] > doorway_front_y) & (pelvis_pos[:, 1] < doorway_back_y)

    # Penalize if pelvis x-position is outside the doorway gap.
    # Penalty increases linearly with how far outside the gap the pelvis is.
    # These are relative position checks.
    pelvis_outside_doorway_x_left = pelvis_pos[:, 0] < object1_inner_x
    pelvis_outside_doorway_x_right = pelvis_pos[:, 0] > object2_inner_x

    # Apply penalty only when pelvis is in the doorway's y-range.
    # The penalty is proportional to the distance outside the boundary. This is a continuous reward.
    collision_reward += torch.where(pelvis_in_doorway_y_condition & pelvis_outside_doorway_x_left,
                                    -5.0 * (object1_inner_x - pelvis_pos[:, 0]), 0.0)
    collision_reward += torch.where(pelvis_in_doorway_y_condition & pelvis_outside_doorway_x_right,
                                    -5.0 * (pelvis_pos[:, 0] - object2_inner_x), 0.0)

    # Normalization using the mandatory pattern
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, collision_reward)
        RewNormalizer.update_stats(normaliser_name, collision_reward)
        return scaled_reward
    return collision_reward


def doorway_posture_centering_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "posture_centering_reward") -> torch.Tensor:
    """
    Encourages the robot to maintain an upright posture by keeping its pelvis at a desired height
    and to stay centered in the x-axis of the doorway while passing through.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    # Object1: Heavy Cube (Wall 1)
    object1 = env.scene['Object1']
    # Object2: Heavy Cube (Wall 2)
    object2 = env.scene['Object2']

    # Access the required robot part(s) using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Desired pelvis height for upright posture. This is an absolute height, which is allowed for z-axis.
    desired_pelvis_z = 0.7

    # Reward for maintaining desired pelvis height (upright posture).
    # Negative absolute difference, so reward is higher when closer to desired height. This is a continuous reward.
    reward_pelvis_z = -torch.abs(pelvis_pos_z - desired_pelvis_z)

    # Calculate the center of the doorway in the x-axis.
    # Wall x-dimension is 0.5m (hardcoded from object config).
    wall_x_dim = 0.5
    # Object1's right edge (inner x-boundary): Object1.x + half_x_dim
    object1_inner_x = object1.data.root_pos_w[:, 0] + (wall_x_dim / 2.0)
    # Object2's left edge (inner x-boundary): Object2.x - half_x_dim
    object2_inner_x = object2.data.root_pos_w[:, 0] - (wall_x_dim / 2.0)
    # Center of the doorway is the midpoint between the inner edges of the walls.
    # All calculations are relative to object positions and hardcoded dimensions.
    doorway_center_x = (object1_inner_x + object2_inner_x) / 2.0

    # Reward for staying centered in the doorway's x-axis.
    # This reward is most critical when the robot is passing through the doorway.
    # Wall y-dimension is 5m (hardcoded from object config).
    wall_y_dim = 5.0
    doorway_front_y = object1.data.root_pos_w[:, 1] - (wall_y_dim / 2.0)
    doorway_back_y = object1.data.root_pos_w[:, 1] + (wall_y_dim / 2.0)

    # Condition: Pelvis is within the y-range of the doorway walls.
    # This uses relative positions to define the activation zone.
    pelvis_in_doorway_y_condition = (pelvis_pos[:, 1] > doorway_front_y) & (pelvis_pos[:, 1] < doorway_back_y)

    # Reward for x-centering, active only when the robot is within the doorway's y-range.
    # Negative absolute difference, so reward is higher when closer to the center. This is a continuous reward.
    reward_x_centering_base = -torch.abs(pelvis_pos_x - doorway_center_x)
    reward_x_centering = torch.where(pelvis_in_doorway_y_condition, reward_x_centering_base, 0.0)

    # Combine posture and centering rewards.
    reward = reward_pelvis_z + reward_x_centering

    # Normalization using the mandatory pattern
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Main reward for passing through the doorway and reaching the target y-position.
    # Weight 1.0 as it's the primary objective.
    WalkThroughDoorwayMainReward = RewTerm(func=walk_through_doorway_main_reward, weight=1.0,
                                           params={"normalise": True, "normaliser_name": "walk_through_doorway_main_reward"})

    # Shaping reward for avoiding collisions with the walls.
    # Weight 0.6 to strongly penalize collisions but allow the main reward to dominate.
    DoorwayCollisionReward = RewTerm(func=doorway_collision_reward, weight=0.6,
                                     params={"normalise": True, "normaliser_name": "doorway_collision_reward"})

    # Shaping reward for maintaining upright posture and staying centered in the doorway.
    # Weight 0.3 to encourage good form without being overly restrictive.
    DoorwayPostureCenteringReward = RewTerm(func=doorway_posture_centering_reward, weight=0.3,
                                            params={"normalise": True, "normaliser_name": "doorway_posture_centering_reward"})