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


def main_doorway_and_goal_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Primary reward for the 'doorway_and_goal_seed456' skill.
    Guides the robot through the doorway and then to the final goal block (Object3).
    It's a continuous reward that first encourages the robot to reach the y-position of the doorway,
    and once past it, encourages movement towards Object3. It also includes a small reward for maintaining a stable pelvis height.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects using approved patterns
    object1 = env.scene['Object1'] # Heavy Cube (Wall 1)
    object2 = env.scene['Object2'] # Heavy Cube (Wall 2)
    object3 = env.scene['Object3'] # small block

    # Access required robot part(s) using approved patterns
    robot = env.scene["robot"] # Access robot object
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Access object positions using approved patterns
    object1_pos_y = object1.data.root_pos_w[:, 1]
    object2_pos_y = object2.data.root_pos_w[:, 1]
    object3_pos_x = object3.data.root_pos_w[:, 0]
    object3_pos_y = object3.data.root_pos_w[:, 1]
    object3_pos_z = object3.data.root_pos_w[:, 2]

    # Calculate doorway y-position (average of wall y-positions)
    # This assumes the doorway is centered along the y-axis between the two walls.
    # The problem description is ambiguous about the exact y-coordinates of the walls.
    # Assuming the walls are placed such that their root_pos_w[:, 1] defines the y-center of the doorway.
    doorway_y_pos = (object1_pos_y + object2_pos_y) / 2.0

    # Phase 1: Approach Doorway
    # Reward for reducing y-distance to doorway. This is a negative absolute distance, so closer is higher reward.
    # This uses relative distance between pelvis and doorway y-center.
    reward_approach_doorway = -torch.abs(pelvis_pos_y - doorway_y_pos)

    # Phase 2: Approach Goal Block (after passing doorway)
    # Condition: pelvis_pos_y is past the doorway's y-coordinate.
    # A small buffer is added to ensure the robot is fully through the doorway.
    # Object dimensions: x=0.5m, y=5m, z=1.5m. The doorway is formed by their x-positions.
    # The problem states "2m past the doorway in the y axis" for Object3.
    # This implies the robot moves along the y-axis.
    # Let's assume the doorway's effective y-extent is defined by the walls' y-dimension (5m).
    # The robot needs to pass the y-center of the doorway.
    # The problem states "y centre of both objects should be 5.5m apart" which is confusing if they are walls for an x-doorway.
    # Given "walk through the doorway and then walk to the small block", the primary movement is along y.
    # Let's interpret the doorway as a region in y.
    # If the objects are at y=5m, and the robot walks "past" them, it means pelvis_pos_y > 5m.
    # Let's use the average y-position of the walls as the doorway's y-coordinate.
    # A small clearance is added to ensure the robot is truly past the doorway.
    doorway_clearance_y = 0.5 # A small buffer to ensure the robot is past the doorway's y-plane.
    past_doorway_condition = pelvis_pos_y > (doorway_y_pos + doorway_clearance_y)

    # Reward for reducing y-distance to Object3. This is a negative absolute distance.
    # This uses relative distance between pelvis and Object3's y-position.
    reward_approach_object3_y = -torch.abs(pelvis_pos_y - object3_pos_y)

    # Combine rewards based on phase. This creates a continuous reward that switches objectives.
    primary_reward_y = torch.where(past_doorway_condition, reward_approach_object3_y, reward_approach_doorway)

    # Reward for reducing x-distance to Object3 (once past doorway).
    # This uses relative distance between pelvis and Object3's x-position.
    reward_x_to_object3 = -torch.abs(pelvis_pos_x - object3_pos_x)
    primary_reward_x = torch.where(past_doorway_condition, reward_x_to_object3, torch.tensor(0.0, device=env.device)) # Only reward x-alignment to object3 after passing doorway

    # Reward for reducing z-distance to Object3 (once past doorway).
    # This uses relative distance between pelvis and Object3's z-position.
    reward_z_to_object3 = -torch.abs(pelvis_pos_z - object3_pos_z)
    primary_reward_z = torch.where(past_doorway_condition, reward_z_to_object3, torch.tensor(0.0, device=env.device)) # Only reward z-alignment to object3 after passing doorway

    # Reward for maintaining stable pelvis height (around 0.7m).
    # This is a general stability reward, always active. 0.7m is a standard target pelvis height for bipedal robots.
    # This uses the absolute z-position of the pelvis, which is allowed for height.
    target_pelvis_z = 0.7
    reward_pelvis_height = -torch.abs(pelvis_pos_z - target_pelvis_z)

    # Combine all primary components. Pelvis height is less weighted as it's a general stability factor.
    reward = primary_reward_y + primary_reward_x + primary_reward_z + (0.1 * reward_pelvis_height)

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def doorway_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_reward") -> torch.Tensor:
    """
    Collision avoidance reward with the doorway walls (Object1 and Object2).
    Heavily penalizes the robot if any part of its body (pelvis, feet) gets too close or collides with the walls,
    especially in the x and y dimensions relevant to passing through.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects using approved patterns
    object1 = env.scene['Object1'] # Heavy Cube (Wall 1)
    object2 = env.scene['Object2'] # Heavy Cube (Wall 2)

    # Access required robot part(s) using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    left_foot_pos_x = left_foot_pos[:, 0]
    left_foot_pos_y = left_foot_pos[:, 1]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    right_foot_pos_x = right_foot_pos[:, 0]
    right_foot_pos_y = right_foot_pos[:, 1]

    # Object dimensions (hardcoded from problem description: x=0.5m, y=5m, z=1.5m)
    # These are the dimensions of the cubes themselves.
    wall_x_dim = 0.5 # width of the wall (along x-axis)
    wall_y_dim = 5.0 # length of the wall (along y-axis)
    # wall_z_dim = 1.5 # height of the wall (along z-axis)

    # Calculate the inner x-edges of the doorway walls.
    # Assuming Object1 is on the left (smaller x-coordinate) and Object2 is on the right (larger x-coordinate).
    # The doorway gap is 0.5m.
    # Object1's center x is object1.data.root_pos_w[:, 0]. Its right edge is center_x + wall_x_dim / 2.
    # Object2's center x is object2.data.root_pos_w[:, 0]. Its left edge is center_x - wall_x_dim / 2.
    object1_inner_x_edge = object1.data.root_pos_w[:, 0] + (wall_x_dim / 2.0)
    object2_inner_x_edge = object2.data.root_pos_w[:, 0] - (wall_x_dim / 2.0)

    # Calculate the y-extent of the doorway walls.
    # Assuming Object1 and Object2 have similar y-positions and dimensions.
    # The y-extent is from center_y - wall_y_dim / 2 to center_y + wall_y_dim / 2.
    # Using Object1's y-position as reference for the doorway's y-extent.
    doorway_y_min = object1.data.root_pos_w[:, 1] - (wall_y_dim / 2.0)
    doorway_y_max = object1.data.root_pos_w[:, 1] + (wall_y_dim / 2.0)

    # Define a small buffer for collision detection.
    collision_buffer = 0.1 # meters

    # Function to calculate collision penalty for a given robot part position
    def calculate_part_collision(part_pos_x, part_pos_y):
        # Collision with Object1 (left wall)
        # Penalize if part is too far left (x) and within the y-extent of the wall.
        # Relative distance to inner right edge of Object1.
        dist_x_obj1_inner = part_pos_x - object1_inner_x_edge
        # Relative distance to outer left edge of Object1 (if robot tries to go around).
        dist_x_obj1_outer = object1.data.root_pos_w[:, 0] - (wall_x_dim / 2.0) - part_pos_x

        # Penalize if part is too close to the inner x-edge of Object1 (negative distance means overlap/past edge)
        collision_x_obj1 = torch.where(dist_x_obj1_inner < collision_buffer, dist_x_obj1_inner - collision_buffer, torch.tensor(0.0, device=env.device))
        # Penalize if part is too close to the outer x-edge of Object1 (e.g., trying to go around)
        collision_x_obj1_outer = torch.where(dist_x_obj1_outer < collision_buffer, dist_x_obj1_outer - collision_buffer, torch.tensor(0.0, device=env.device))

        # Collision with Object2 (right wall)
        # Penalize if part is too far right (x) and within the y-extent of the wall.
        # Relative distance to inner left edge of Object2.
        dist_x_obj2_inner = object2_inner_x_edge - part_pos_x
        # Relative distance to outer right edge of Object2.
        dist_x_obj2_outer = part_pos_x - (object2.data.root_pos_w[:, 0] + (wall_x_dim / 2.0))

        # Penalize if part is too close to the inner x-edge of Object2
        collision_x_obj2 = torch.where(dist_x_obj2_inner < collision_buffer, dist_x_obj2_inner - collision_buffer, torch.tensor(0.0, device=env.device))
        # Penalize if part is too close to the outer x-edge of Object2
        collision_x_obj2_outer = torch.where(dist_x_obj2_outer < collision_buffer, dist_x_obj2_outer - collision_buffer, torch.tensor(0.0, device=env.device))

        # Penalize if part is outside the y-extent of the walls (i.e., trying to go around the walls in y)
        dist_y_lower = part_pos_y - doorway_y_min
        dist_y_upper = doorway_y_max - part_pos_y
        collision_y = torch.where((dist_y_lower < collision_buffer) | (dist_y_upper < collision_buffer),
                                  torch.min(dist_y_lower, dist_y_upper) - collision_buffer,
                                  torch.tensor(0.0, device=env.device))

        # Sum penalties for this part. Negative values indicate collision/proximity.
        return collision_x_obj1 + collision_x_obj1_outer + collision_x_obj2 + collision_x_obj2_outer + collision_y

    # Calculate collision penalties for relevant robot parts
    pelvis_collision_penalty = calculate_part_collision(pelvis_pos_x, pelvis_pos_y)
    left_foot_collision_penalty = calculate_part_collision(left_foot_pos_x, left_foot_pos_y)
    right_foot_collision_penalty = calculate_part_collision(right_foot_pos_x, right_foot_pos_y)

    # Combine all collision penalties. Max ensures the most severe penalty dominates.
    # We want to maximize this reward, so we take the maximum of the negative penalties (closest to zero).
    # A negative value means collision, so we want to push it towards zero.
    reward = torch.max(pelvis_collision_penalty, torch.max(left_foot_collision_penalty, right_foot_collision_penalty))

    # Activation: This reward applies when the robot's pelvis is within the y-extent of the doorway walls,
    # plus a small buffer, to ensure it's active before and during passing through.
    activation_buffer_y = 1.0 # Extend activation zone slightly beyond wall length
    is_near_doorway_y = (pelvis_pos_y > doorway_y_min - activation_buffer_y) & \
                         (pelvis_pos_y < doorway_y_max + activation_buffer_y)

    # Apply reward only when near the doorway
    reward = torch.where(is_near_doorway_y, reward, torch.tensor(0.0, device=env.device))

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def doorway_centering_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "centering_reward") -> torch.Tensor:
    """
    Doorway centering reward. Encourages the robot to stay centered in the x-axis while passing through the doorway.
    This helps prevent it from hitting the sides of the doorway.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects using approved patterns
    object1 = env.scene['Object1'] # Heavy Cube (Wall 1)
    object2 = env.scene['Object2'] # Heavy Cube (Wall 2)

    # Access required robot part(s) using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]

    # Object dimensions (hardcoded from problem description: x=0.5m, y=5m, z=1.5m)
    wall_x_dim = 0.5 # width of the wall

    # Calculate the center x-position of the doorway gap.
    # The inner right edge of Object1 is object1.data.root_pos_w[:, 0] + (wall_x_dim / 2.0)
    # The inner left edge of Object2 is object2.data.root_pos_w[:, 0] - (wall_x_dim / 2.0)
    # The doorway center is the average of these two inner edges.
    doorway_center_x = (object1.data.root_pos_w[:, 0] + (wall_x_dim / 2.0) +
                        object2.data.root_pos_w[:, 0] - (wall_x_dim / 2.0)) / 2.0

    # Reward for reducing x-distance to the doorway center. Negative absolute distance.
    # This uses relative distance between pelvis and doorway x-center.
    reward_centering_x = -torch.abs(pelvis_pos_x - doorway_center_x)

    # Condition: Robot's pelvis is within the y-extent of the doorway walls.
    # This ensures the centering reward is only active when the robot is actually traversing the doorway.
    wall_y_dim = 5.0 # length of the wall
    doorway_y_min = object1.data.root_pos_w[:, 1] - (wall_y_dim / 2.0)
    doorway_y_max = object1.data.root_pos_w[:, 1] + (wall_y_dim / 2.0)
    is_in_doorway_y = (pelvis_pos_y > doorway_y_min) & (pelvis_pos_y < doorway_y_max)

    # Apply reward only when in the doorway's y-extent.
    reward = torch.where(is_in_doorway_y, reward_centering_x, torch.tensor(0.0, device=env.device))

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Primary reward for guiding the robot through the doorway and to the goal block.
    # Weight is 1.0 as it's the main objective.
    MainDoorwayAndGoalReward = RewTerm(func=main_doorway_and_goal_reward, weight=1.0,
                                       params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for collision avoidance with the doorway walls.
    # Weight is 0.6 to heavily penalize collisions but not completely overshadow the main goal.
    DoorwayCollisionAvoidanceReward = RewTerm(func=doorway_collision_avoidance_reward, weight=0.6,
                                              params={"normalise": True, "normaliser_name": "collision_reward"})

    # Shaping reward for centering the robot within the doorway.
    # Weight is 0.4 to encourage good posture through the doorway, less critical than collision avoidance.
    DoorwayCenteringReward = RewTerm(func=doorway_centering_reward, weight=0.4,
                                     params={"normalise": True, "normaliser_name": "centering_reward"})