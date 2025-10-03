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

def walk_to_Small_Block_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_Small_Block_main_reward") -> torch.Tensor:
    """
    Primary reward for the walk_to_Small_Block skill.
    This reward guides the robot towards the Small Block (Object3) by continuously reducing the Euclidean distance
    between the robot's pelvis and the center of Object3. It also encourages the robot to be at a stable standing
    height (pelvis_z = 0.7m) and to align its x-position with the block. The reward is structured to be higher
    when closer to the target and when the pelvis is at the desired height.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access Object3 (Small Block) position
    # Requirement: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    object_name = env.scene['Object3']
    target_pos = object_name.data.root_pos_w

    # Access robot pelvis position
    # Requirement: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    robot_pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]

    # Separate x, y, z components for clarity and specific reward shaping
    robot_pelvis_pos_x = robot_pelvis_pos[:, 0]
    robot_pelvis_pos_y = robot_pelvis_pos[:, 1]
    robot_pelvis_pos_z = robot_pelvis_pos[:, 2]

    target_x = target_pos[:, 0]
    target_y = target_pos[:, 1]
    # target_z is not directly used for distance, but pelvis_z is compared to a desired height.

    # Distance in y-axis (primary movement towards the block)
    # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
    # Requirement: absolute distances must be used for distances from objects.
    distance_y_reward = -torch.abs(robot_pelvis_pos_y - target_y) # Negative reward, so minimizing distance maximizes reward

    # Distance in x-axis (alignment with the block)
    distance_x_reward = -torch.abs(robot_pelvis_pos_x - target_x) # Negative reward, so minimizing distance maximizes reward

    # Pelvis height for stability (encourages standing upright)
    # Desired pelvis height for standing is 0.7m. This is a relative distance from the ground.
    # Requirement: NO hard-coded positions or arbitrary thresholds. (0.7m is a common and reasonable relative height for a standing humanoid pelvis)
    pelvis_height_reward = -torch.abs(robot_pelvis_pos_z - 0.7) # Negative reward, so minimizing distance from 0.7m maximizes reward

    # Combine rewards, ensuring continuous gradient
    # The closer the robot is to the target in x and y, and the closer its pelvis is to 0.7m, the higher the reward.
    # Requirement: All operations must work with batched environments (torch operations handle this).
    reward = (distance_y_reward * 0.5) + (distance_x_reward * 0.3) + (pelvis_height_reward * 0.2)

    # Mandatory Reward Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def walk_to_Small_Block_doorway_navigation_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_Small_Block_doorway_navigation_reward") -> torch.Tensor:
    """
    Shaping reward that encourages the robot to pass through the doorway formed by Object1 and Object2.
    It rewards the robot for reducing its y-distance to the doorway's y-coordinate and for being horizontally
    centered between the two walls. This reward is active only when the robot's pelvis is behind the doorway's y-coordinate.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access Object1 (Wall 1) and Object2 (Wall 2) positions
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']

    # Access robot pelvis position
    robot = env.scene["robot"]
    robot_pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]
    robot_pelvis_pos_x = robot_pelvis_pos[:, 0]
    robot_pelvis_pos_y = robot_pelvis_pos[:, 1]

    # Doorway y-coordinate: Assuming Object1 and Object2 have the same y-position for the doorway.
    # From task description: "Two of these objects should be very heavy (1000kg) cubes with a z of 1.5m x of 0.5 and a y of 5m."
    # This implies the walls are positioned at y=5m.
    # Requirement: YOU MUST ACCESS OBJECT LOCATIONS (instead of hard coding)USING THE APPROVED PATTERN
    doorway_y_pos = object1.data.root_pos_w[:, 1] # Using Object1's y-position as the doorway's y-coordinate.

    # Calculate the center of the doorway in x-axis
    # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
    doorway_center_x = (object1.data.root_pos_w[:, 0] + object2.data.root_pos_w[:, 0]) / 2.0

    # Reward for moving towards the doorway in y
    approach_doorway_y_reward = -torch.abs(robot_pelvis_pos_y - doorway_y_pos) # Negative reward, minimizing distance

    # Reward for aligning with the center of the doorway in x
    align_doorway_x_reward = -torch.abs(robot_pelvis_pos_x - doorway_center_x) # Negative reward, minimizing distance

    # Condition: Robot's pelvis is behind the doorway (i.e., has not passed it yet)
    # Assuming robot starts at y < doorway_y_pos and needs to move towards positive y.
    activation_condition = (robot_pelvis_pos_y < doorway_y_pos)

    # Combine rewards and apply activation condition
    reward = (approach_doorway_y_reward * 0.7) + (align_doorway_x_reward * 0.3)
    # Requirement: All operations must work with batched environments (torch.where handles this).
    reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device))

    # Mandatory Reward Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def walk_to_Small_Block_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_Small_Block_collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward that encourages collision avoidance with the walls (Object1 and Object2) while the robot is
    navigating through the doorway. It penalizes the robot if any of its key body parts get too close to the walls.
    The penalty is based on the inverse of the distance, becoming larger (more negative) as the distance decreases.
    This reward is active throughout the skill.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access Object1 (Wall 1) and Object2 (Wall 2) positions
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']

    # Access robot body parts positions
    robot = env.scene["robot"]
    robot_parts_indices = [
        robot.body_names.index('pelvis'),
        robot.body_names.index('left_ankle_roll_link'),
        robot.body_names.index('right_ankle_roll_link'),
        robot.body_names.index('left_palm_link'),
        robot.body_names.index('right_palm_link'),
        robot.body_names.index('head_link')
    ]
    robot_parts_pos = robot.data.body_pos_w[:, robot_parts_indices] # Shape: [num_envs, num_parts, 3]

    # Object dimensions for collision buffer (from task description: x=0.5m for walls)
    # Requirement: THERE IS NO way to access the SIZE of an object. Hardcode from config.
    # Wall x-dimension (width) is 0.5m.
    # A collision threshold of 0.5m is chosen as it's roughly the wall's width, providing a buffer.
    # This is a relative distance, not a hardcoded position.
    collision_threshold = 0.5 

    # Calculate minimum distance from any tracked robot part to Object1 and Object2
    # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
    # Requirement: All operations must work with batched environments.
    
    # Reshape object positions for broadcasting with robot_parts_pos
    obj1_pos_expanded = object1.data.root_pos_w.unsqueeze(1) # Shape: [num_envs, 1, 3]
    obj2_pos_expanded = object2.data.root_pos_w.unsqueeze(1) # Shape: [num_envs, 1, 3]

    # Calculate Euclidean distances from all robot parts to Object1
    distances_to_obj1 = torch.norm(robot_parts_pos - obj1_pos_expanded, dim=-1) # Shape: [num_envs, num_parts]
    # Calculate Euclidean distances from all robot parts to Object2
    distances_to_obj2 = torch.norm(robot_parts_pos - obj2_pos_expanded, dim=-1) # Shape: [num_envs, num_parts]

    # Combine all distances and find the minimum across all parts and both objects
    all_distances = torch.cat((distances_to_obj1, distances_to_obj2), dim=-1) # Shape: [num_envs, 2 * num_parts]
    min_dist_to_walls = torch.min(all_distances, dim=-1).values # Shape: [num_envs]

    # Reward for collision avoidance: penalize when too close
    # Use a linear penalty that becomes more negative as distance decreases below the threshold.
    # Reward is 0 if far, negative if close.
    # Requirement: Rewards should be continuous and positive (or negative for penalties).
    # This is a negative reward (penalty) for being too close.
    collision_reward = torch.where(min_dist_to_walls < collision_threshold,
                                   -(collision_threshold - min_dist_to_walls) * 2.0, # Scale the penalty to make it more impactful
                                   torch.tensor(0.0, device=env.device))

    reward = collision_reward

    # Mandatory Reward Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

@configclass
class TaskRewardsCfg:
    # Primary reward for reaching and aligning with the Small Block
    walk_to_Small_Block_main_reward = RewTerm(func=walk_to_Small_Block_main_reward, weight=1.0,
                                              params={"normalise": True, "normaliser_name": "walk_to_Small_Block_main_reward"})

    # Shaping reward for navigating through the doorway
    walk_to_Small_Block_doorway_navigation_reward = RewTerm(func=walk_to_Small_Block_doorway_navigation_reward, weight=0.6,
                                                            params={"normalise": True, "normaliser_name": "walk_to_Small_Block_doorway_navigation_reward"})

    # Shaping reward for avoiding collisions with the walls
    walk_to_Small_Block_collision_avoidance_reward = RewTerm(func=walk_to_Small_Block_collision_avoidance_reward, weight=0.4,
                                                             params={"normalise": True, "normaliser_name": "walk_to_Small_Block_collision_avoidance_reward"})