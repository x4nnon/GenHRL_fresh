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


def main_push_largeSphere_into_highWall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the push_largeSphere_into_highWall skill.
    This reward encourages the robot to push the large sphere (Object1) towards the high wall (Object4)
    and rewards the high wall falling over. It also provides a bonus for the robot's hands being
    in a good position to push the sphere.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts
    large_sphere = env.scene['Object1'] # Accessing Object1 (large sphere)
    high_wall = env.scene['Object4']   # Accessing Object4 (high wall)
    robot = env.scene["robot"]         # Accessing the robot

    right_hand_idx = robot.body_names.index('right_palm_link') # Accessing right hand index
    left_hand_idx = robot.body_names.index('left_palm_link')   # Accessing left hand index

    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx] # Accessing right hand position
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]   # Accessing left hand position

    # Object dimensions (hardcoded from task description as per rules)
    large_sphere_radius = 1.0
    high_wall_thickness = 0.3
    high_wall_height = 1.0

    # --- Reward Component 1: Moving the large sphere towards the high wall ---
    # Calculate the x-position of the sphere's leading edge and the wall's leading edge.
    # This uses relative distances between object positions and their hardcoded dimensions.
    sphere_leading_edge_x = large_sphere.data.root_pos_w[:, 0] + large_sphere_radius
    wall_leading_edge_x = high_wall.data.root_pos_w[:, 0] - (high_wall_thickness / 2.0)

    # Distance from sphere's leading edge to wall's leading edge.
    # A negative value means the sphere has passed the wall's initial contact point.
    distance_sphere_to_wall_contact_x = wall_leading_edge_x - sphere_leading_edge_x

    # Reward for reducing this distance. Maximize when distance is <= 0 (contact/pushing).
    # Using a negative exponential to provide a continuous reward that increases as the sphere gets closer.
    # The reward is capped to prevent excessive reward if the sphere goes far past the wall.
    # The initial distance is approx 1.85m (3m separation - 1m sphere radius - 0.15m half wall thickness).
    reward_sphere_movement = torch.exp(-torch.relu(distance_sphere_to_wall_contact_x) * 1.0) * 2.0 # Max 2.0 when contact is made or passed

    # --- Reward Component 2: High wall falling over ---
    # Check if the high wall's z-position has dropped significantly.
    # This uses the relative z-position of the wall to a threshold based on its height.
    wall_fallen_threshold_z = high_wall_height * 0.5 # Wall is considered fallen if its center drops below half its height
    
    # Reward is higher as wall_z approaches 0 (or the fallen threshold).
    # Using an exponential decay from the threshold to provide a continuous reward.
    # Max reward when wall_z is at or below the threshold.
    reward_wall_fall = torch.exp(-torch.abs(high_wall.data.root_pos_w[:, 2] - wall_fallen_threshold_z)) * 5.0 # Max 5.0 when at threshold

    # --- Reward Component 3: Robot hands close to and behind the sphere ---
    # Calculate average hand position relative to the sphere's center.
    avg_hand_pos_x = (right_hand_pos[:, 0] + left_hand_pos[:, 0]) / 2.0
    avg_hand_pos_y = (right_hand_pos[:, 1] + left_hand_pos[:, 1]) / 2.0
    avg_hand_pos_z = (right_hand_pos[:, 2] + left_hand_pos[:, 2]) / 2.0

    # Relative distances from average hand position to sphere center.
    dist_hands_to_sphere_x = large_sphere.data.root_pos_w[:, 0] - avg_hand_pos_x
    dist_hands_to_sphere_y = large_sphere.data.root_pos_w[:, 1] - avg_hand_pos_y
    dist_hands_to_sphere_z = large_sphere.data.root_pos_w[:, 2] - avg_hand_pos_z

    # Condition: hands are behind the sphere (x-wise) and within a reasonable range in y and z.
    # This ensures the robot is positioned to push.
    hands_behind_sphere_condition = (avg_hand_pos_x < large_sphere.data.root_pos_w[:, 0] + large_sphere_radius * 0.5) & \
                                    (torch.abs(dist_hands_to_sphere_y) < large_sphere_radius + 0.5) & \
                                    (torch.abs(dist_hands_to_sphere_z) < large_sphere_radius + 0.5)

    # Reward for hands being close to the sphere, only when the condition is met.
    # Using negative absolute distances for continuous reward, scaled.
    reward_hands_on_sphere_raw = -torch.abs(dist_hands_to_sphere_x) - torch.abs(dist_hands_to_sphere_y) - torch.abs(dist_hands_to_sphere_z)
    reward_hands_on_sphere = torch.where(hands_behind_sphere_condition, reward_hands_on_sphere_raw * 0.5, -1.0) # Penalize if not in good pushing position

    # Combine all reward components
    reward = reward_sphere_movement + reward_wall_fall + reward_hands_on_sphere

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def shaping_approach_sphere_and_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_approach_sphere_and_stability_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to approach the large sphere (Object1)
    while maintaining a stable, upright posture.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts
    large_sphere = env.scene['Object1'] # Accessing Object1 (large sphere)
    high_wall = env.scene['Object4']   # Accessing Object4 (high wall)
    robot = env.scene["robot"]         # Accessing the robot

    pelvis_idx = robot.body_names.index('pelvis') # Accessing pelvis index
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing pelvis position

    # Object dimensions (hardcoded from task description as per rules)
    large_sphere_radius = 1.0
    high_wall_thickness = 0.3

    # --- Reward Component 1: Approaching the large sphere (x-axis) ---
    # Calculate the x-distance from pelvis to the sphere's center.
    # This uses relative distance between robot part and object.
    dist_pelvis_to_sphere_x = large_sphere.data.root_pos_w[:, 0] - pelvis_pos[:, 0]

    # Reward for decreasing this distance. Maximize when pelvis is close to sphere's x-position.
    # Using negative absolute distance for continuous reward.
    reward_approach_sphere_x = -torch.abs(dist_pelvis_to_sphere_x) * 0.1

    # --- Reward Component 2: Maintaining pelvis z-height for stability ---
    # Target pelvis z-position for a stable, upright posture (e.g., typical humanoid standing height).
    # This uses the absolute z-position of the pelvis, which is allowed for height.
    target_pelvis_z = 0.7 # meters

    # Reward for staying close to the target z-height.
    # Using negative absolute difference for continuous reward.
    reward_pelvis_z = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z) * 0.2

    # --- Reward Component 3: Staying aligned on y-axis ---
    # Assuming objects are centered at y=0, penalize deviation from y=0.
    # This uses the absolute y-position of the pelvis.
    reward_pelvis_y = -torch.abs(pelvis_pos[:, 1]) * 0.1

    # Combine approach and stability rewards
    shaping_reward = reward_approach_sphere_x + reward_pelvis_z + reward_pelvis_y

    # Activation condition: This reward applies when the robot is still approaching the sphere
    # (pelvis x is less than sphere x plus a small buffer) AND the sphere has not yet reached the high wall.
    # This prevents the reward from interfering with the main pushing task.
    sphere_leading_edge_x = large_sphere.data.root_pos_w[:, 0] + large_sphere_radius
    wall_leading_edge_x = high_wall.data.root_pos_w[:, 0] - (high_wall_thickness / 2.0)
    
    # Condition for sphere not yet at wall (allowing a small overlap for contact)
    sphere_not_at_wall_condition = sphere_leading_edge_x < wall_leading_edge_x + 0.1

    # Condition for robot pelvis still approaching the sphere (before it's fully behind it to push)
    pelvis_approaching_sphere_condition = pelvis_pos[:, 0] < large_sphere.data.root_pos_w[:, 0] + large_sphere_radius

    activation_condition = pelvis_approaching_sphere_condition & sphere_not_at_wall_condition

    # Apply the shaping reward only when the activation condition is met, otherwise 0.
    reward = torch.where(activation_condition, shaping_reward, torch.tensor(0.0, device=env.device))

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def shaping_collision_avoidance_high_wall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_collision_avoidance_high_wall_reward") -> torch.Tensor:
    """
    Shaping reward to penalize the robot for getting too close to or colliding with the high wall (Object4).
    The robot should push the sphere into the wall, not directly interact with the wall itself.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts
    high_wall = env.scene['Object4'] # Accessing Object4 (high wall)
    robot = env.scene["robot"]       # Accessing the robot

    pelvis_idx = robot.body_names.index('pelvis')             # Accessing pelvis index
    right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Accessing right foot index
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')   # Accessing left foot index

    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]         # Accessing pelvis position
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Accessing right foot position
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]   # Accessing left foot position

    # Object dimensions (hardcoded from task description as per rules)
    high_wall_x_size = 0.3
    high_wall_y_size = 5.0
    high_wall_z_size = 1.0

    # Define a safety margin around the wall for penalty activation
    safety_margin = 0.2 # meters

    # Calculate the minimum distance from robot parts (pelvis, feet) to the high wall's surfaces.
    # This involves calculating the distance to the closest point on the wall's bounding box.
    # This uses relative distances between robot parts and object positions, considering object dimensions.

    # Helper function to calculate distance from a point to a box
    def dist_point_to_box(point_pos, box_center, box_x_size, box_y_size, box_z_size):
        half_x = box_x_size / 2.0
        half_y = box_y_size / 2.0
        half_z = box_z_size / 2.0

        # Calculate distances to each face, clamped at 0 if inside
        dx = torch.max(torch.tensor(0.0, device=env.device), torch.abs(point_pos[:, 0] - box_center[:, 0]) - half_x)
        dy = torch.max(torch.tensor(0.0, device=env.device), torch.abs(point_pos[:, 1] - box_center[:, 1]) - half_y)
        dz = torch.max(torch.tensor(0.0, device=env.device), torch.abs(point_pos[:, 2] - box_center[:, 2]) - half_z)

        # Euclidean distance to the closest point on the box surface
        return torch.sqrt(dx**2 + dy**2 + dz**2)

    # Calculate distances for each tracked robot part
    min_dist_pelvis_to_wall = dist_point_to_box(pelvis_pos, high_wall.data.root_pos_w, high_wall_x_size, high_wall_y_size, high_wall_z_size)
    min_dist_right_foot_to_wall = dist_point_to_box(right_foot_pos, high_wall.data.root_pos_w, high_wall_x_size, high_wall_y_size, high_wall_z_size)
    min_dist_left_foot_to_wall = dist_point_to_box(left_foot_pos, high_wall.data.root_pos_w, high_wall_x_size, high_wall_y_size, high_wall_z_size)

    # Take the minimum distance among all tracked parts to represent the closest robot part to the wall
    min_dist_robot_to_wall = torch.min(torch.min(min_dist_pelvis_to_wall, min_dist_right_foot_to_wall), min_dist_left_foot_to_wall)

    # Reward is negative and increases in magnitude as robot gets closer to wall (within safety margin).
    # Using a negative exponential function for a continuous penalty that gets stronger closer to the wall.
    # Penalty is applied when min_dist_robot_to_wall is less than safety_margin.
    shaping_reward = torch.where(min_dist_robot_to_wall < safety_margin,
                                 -torch.exp(-(min_dist_robot_to_wall / safety_margin)) * 2.0, # Max penalty -2.0 at 0 distance
                                 torch.tensor(0.0, device=env.device))

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward) # Corrected: Use shaping_reward here
        RewNormalizer.update_stats(normaliser_name, shaping_reward) # Corrected: Use shaping_reward here
        return scaled_reward
    return shaping_reward # Corrected: Return shaping_reward here


@configclass
class TaskRewardsCfg:
    """
    Reward terms for the push_largeSphere_into_highWall skill.
    """
    # Main reward for pushing the sphere and making the wall fall
    MainPushLargeSphereIntoHighWallReward = RewTerm(
        func=main_push_largeSphere_into_highWall_reward,
        weight=1.0, # Primary reward, highest weight
        params={"normalise": True, "normaliser_name": "main_reward"}
    )

    # Shaping reward for approaching the sphere and maintaining stability
    ShapingApproachSphereAndStabilityReward = RewTerm(
        func=shaping_approach_sphere_and_stability_reward,
        weight=0.4, # Lower weight for shaping behavior
        params={"normalise": True, "normaliser_name": "shaping_approach_sphere_and_stability_reward"}
    )

    # Shaping reward for avoiding collision with the high wall
    ShapingCollisionAvoidanceHighWallReward = RewTerm(
        func=shaping_collision_avoidance_high_wall_reward,
        weight=0.3, # Lower weight for penalty/avoidance behavior
        params={"normalise": True, "normaliser_name": "shaping_collision_avoidance_high_wall_reward"}
    )