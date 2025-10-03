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

def approach_doorway_and_align_primary_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_doorway_and_align_primary_reward") -> torch.Tensor:
    """
    Primary reward for the 'approach_doorway_and_align' skill.
    Encourages the robot to walk towards the doorway, center itself between the walls,
    and align its body to face directly through the gap.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects using approved pattern
    # Object1: Heavy Cube (Wall 1) forming the left wall of the doorway
    object1 = env.scene['Object1']
    # Object2: Heavy Cube (Wall 2) forming the right wall of the doorway
    object2 = env.scene['Object2']

    # Access required robot part(s) using approved pattern
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_quat = robot.data.body_quat_w[:, pelvis_idx] # Quaternion (w, x, y, z)

    # Calculate doorway position as midpoint between object1 and object2
    doorway_pos = (object1.data.root_pos_w + object2.data.root_pos_w) / 2.0

    # Calculate distance from robot pelvis to doorway center
    # Using negative distance to encourage minimizing it, ensuring continuous reward
    distance_to_doorway = -torch.norm(pelvis_pos - doorway_pos, dim=1)

    # Calculate orientation alignment (facing through the doorway along world Y-axis)
    # The robot's local X-axis should align with the world Y-axis [0, 1, 0]
    # For quaternion (w, x, y, z), R_10 = 2*(qx*qy + qw*qz)
    orientation_dot_product = 2 * (pelvis_quat[:, 1] * pelvis_quat[:, 2] + pelvis_quat[:, 0] * pelvis_quat[:, 3])
    # Reward for orientation: penalize deviation from 1.0
    reward_orientation = -torch.abs(1.0 - orientation_dot_product)

    # Combine rewards - both components are continuous and based on relative positions/orientations
    reward = distance_to_doorway + reward_orientation

    # Mandatory reward normalization using the approved pattern
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def pelvis_height_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_height_stability_reward") -> torch.Tensor:
    """
    Shaping reward for maintaining a stable, upright posture by keeping the pelvis at a desired height.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required robot part(s) using approved pattern
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2] # Z-position is an absolute height in world coordinates, allowed for height.

    # Target pelvis height for stability (hardcoded as per plan)
    target_pelvis_z = 0.7

    # Reward for maintaining pelvis height close to target_pelvis_z.
    # Using negative absolute difference to encourage minimizing the deviation from target height, ensuring continuous reward.
    reward = -torch.abs(pelvis_pos_z - target_pelvis_z)

    # Mandatory reward normalization using the approved pattern
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def collision_avoidance_doorway_walls_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_doorway_walls_reward") -> torch.Tensor:
    """
    Shaping reward that penalizes the robot if any part of its body gets too close to or collides with the doorway walls.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects using approved pattern
    # Object1: Heavy Cube (Wall 1) forming the left wall of the doorway
    object1 = env.scene['Object1']
    # Object2: Heavy Cube (Wall 2) forming the right wall of the doorway
    object2 = env.scene['Object2']

    # Access required robot part(s) using approved pattern
    robot = env.scene["robot"]
    # List of robot body parts to monitor for collisions, including 'head_link' for robustness.
    monitored_body_parts = [
        'pelvis', 'left_ankle_roll_link', 'right_ankle_roll_link',
        'left_palm_link', 'right_palm_link', 'head_link'
    ]
    
    # Get positions for all monitored parts using approved pattern
    part_positions = []
    for part_name in monitored_body_parts:
        part_idx = robot.body_names.index(part_name)
        part_positions.append(robot.data.body_pos_w[:, part_idx])

    # Hardcoded object dimensions from the description for relative calculations
    # "x of 0.5" for the heavy cubes
    object_x_dim = 0.5
    # Define a safety margin for collision avoidance, ensuring continuous penalty.
    safety_margin = 0.1 # 10 cm buffer around the walls

    # Calculate the x-coordinates of the inner edges of the walls, considering the safety margin.
    # Object1 (left wall): its right edge is object1.data.root_pos_w[:, 0] + object_x_dim / 2.0
    # The safe boundary for robot parts to the right of Object1 is this edge + safety_margin.
    # Using approved pattern for object position access.
    obj1_safe_boundary_x = object1.data.root_pos_w[:, 0] + object_x_dim / 2.0 + safety_margin

    # Object2 (right wall): its left edge is object2.data.root_pos_w[:, 0] - object_x_dim / 2.0
    # The safe boundary for robot parts to the left of Object2 is this edge - safety_margin.
    # Using approved pattern for object position access.
    obj2_safe_boundary_x = object2.data.root_pos_w[:, 0] - object_x_dim / 2.0 - safety_margin

    # Initialize total penalty for batched environments
    total_penalty = torch.zeros_like(part_positions[0][:, 0]) # Use shape of any part's x-component

    # Iterate through each monitored robot part to calculate collision penalties
    for part_pos in part_positions:
        # Penalty for being too far left (colliding with Object1)
        # Penalize if part_pos_x is less than obj1_safe_boundary_x.
        # The distance into the forbidden zone is (obj1_safe_boundary_x - part_pos[:, 0]).
        # This distance is positive if the part is inside the forbidden zone.
        distance_into_obj1_forbidden_zone = obj1_safe_boundary_x - part_pos[:, 0]
        # Apply penalty only if the part is inside the forbidden zone (distance > 0).
        # The penalty is linear, proportional to how deep the part is into the zone, ensuring continuous reward.
        total_penalty += torch.where(distance_into_obj1_forbidden_zone > 0, -distance_into_obj1_forbidden_zone, 0.0)

        # Penalty for being too far right (colliding with Object2)
        # Penalize if part_pos_x is greater than obj2_safe_boundary_x.
        # The distance into the forbidden zone is (part_pos[:, 0] - obj2_safe_boundary_x).
        distance_into_obj2_forbidden_zone = part_pos[:, 0] - obj2_safe_boundary_x
        # Apply penalty only if the part is inside the forbidden zone (distance > 0).
        # The penalty is linear, proportional to how deep the part is into the zone, ensuring continuous reward.
        total_penalty += torch.where(distance_into_obj2_forbidden_zone > 0, -distance_into_obj2_forbidden_zone, 0.0)

    # The reward is the sum of all penalties (which are negative values).
    # This is a continuous reward based on relative distances.
    reward = total_penalty

    # Mandatory reward normalization using the approved pattern
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
    Configuration for the rewards used in the 'approach_doorway_and_align' skill.
    """
    # Primary reward for approaching the doorway, centering, and aligning.
    # Weight 1.0 indicates this is the main objective.
    approach_doorway_and_align_primary = RewTerm(
        func=approach_doorway_and_align_primary_reward,
        weight=1.0,
        params={"normalise": True, "normaliser_name": "approach_doorway_and_align_primary_reward"}
    )

    # Shaping reward for maintaining stable pelvis height.
    # Weight 0.4 encourages good posture without dominating the primary goal.
    pelvis_height_stability = RewTerm(
        func=pelvis_height_stability_reward,
        weight=0.4,
        params={"normalise": True, "normaliser_name": "pelvis_height_stability_reward"}
    )

    # Shaping reward for avoiding collisions with doorway walls.
    # Weight 0.5 provides a significant penalty for unsafe navigation.
    collision_avoidance_doorway_walls = RewTerm(
        func=collision_avoidance_doorway_walls_reward,
        weight=0.5,
        params={"normalise": True, "normaliser_name": "collision_avoidance_doorway_walls_reward"}
    )