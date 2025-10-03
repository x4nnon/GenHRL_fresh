from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.utils import math_utils
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


def main_navigate_through_doorway_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the navigate_through_doorway skill.
    The robot should approach the doorway, align itself, walk completely through it, and then walk to the small block (Object3).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects using approved patterns
    object1 = env.scene['Object1'] # Heavy Cube (Wall 1)
    object2 = env.scene['Object2'] # Heavy Cube (Wall 2)
    object3 = env.scene['Object3'] # small block for the robot to walk to

    # Access required robot part(s) using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Pelvis position in world frame
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Calculate doorway center coordinates based on Object1 and Object2 positions.
    # The doorway is a gap in x, formed by two walls.
    # Object1 and Object2 are walls with x=0.5m, y=5m, z=1.5m.
    # The doorway gap is 0.5m.
    # Assuming Object1 is to the left and Object2 to the right, and they are aligned in y.
    # The doorway center x is the average of their x-coordinates.
    doorway_center_x = (object1.data.root_pos_w[:, 0] + object2.data.root_pos_w[:, 0]) / 2.0
    # The doorway y-position is the average y of the two walls.
    doorway_y_center = (object1.data.root_pos_w[:, 1] + object2.data.root_pos_w[:, 1]) / 2.0

    # Target y-position for skill completion: Object3's y-coordinate.
    # This uses a relative position to Object3, avoiding hard-coded absolute values.
    target_y_pos = object3.data.root_pos_w[:, 1] # Robot needs to walk to Object3

    # Reward for moving along y-axis towards the target_y_pos (Object3).
    # This encourages forward progress towards the final goal.
    # Using a negative absolute difference for a continuous reward that is maximized at target_y_pos.
    y_progress_reward = -torch.abs(pelvis_pos_y - target_y_pos)

    # Reward for x-alignment with the doorway center.
    # This encourages the robot to stay centered horizontally when passing through the doorway.
    x_alignment_reward = -torch.abs(pelvis_pos_x - doorway_center_x)

    # Encourage standing upright.
    # A small penalty for deviation from a target pelvis height (e.g., 0.7m for standing).
    # This is a relative target height for the robot's own posture.
    pelvis_z_target = 0.7 # Approximate standing height for the robot's pelvis
    upright_reward = -torch.abs(pelvis_pos_z - pelvis_z_target) * 0.1 # Small weight to make it a minor shaping factor

    # Combine rewards with dynamic weighting for x-alignment.
    # The x-alignment reward is more critical when the robot is near or inside the doorway.
    # A Gaussian function is used to smoothly activate the x-alignment reward based on the pelvis's y-position
    # relative to the doorway's y-center. The standard deviation (0.5) defines the "width" of the activation region.
    x_alignment_weight = torch.exp(-((pelvis_pos_y - doorway_y_center)**2) / (2 * (0.5**2))) * 0.5

    # Primary reward is always active, driving progress towards Object3.
    reward = y_progress_reward * 1.0
    # Add x-alignment reward, weighted to be more influential near the doorway.
    reward += x_alignment_reward * x_alignment_weight
    # Add upright reward as a general stability encouragement.
    reward += upright_reward

    # Mandatory reward normalization using the approved pattern.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_reward") -> torch.Tensor:
    """
    Shaping reward for collision avoidance with the doorway walls (Object1 and Object2).
    Penalizes the robot if its pelvis or feet get too close to or collide with the walls.
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

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Hardcoded wall dimensions from the environment description.
    # This is necessary as object dimensions cannot be accessed dynamically from RigidObjectData.
    wall_x_dim = 0.5 # Depth of the wall (from "x of 0.5")
    wall_y_dim = 5.0 # Length of the wall (from "y of 5m")
    wall_z_dim = 1.5 # Height of the wall (from "z of 1.5m")
    robot_radius_approx = 0.2 # Approximate robot body radius for clearance, a hardcoded value.

    # Initialize collision reward tensor
    collision_reward = torch.zeros_like(pelvis_pos[:, 0])

    # Define thresholds for collision detection for each dimension.
    # These thresholds are half the wall dimension plus an approximate robot radius for clearance.
    threshold_x = (wall_x_dim / 2.0) + robot_radius_approx
    threshold_y = (wall_y_dim / 2.0) + robot_radius_approx
    threshold_z = (wall_z_dim / 2.0) + robot_radius_approx

    # Calculate relative distances for pelvis to Object1 (left wall)
    dist_pelvis_obj1_x = torch.abs(pelvis_pos[:, 0] - object1.data.root_pos_w[:, 0])
    dist_pelvis_obj1_y = torch.abs(pelvis_pos[:, 1] - object1.data.root_pos_w[:, 1])
    dist_pelvis_obj1_z = torch.abs(pelvis_pos[:, 2] - object1.data.root_pos_w[:, 2])

    # Calculate relative distances for pelvis to Object2 (right wall)
    dist_pelvis_obj2_x = torch.abs(pelvis_pos[:, 0] - object2.data.root_pos_w[:, 0])
    dist_pelvis_obj2_y = torch.abs(pelvis_pos[:, 1] - object2.data.root_pos_w[:, 1])
    dist_pelvis_obj2_z = torch.abs(pelvis_pos[:, 2] - object2.data.root_pos_w[:, 2])

    # Calculate relative distances for left foot to Object1
    dist_lfoot_obj1_x = torch.abs(left_foot_pos[:, 0] - object1.data.root_pos_w[:, 0])
    dist_lfoot_obj1_y = torch.abs(left_foot_pos[:, 1] - object1.data.root_pos_w[:, 1])
    dist_lfoot_obj1_z = torch.abs(left_foot_pos[:, 2] - object1.data.root_pos_w[:, 2])

    # Calculate relative distances for right foot to Object1
    dist_rfoot_obj1_x = torch.abs(right_foot_pos[:, 0] - object1.data.root_pos_w[:, 0])
    dist_rfoot_obj1_y = torch.abs(right_foot_pos[:, 1] - object1.data.root_pos_w[:, 1])
    dist_rfoot_obj1_z = torch.abs(right_foot_pos[:, 2] - object1.data.root_pos_w[:, 2])

    # Calculate relative distances for left foot to Object2
    dist_lfoot_obj2_x = torch.abs(left_foot_pos[:, 0] - object2.data.root_pos_w[:, 0])
    dist_lfoot_obj2_y = torch.abs(left_foot_pos[:, 1] - object2.data.root_pos_w[:, 1])
    dist_lfoot_obj2_z = torch.abs(left_foot_pos[:, 2] - object2.data.root_pos_w[:, 2])

    # Calculate relative distances for right foot to Object2
    dist_rfoot_obj2_x = torch.abs(right_foot_pos[:, 0] - object2.data.root_pos_w[:, 0])
    dist_rfoot_obj2_y = torch.abs(right_foot_pos[:, 1] - object2.data.root_pos_w[:, 1])
    dist_rfoot_obj2_z = torch.abs(right_foot_pos[:, 2] - object2.data.root_pos_w[:, 2])

    # Penalize if any part is within collision threshold for Object1.
    # The penalty is scaled by how deep the penetration is (1.0 - (distance / threshold)).
    # A small weight (0.5) is applied to the penalty.
    # All dimensions (x, y, z) are considered for collision detection.
    collision_pelvis_obj1 = (dist_pelvis_obj1_x < threshold_x) & (dist_pelvis_obj1_y < threshold_y) & (dist_pelvis_obj1_z < threshold_z)
    collision_reward -= torch.where(collision_pelvis_obj1, 1.0 - (dist_pelvis_obj1_x / threshold_x), 0.0) * 0.5

    collision_lfoot_obj1 = (dist_lfoot_obj1_x < threshold_x) & (dist_lfoot_obj1_y < threshold_y) & (dist_lfoot_obj1_z < threshold_z)
    collision_reward -= torch.where(collision_lfoot_obj1, 1.0 - (dist_lfoot_obj1_x / threshold_x), 0.0) * 0.5

    collision_rfoot_obj1 = (dist_rfoot_obj1_x < threshold_x) & (dist_rfoot_obj1_y < threshold_y) & (dist_rfoot_obj1_z < threshold_z)
    collision_reward -= torch.where(collision_rfoot_obj1, 1.0 - (dist_rfoot_obj1_x / threshold_x), 0.0) * 0.5

    # Penalize if any part is within collision threshold for Object2.
    collision_pelvis_obj2 = (dist_pelvis_obj2_x < threshold_x) & (dist_pelvis_obj2_y < threshold_y) & (dist_pelvis_obj2_z < threshold_z)
    collision_reward -= torch.where(collision_pelvis_obj2, 1.0 - (dist_pelvis_obj2_x / threshold_x), 0.0) * 0.5

    collision_lfoot_obj2 = (dist_lfoot_obj2_x < threshold_x) & (dist_lfoot_obj2_y < threshold_y) & (dist_lfoot_obj2_z < threshold_z)
    collision_reward -= torch.where(collision_lfoot_obj2, 1.0 - (dist_lfoot_obj2_x / threshold_x), 0.0) * 0.5

    collision_rfoot_obj2 = (dist_rfoot_obj2_x < threshold_x) & (dist_rfoot_obj2_y < threshold_y) & (dist_rfoot_obj2_z < threshold_z)
    collision_reward -= torch.where(collision_rfoot_obj2, 1.0 - (dist_rfoot_obj2_x / threshold_x), 0.0) * 0.5

    # Mandatory reward normalization using the approved pattern.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, collision_reward)
        RewNormalizer.update_stats(normaliser_name, collision_reward)
        return scaled_reward
    return collision_reward


def forward_orientation_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "orientation_reward") -> torch.Tensor:
    """
    Shaping reward that encourages the robot to maintain a forward-facing orientation (along the positive y-axis)
    while traversing the doorway and moving towards Object3.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required robot part(s) using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos_y = robot.data.body_pos_w[:, pelvis_idx, 1] # Pelvis y-position

    # Access objects to define the doorway region for activation using approved patterns
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']
    doorway_y_center = (object1.data.root_pos_w[:, 1] + object2.data.root_pos_w[:, 1]) / 2.0

    # Compute the y-component of the pelvis's local y-axis (forward direction) in world coordinates using quaternion.
    # Convert local forward axis [0, 1, 0] by the pelvis world orientation and take its y-component.
    pelvis_quat_w = robot.data.body_quat_w[:, pelvis_idx]
    local_forward_y_axis = torch.tensor([0.0, 1.0, 0.0], device=env.device, dtype=pelvis_quat_w.dtype)
    local_forward_y_axis = local_forward_y_axis.unsqueeze(0).expand(pelvis_quat_w.shape[0], 3)
    pelvis_forward_vec_w = math_utils.quat_apply(pelvis_quat_w, local_forward_y_axis)
    pelvis_forward_y = pelvis_forward_vec_w[:, 1]

    # Reward for facing forward (positive y-direction).
    # This reward is maximized when pelvis_forward_y is 1.0.
    orientation_reward = pelvis_forward_y

    # Activation condition: Only apply this reward when the robot is approaching or inside the doorway region.
    # This prevents the reward from being overly restrictive when the robot is far away from the task area.
    # The region is defined as 1.0m before the doorway center and 2.0m past it, using relative distances.
    activation_condition = (pelvis_pos_y > doorway_y_center - 1.0) & (pelvis_pos_y < doorway_y_center + 2.0)

    # Apply the orientation reward only when the activation condition is met, otherwise reward is 0.
    # This ensures the reward is continuous and smoothly activated.
    reward = torch.where(activation_condition, orientation_reward, torch.tensor(0.0, device=env.device))

    # Mandatory reward normalization using the approved pattern.
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
    Reward terms for the navigate_through_doorway skill.
    """
    # Main reward for progressing through the doorway and towards the final block.
    # Weight is 1.0 as it's the primary objective.
    main_reward = RewTerm(func=main_navigate_through_doorway_reward, weight=1.0,
                          params={"normalise": True, "normaliser_name": "main_navigate_through_doorway_reward"})

    # Shaping reward to penalize collisions with the doorway walls.
    # Weight is 0.6, a significant shaping factor to avoid collisions.
    collision_avoidance = RewTerm(func=collision_avoidance_reward, weight=0.6,
                                  params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})

    # Shaping reward to encourage maintaining a forward orientation.
    # Weight is 0.3, a moderate shaping factor to guide orientation.
    forward_orientation = RewTerm(func=forward_orientation_reward, weight=0.3,
                                  params={"normalise": True, "normaliser_name": "forward_orientation_reward"})