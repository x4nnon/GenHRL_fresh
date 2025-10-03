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
    Main reward for the doorway_and_goal_seed42 skill.
    The robot should walk through the doorway formed by two heavy cubes and then walk to the small block.
    This reward is phased:
    Phase 1: Encourages the robot to approach and pass through the doorway.
    Phase 2: Encourages the robot to approach the small block after passing the doorway.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects using approved patterns
    object1 = env.scene['Object1'] # Heavy Cube (Wall 1)
    object2 = env.scene['Object2'] # Heavy Cube (Wall 2)
    object3 = env.scene['Object3'] # Small Block

    # Access required robot part(s) using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]

    # Object dimensions (hardcoded from description as per rules)
    # "Two of these objects should be very heavy (1000kg) cubes with a z of 1.5m x of 0.5 and a y of 5m."
    wall_x_dim = 0.5 # x-dimension of the heavy cubes
    wall_y_dim = 5.0 # y-dimension of the heavy cubes (length along y-axis)
    # "The third object should be a small block of 0.3m cubed."
    block_size = 0.3 # side length of the small block

    # Calculate doorway center x and y for Phase 1
    # The doorway is formed by Object1 and Object2.
    # Doorway center x is the midpoint between the centers of Object1 and Object2.
    doorway_center_x = (object1.data.root_pos_w[:, 0] + object2.data.root_pos_w[:, 0]) / 2.0
    # Doorway y-coordinate is the y-position of the walls. Assuming they are aligned.
    doorway_y_pos = object1.data.root_pos_w[:, 1]

    # Phase 1: Approaching and passing through the doorway
    # Reward for moving towards the doorway's y-position (reducing distance in y to the doorway's y-center)
    # This encourages forward movement towards the doorway.
    distance_to_doorway_y = torch.abs(pelvis_pos_y - doorway_y_pos)
    reward_approach_doorway_y = -distance_to_doorway_y # Negative reward for distance, so closer is better.

    # Reward for being centered in the doorway's x-gap
    # This encourages the robot to align horizontally with the doorway.
    reward_center_x = -torch.abs(pelvis_pos_x - doorway_center_x) # Negative reward for distance from center.

    # Combine phase 1 rewards with weights
    reward_phase1 = reward_approach_doorway_y * 0.7 + reward_center_x * 0.3

    # Condition for transitioning to Phase 2: Pelvis has passed the doorway's y-coordinate
    # The walls are 5m long (y-dimension) and centered at doorway_y_pos.
    # The "far" end of the doorway along the y-axis is doorway_y_pos + wall_y_dim / 2.
    # We add a small buffer (0.1m) to ensure the robot is truly past the doorway.
    past_doorway_threshold_y = doorway_y_pos + (wall_y_dim / 2.0) + 0.1
    past_doorway_condition = (pelvis_pos_y > past_doorway_threshold_y)

    # Phase 2: Approaching the small block (Object3)
    # Reward for reducing the L1 distance to the small block.
    # This encourages the robot to move towards the final target.
    distance_to_block_x = torch.abs(pelvis_pos_x - object3.data.root_pos_w[:, 0])
    distance_to_block_y = torch.abs(pelvis_pos_y - object3.data.root_pos_w[:, 1])
    reward_approach_block = -(distance_to_block_x + distance_to_block_y) # L1 distance, negative for penalty.

    # Combine phases using torch.where for a continuous transition
    # If past the doorway, use phase 2 reward, otherwise use phase 1 reward.
    primary_reward = torch.where(past_doorway_condition, reward_approach_block, reward_phase1)

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, primary_reward)
        RewNormalizer.update_stats(normaliser_name, primary_reward)
        return scaled_reward
    return primary_reward

def collision_avoidance_doorway_walls_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward that penalizes the robot for getting too close to the heavy cubes (Object1 and Object2)
    that form the doorway. It encourages the robot to stay within the doorway gap.
    This reward is active only when the robot is approaching or within the y-extent of the doorway.
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

    # Object dimensions (hardcoded from description as per rules)
    wall_x_dim = 0.5 # x-dimension of the heavy cubes
    wall_y_dim = 5.0 # y-dimension of the heavy cubes (length along y-axis)

    # Calculate inner edges of the doorway for collision detection
    # Object1 is the left wall, Object2 is the right wall.
    # Inner edge of Object1 (right side of Object1)
    object1_inner_x = object1.data.root_pos_w[:, 0] + (wall_x_dim / 2.0)
    # Inner edge of Object2 (left side of Object2)
    object2_inner_x = object2.data.root_pos_w[:, 0] - (wall_x_dim / 2.0)

    # Collision distance threshold (e.g., robot's width / 2 + small buffer)
    # A small buffer (0.3m) to define "too close" to the inner edge of the walls.
    collision_threshold_x = 0.3

    # Reward for avoiding Object1 (left wall)
    # Penalize if pelvis_pos_x is less than object1_inner_x but too close to it.
    # The distance is positive if pelvis is to the right of the inner edge, negative if to the left.
    distance_to_object1_inner_x = pelvis_pos_x - object1_inner_x
    # Penalize if the robot is too far left (distance_to_object1_inner_x is negative and its absolute value is small)
    # We want distance_to_object1_inner_x to be >= collision_threshold_x.
    # If it's less, we penalize. The penalty should be stronger the closer it gets to the wall.
    # Using torch.clamp to ensure penalty is only applied when too close.
    # Max(0, threshold - actual_distance) gives a positive value when too close, 0 otherwise.
    penalty_object1 = torch.clamp(collision_threshold_x - distance_to_object1_inner_x, min=0.0)

    # Reward for avoiding Object2 (right wall)
    # Penalize if pelvis_pos_x is greater than object2_inner_x but too close to it.
    # The distance is positive if pelvis is to the left of the inner edge, negative if to the right.
    distance_to_object2_inner_x = object2_inner_x - pelvis_pos_x
    # Penalize if the robot is too far right (distance_to_object2_inner_x is negative and its absolute value is small)
    penalty_object2 = torch.clamp(collision_threshold_x - distance_to_object2_inner_x, min=0.0)

    # Condition: Only active when robot is within the y-extent of the doorway walls
    # Assuming walls are centered at object1.data.root_pos_w[:, 1] and extend wall_y_dim/2 in each direction.
    doorway_y_start = object1.data.root_pos_w[:, 1] - (wall_y_dim / 2.0)
    doorway_y_end = object1.data.root_pos_w[:, 1] + (wall_y_dim / 2.0)
    activation_condition = (pelvis_pos_y > doorway_y_start) & (pelvis_pos_y < doorway_y_end)

    # Combine penalties and apply condition
    # The total penalty is the sum of individual penalties.
    total_penalty = penalty_object1 + penalty_object2
    shaping_reward1 = torch.where(activation_condition, -total_penalty, torch.tensor(0.0, device=env.device))

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward1)
        RewNormalizer.update_stats(normaliser_name, shaping_reward1)
        return scaled_reward
    return shaping_reward1

def maintain_pelvis_height_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_height_reward") -> torch.Tensor:
    """
    Shaping reward that encourages the robot to maintain a stable upright posture
    by keeping its pelvis at a desired height (0.7m).
    This helps ensure stability throughout the skill.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required robot part(s) using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2] # Z-component is height

    # Desired pelvis height (hardcoded as per rules)
    desired_pelvis_z = 0.7

    # Reward for maintaining pelvis height
    # Penalize deviation from the desired height.
    shaping_reward2 = -torch.abs(pelvis_pos_z - desired_pelvis_z)

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward2)
        RewNormalizer.update_stats(normaliser_name, shaping_reward2)
        return scaled_reward
    return shaping_reward2

@configclass
class TaskRewardsCfg:
    # Primary reward for the main task objective
    MainDoorwayAndGoalReward = RewTerm(func=main_doorway_and_goal_reward, weight=0.1,
                                       params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for collision avoidance with doorway walls
    CollisionAvoidanceDoorwayWallsReward = RewTerm(func=collision_avoidance_doorway_walls_reward, weight=0.0,
                                                   params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})

    # Shaping reward for maintaining stable pelvis height
    MaintainPelvisHeightReward = RewTerm(func=maintain_pelvis_height_reward, weight=0.0, # Reduced weight as it's a general stability reward
                                          params={"normalise": True, "normaliser_name": "pelvis_height_reward"})