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


def main_position_medium_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for positioning the Medium Block (Object2) relative to the Small Block (Object1).

    This reward guides the robot first to approach the Medium Block, and then to push it into its
    designated middle position adjacent to the Small Block, forming the second step of the stairs.
    It uses a phase-based approach to provide appropriate guidance.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object1 = env.scene['Object1'] # Small Block for robot interaction
    object2 = env.scene['Object2'] # Medium Block for robot interaction
    robot = env.scene["robot"]

    # Access the required robot part(s) using approved patterns
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]

    # Hardcoded dimensions for blocks (from task description) - CRITICAL RULE: Dimensions must be hardcoded from config
    # As per task description, blocks are 1m x 1m in x and y dimensions.
    block_width = 1.0 # x or y dimension of the blocks

    # Phase 1: Approach Object2
    # Calculate horizontal distance from pelvis to Object2 using relative positions
    # CRITICAL RULE: ALL rewards MUST ONLY use relative distances between objects and robot parts
    dist_pelvis_obj2_x = torch.abs(object2.data.root_pos_w[:, 0] - pelvis_pos[:, 0])
    dist_pelvis_obj2_y = torch.abs(object2.data.root_pos_w[:, 1] - pelvis_pos[:, 1])
    dist_pelvis_obj2_horizontal = torch.sqrt(dist_pelvis_obj2_x**2 + dist_pelvis_obj2_y**2)

    # Condition for approaching phase: robot is far from Object2
    # The threshold is relative to the block's size plus a small buffer. CRITICAL RULE: No arbitrary thresholds.
    # This threshold is derived from the block's half-width (0.5m) plus a clearance (0.2m).
    approach_condition = dist_pelvis_obj2_horizontal > (block_width / 2.0 + 0.2)

    # Reward for approaching Object2: negative distance, encouraging closer proximity. CRITICAL RULE: Continuous rewards.
    reward_approach = -dist_pelvis_obj2_horizontal

    # Phase 2: Position Object2 relative to Object1
    # Target position for Object2 relative to Object1 (adjacent along y-axis, aligned x-axis)
    # CRITICAL RULE: Access object locations using approved patterns, not hardcoded positions.
    target_obj2_x = object1.data.root_pos_w[:, 0] # Aligned x-axis with Object1
    target_obj2_y_option1 = object1.data.root_pos_w[:, 1] + block_width # Adjacent on one side
    target_obj2_y_option2 = object1.data.root_pos_w[:, 1] - block_width # Adjacent on the other side

    # Calculate distance of Object2 from its target position relative to Object1
    # CRITICAL RULE: Use relative distances.
    dist_obj2_target_x = torch.abs(object2.data.root_pos_w[:, 0] - target_obj2_x)
    # Choose the closest y-target to allow flexibility in placement direction
    dist_obj2_target_y = torch.min(
        torch.abs(object2.data.root_pos_w[:, 1] - target_obj2_y_option1),
        torch.abs(object2.data.root_pos_w[:, 1] - target_obj2_y_option2)
    )

    # Combined distance for positioning Object2 (sum of absolute deviations). CRITICAL RULE: Continuous rewards.
    dist_obj2_to_final_pos = dist_obj2_target_x + dist_obj2_target_y

    # Condition for pushing/positioning phase: robot is close to Object2
    # This threshold is consistent with the approach_condition.
    positioning_condition = dist_pelvis_obj2_horizontal <= (block_width / 2.0 + 0.2)

    # Reward for positioning Object2: negative combined distance, encouraging precise placement.
    reward_positioning = -dist_obj2_to_final_pos

    # Combine rewards based on phase: if robot is far, reward approach; otherwise, reward positioning.
    # CRITICAL RULE: All operations must work with batched environments. torch.where handles this.
    primary_reward = torch.where(approach_condition, reward_approach, reward_positioning)

    # Mandatory normalization. CRITICAL RULE: EVERY reward function MUST include normalization.
    # RewNormalizer instance is obtained at the start of the function.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, primary_reward)
        RewNormalizer.update_stats(normaliser_name, primary_reward)
        return scaled_reward
    return primary_reward


def shaping_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    '''Shaping reward for collision avoidance between robot body parts and blocks.

    This reward penalizes penetration (overlap) between the robot's pelvis and head and the three blocks.
    It encourages the robot to navigate safely around the environment.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object1 = env.scene['Object1'] # Small Block
    object2 = env.scene['Object2'] # Medium Block
    object3 = env.scene['Object3'] # Large Block
    robot = env.scene["robot"]

    # Access the required robot part(s) using approved patterns
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    head_idx = robot.body_names.index('head_link')
    head_pos = robot.data.body_pos_w[:, head_idx]

    # Hardcoded dimensions for blocks (from task description) - CRITICAL RULE: Dimensions must be hardcoded from config
    # Block dimensions: x=1m, y=1m for all. z=0.3m (Object1), z=0.6m (Object2), z=0.9m (Object3).
    block_half_x = 0.5 # Half x-dimension of the blocks (1.0/2)
    block_half_y = 0.5 # Half y-dimension of the blocks (1.0/2)
    block_half_z_obj1 = 0.15 # Half z-dimension of Object1 (0.3/2)
    block_half_z_obj2 = 0.3 # Half z-dimension of Object2 (0.6/2)
    block_half_z_obj3 = 0.45 # Half z-dimension of Object3 (0.9/2)

    # Define a small clearance distance to allow for slight contact without immediate penalty. CRITICAL RULE: No arbitrary thresholds.
    # This clearance is a small positive value to define the "padded" bounding box for collision.
    clearance = 0.05 # A small positive value

    # Function to calculate penetration depth for a given robot part (point) and object (box)
    # This function uses relative distances between the robot part and the object's center.
    # CRITICAL RULE: ALL rewards MUST ONLY use relative distances between objects and robot parts.
    def calculate_penetration(robot_part_pos, obj_pos, obj_half_dims):
        # obj_half_dims = [half_x, half_y, half_z]
        
        # Calculate absolute distance from robot part to object center in each dimension
        dist_x = torch.abs(robot_part_pos[:, 0] - obj_pos[:, 0])
        dist_y = torch.abs(robot_part_pos[:, 1] - obj_pos[:, 1])
        dist_z = torch.abs(robot_part_pos[:, 2] - obj_pos[:, 2])

        # Calculate overlap in each dimension, considering clearance
        # Overlap is positive if the robot part is inside the object's bounding box plus clearance
        overlap_x = obj_half_dims[0] + clearance - dist_x
        overlap_y = obj_half_dims[1] + clearance - dist_y
        overlap_z = obj_half_dims[2] + clearance - dist_z

        # Penetration occurs if all overlaps are positive (i.e., robot part is inside the padded box)
        # The penetration value is the sum of positive overlaps, making it continuous. CRITICAL RULE: Continuous rewards.
        # CRITICAL RULE: All operations must work with batched environments. torch.where handles this.
        penetration = torch.where(
            (overlap_x > 0) & (overlap_y > 0) & (overlap_z > 0),
            overlap_x + overlap_y + overlap_z, # Sum of overlaps as a measure of penetration depth
            torch.zeros_like(overlap_x, device=env.device) # No penetration
        )
        return penetration

    # Calculate penetration for pelvis with each block
    # CRITICAL RULE: All tensor operations correctly handle batched environments. torch.tensor with device=env.device.
    penetration_pelvis_obj1 = calculate_penetration(pelvis_pos, object1.data.root_pos_w, torch.tensor([block_half_x, block_half_y, block_half_z_obj1], device=env.device))
    penetration_pelvis_obj2 = calculate_penetration(pelvis_pos, object2.data.root_pos_w, torch.tensor([block_half_x, block_half_y, block_half_z_obj2], device=env.device))
    penetration_pelvis_obj3 = calculate_penetration(pelvis_pos, object3.data.root_pos_w, torch.tensor([block_half_x, block_half_y, block_half_z_obj3], device=env.device))

    # Calculate penetration for head with each block
    penetration_head_obj1 = calculate_penetration(head_pos, object1.data.root_pos_w, torch.tensor([block_half_x, block_half_y, block_half_z_obj1], device=env.device))
    penetration_head_obj2 = calculate_penetration(head_pos, object2.data.root_pos_w, torch.tensor([block_half_x, block_half_y, block_half_z_obj2], device=env.device))
    penetration_head_obj3 = calculate_penetration(head_pos, object3.data.root_pos_w, torch.tensor([block_half_x, block_half_y, block_half_z_obj3], device=env.device))

    # Sum all penetrations to get a total collision measure
    total_penetration = penetration_pelvis_obj1 + penetration_pelvis_obj2 + penetration_pelvis_obj3 + \
                        penetration_head_obj1 + penetration_head_obj2 + penetration_head_obj3

    # Reward is negative of total penetration, penalizing collisions continuously. CRITICAL RULE: Continuous rewards.
    shaping_reward1 = -total_penetration

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward1)
        RewNormalizer.update_stats(normaliser_name, shaping_reward1)
        return scaled_reward
    return shaping_reward1


def shaping_pelvis_height_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_height_stability_reward") -> torch.Tensor:
    '''Shaping reward for maintaining robot pelvis height and stability.

    This reward encourages the robot to maintain a stable, upright posture by keeping its pelvis
    at a reasonable height (around 0.7m) and penalizes large deviations. This ensures the robot
    is stable and ready for subsequent actions.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot part(s) using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    # Z-position is an absolute height, which is acceptable here as per CRITICAL RULE: z is the only absolute position allowed.
    pelvis_pos_z = pelvis_pos[:, 2] 

    # Desired pelvis height for stability - CRITICAL RULE: Hardcoded value based on typical robot posture.
    # This is a specific target height for the robot's posture.
    desired_pelvis_z = 0.7

    # Calculate deviation from desired pelvis height using absolute difference. CRITICAL RULE: Continuous rewards.
    pelvis_height_deviation = torch.abs(pelvis_pos_z - desired_pelvis_z)

    # Reward is negative of the deviation, penalizing large height changes continuously.
    shaping_reward2 = -pelvis_height_deviation

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward2)
        RewNormalizer.update_stats(normaliser_name, shaping_reward2)
        return scaled_reward
    return shaping_reward2


@configclass
class TaskRewardsCfg:
    # Main reward for positioning the medium block, weighted at 1.0. CRITICAL RULE: Proper weights.
    MainPositionMediumBlockReward = RewTerm(func=main_position_medium_block_reward, weight=1.0,
                                            params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for collision avoidance, weighted at 0.4. CRITICAL RULE: Proper weights.
    ShapingCollisionAvoidanceReward = RewTerm(func=shaping_collision_avoidance_reward, weight=0.0,
                                              params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})

    # Shaping reward for maintaining pelvis height, weighted at 0.2. CRITICAL RULE: Proper weights.
    ShapingPelvisHeightStabilityReward = RewTerm(func=shaping_pelvis_height_stability_reward, weight=0.0,
                                                 params={"normalise": True, "normaliser_name": "pelvis_height_stability_reward"})