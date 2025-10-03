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

# Object dimensions (hardcoded from description as per rules)
# Rule: THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it.
OBJECT1_HEIGHT = 0.3
OBJECT2_HEIGHT = 0.6
OBJECT3_HEIGHT = 0.9
BLOCK_X_SIZE = 1.0
BLOCK_Y_SIZE = 1.0
PELVIS_STABLE_HEIGHT_ABOVE_BLOCK = 0.7 # Target pelvis height relative to block top

def climb_stairs_primary_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "climb_stairs_primary_reward") -> torch.Tensor:
    '''
    Primary reward for the climb_stairs skill.
    Guides the robot to sequentially land both feet on the top surface of Object1, then Object2, and finally Object3.
    Provides a continuous gradient based on the horizontal distance of the robot's feet to the center of the current target block
    and the vertical distance of the feet to the top surface of that block.
    The target block progresses once the robot's feet are sufficiently close to and on the current block.
    '''
    # Rule: MANDATORY REWARD NORMALIZATION - Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Rule: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']

    # Rule: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Calculate average foot position for simplicity in distance calculations
    # Rule: Handle tensor operations correctly for batched environments
    avg_foot_pos_x = (left_foot_pos[:, 0] + right_foot_pos[:, 0]) / 2
    avg_foot_pos_y = (left_foot_pos[:, 1] + right_foot_pos[:, 1]) / 2
    avg_foot_pos_z = (left_foot_pos[:, 2] + right_foot_pos[:, 2]) / 2

    # Define conditions for being "on" each block.
    # Feet are horizontally within block bounds and vertically on top of the block.
    # Vertical tolerance is half the block height + a small buffer (0.1m) to account for foot thickness/landing.
    # Rule: ALL rewards MUST ONLY use relative distances between objects and robot parts
    # Rule: NEVER use hard-coded positions or arbitrary thresholds. (0.1m is a small tolerance, not an arbitrary threshold for position)
    on_object1_condition = (torch.abs(avg_foot_pos_x - object1.data.root_pos_w[:, 0]) < BLOCK_X_SIZE / 2) & \
                           (torch.abs(avg_foot_pos_y - object1.data.root_pos_w[:, 1]) < BLOCK_Y_SIZE / 2) & \
                           (torch.abs(avg_foot_pos_z - (object1.data.root_pos_w[:, 2] + OBJECT1_HEIGHT / 2)) < OBJECT1_HEIGHT / 2 + 0.1)

    on_object2_condition = (torch.abs(avg_foot_pos_x - object2.data.root_pos_w[:, 0]) < BLOCK_X_SIZE / 2) & \
                           (torch.abs(avg_foot_pos_y - object2.data.root_pos_w[:, 1]) < BLOCK_Y_SIZE / 2) & \
                           (torch.abs(avg_foot_pos_z - (object2.data.root_pos_w[:, 2] + OBJECT2_HEIGHT / 2)) < OBJECT2_HEIGHT / 2 + 0.1)

    on_object3_condition = (torch.abs(avg_foot_pos_x - object3.data.root_pos_w[:, 0]) < BLOCK_X_SIZE / 2) & \
                           (torch.abs(avg_foot_pos_y - object3.data.root_pos_w[:, 1]) < BLOCK_Y_SIZE / 2) & \
                           (torch.abs(avg_foot_pos_z - (object3.data.root_pos_w[:, 2] + OBJECT3_HEIGHT / 2)) < OBJECT3_HEIGHT / 2 + 0.1)

    # Stage 1: Target Object1
    # Rule: Rewards should be continuous and positive where possible. Using negative distance for gradient.
    # Rule: Separate x, y, z components of distances.
    reward_stage1 = -torch.abs(avg_foot_pos_x - object1.data.root_pos_w[:, 0]) * 0.5 \
                    -torch.abs(avg_foot_pos_y - object1.data.root_pos_w[:, 1]) * 0.5 \
                    -torch.abs(avg_foot_pos_z - (object1.data.root_pos_w[:, 2] + OBJECT1_HEIGHT / 2)) * 1.0

    # Stage 2: Target Object2 (activated if on_object1_condition is met)
    reward_stage2 = -torch.abs(avg_foot_pos_x - object2.data.root_pos_w[:, 0]) * 0.5 \
                    -torch.abs(avg_foot_pos_y - object2.data.root_pos_w[:, 1]) * 0.5 \
                    -torch.abs(avg_foot_pos_z - (object2.data.root_pos_w[:, 2] + OBJECT2_HEIGHT / 2)) * 1.0

    # Stage 3: Target Object3 (activated if on_object2_condition is met)
    reward_stage3 = -torch.abs(avg_foot_pos_x - object3.data.root_pos_w[:, 0]) * 0.5 \
                    -torch.abs(avg_foot_pos_y - object3.data.root_pos_w[:, 1]) * 0.5 \
                    -torch.abs(avg_foot_pos_z - (object3.data.root_pos_w[:, 2] + OBJECT3_HEIGHT / 2)) * 1.0

    # Combine stages using torch.where for sequential progression
    # If on Object2, reward for Object3. Else if on Object1, reward for Object2. Else reward for Object1.
    primary_reward = torch.where(on_object2_condition, reward_stage3,
                       torch.where(on_object1_condition, reward_stage2, reward_stage1))

    # Add a bonus for successfully landing on each block
    # Rule: Rewards should be continuous and positive where possible. Bonuses are positive.
    primary_reward += torch.where(on_object1_condition, 0.5, 0.0) # Bonus for reaching Object1
    primary_reward += torch.where(on_object2_condition, 1.0, 0.0) # Bonus for reaching Object2
    primary_reward += torch.where(on_object3_condition, 2.0, 0.0) # Bonus for reaching Object3 (final goal)

    # Rule: MANDATORY REWARD NORMALIZATION - Complete normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, primary_reward)
        RewNormalizer.update_stats(normaliser_name, primary_reward)
        return scaled_reward
    return primary_reward

def climb_stairs_pelvis_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "climb_stairs_pelvis_stability_reward") -> torch.Tensor:
    '''
    Shaping reward to encourage pelvis stability and height.
    Encourages the robot to maintain a stable pelvis height (around 0.7m above the current block's surface)
    and reduces horizontal pelvis movement once on a block.
    '''
    # Rule: MANDATORY REWARD NORMALIZATION - Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Rule: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']

    # Rule: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Re-calculate conditions for being on a block (must be consistent with primary reward)
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    avg_foot_pos_x = (left_foot_pos[:, 0] + right_foot_pos[:, 0]) / 2
    avg_foot_pos_y = (left_foot_pos[:, 1] + right_foot_pos[:, 1]) / 2
    avg_foot_pos_z = (left_foot_pos[:, 2] + right_foot_pos[:, 2]) / 2

    on_object1_condition = (torch.abs(avg_foot_pos_x - object1.data.root_pos_w[:, 0]) < BLOCK_X_SIZE / 2) & \
                           (torch.abs(avg_foot_pos_y - object1.data.root_pos_w[:, 1]) < BLOCK_Y_SIZE / 2) & \
                           (torch.abs(avg_foot_pos_z - (object1.data.root_pos_w[:, 2] + OBJECT1_HEIGHT / 2)) < OBJECT1_HEIGHT / 2 + 0.1)

    on_object2_condition = (torch.abs(avg_foot_pos_x - object2.data.root_pos_w[:, 0]) < BLOCK_X_SIZE / 2) & \
                           (torch.abs(avg_foot_pos_y - object2.data.root_pos_w[:, 1]) < BLOCK_Y_SIZE / 2) & \
                           (torch.abs(avg_foot_pos_z - (object2.data.root_pos_w[:, 2] + OBJECT2_HEIGHT / 2)) < OBJECT2_HEIGHT / 2 + 0.1)

    on_object3_condition = (torch.abs(avg_foot_pos_x - object3.data.root_pos_w[:, 0]) < BLOCK_X_SIZE / 2) & \
                           (torch.abs(avg_foot_pos_y - object3.data.root_pos_w[:, 1]) < BLOCK_Y_SIZE / 2) & \
                           (torch.abs(avg_foot_pos_z - (object3.data.root_pos_w[:, 2] + OBJECT3_HEIGHT / 2)) < OBJECT3_HEIGHT / 2 + 0.1)

    # Target pelvis Z position relative to the current block's top surface
    # Rule: ALL rewards MUST ONLY use relative distances between objects and robot parts
    target_pelvis_z_on_object1 = object1.data.root_pos_w[:, 2] + OBJECT1_HEIGHT + PELVIS_STABLE_HEIGHT_ABOVE_BLOCK
    target_pelvis_z_on_object2 = object2.data.root_pos_w[:, 2] + OBJECT2_HEIGHT + PELVIS_STABLE_HEIGHT_ABOVE_BLOCK
    target_pelvis_z_on_object3 = object3.data.root_pos_w[:, 2] + OBJECT3_HEIGHT + PELVIS_STABLE_HEIGHT_ABOVE_BLOCK

    # Target pelvis XY position relative to the current block's center
    target_pelvis_x_on_object1 = object1.data.root_pos_w[:, 0]
    target_pelvis_y_on_object1 = object1.data.root_pos_w[:, 1]
    target_pelvis_x_on_object2 = object2.data.root_pos_w[:, 0]
    target_pelvis_y_on_object2 = object2.data.root_pos_w[:, 1]
    target_pelvis_x_on_object3 = object3.data.root_pos_w[:, 0]
    target_pelvis_y_on_object3 = object3.data.root_pos_w[:, 1]

    # Reward for stability on Object1
    # Rule: Rewards should be continuous.
    stability_reward_1 = -torch.abs(pelvis_pos_z - target_pelvis_z_on_object1) * 0.5 \
                         -torch.abs(pelvis_pos_x - target_pelvis_x_on_object1) * 0.2 \
                         -torch.abs(pelvis_pos_y - target_pelvis_y_on_object1) * 0.2
    stability_reward_1 = torch.where(on_object1_condition, stability_reward_1, torch.tensor(0.0, device=env.device))

    # Reward for stability on Object2
    stability_reward_2 = -torch.abs(pelvis_pos_z - target_pelvis_z_on_object2) * 0.5 \
                         -torch.abs(pelvis_pos_x - target_pelvis_x_on_object2) * 0.2 \
                         -torch.abs(pelvis_pos_y - target_pelvis_y_on_object2) * 0.2
    stability_reward_2 = torch.where(on_object2_condition, stability_reward_2, torch.tensor(0.0, device=env.device))

    # Reward for stability on Object3
    stability_reward_3 = -torch.abs(pelvis_pos_z - target_pelvis_z_on_object3) * 0.5 \
                         -torch.abs(pelvis_pos_x - target_pelvis_x_on_object3) * 0.2 \
                         -torch.abs(pelvis_pos_y - target_pelvis_y_on_object3) * 0.2
    stability_reward_3 = torch.where(on_object3_condition, stability_reward_3, torch.tensor(0.0, device=env.device))

    shaping_reward_1 = stability_reward_1 + stability_reward_2 + stability_reward_3

    # Rule: MANDATORY REWARD NORMALIZATION - Complete normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward_1)
        RewNormalizer.update_stats(normaliser_name, shaping_reward_1)
        return scaled_reward
    return shaping_reward_1

def climb_stairs_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "climb_stairs_collision_avoidance_reward") -> torch.Tensor:
    '''
    Shaping reward for collision avoidance.
    Penalizes the robot if its non-foot body parts collide with or get too close to the blocks.
    Encourages clean jumps and prevents snagging.
    '''
    # Rule: MANDATORY REWARD NORMALIZATION - Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Rule: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']

    # Rule: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]

    # Access relevant robot body parts for collision avoidance
    # Rule: NEVER use hard-coded body indices
    body_parts_to_check = ['pelvis', 'head_link', 'left_knee_link', 'right_knee_link',
                           'left_palm_link', 'right_palm_link']
    collision_penalty = torch.zeros(env.num_envs, device=env.device)

    # Rule: ALL rewards MUST ONLY use relative distances between objects and robot parts
    # Rule: Handle tensor operations correctly for batched environments
    for part_name in body_parts_to_check:
        part_idx = robot.body_names.index(part_name)
        part_pos = robot.data.body_pos_w[:, part_idx]
        part_pos_x = part_pos[:, 0]
        part_pos_y = part_pos[:, 1]
        part_pos_z = part_pos[:, 2]

        # Check proximity to Object1
        # Rule: Separate x, y, z components of distances.
        # Rule: NEVER use hard-coded positions or arbitrary thresholds. (0.1m is a small buffer, not an arbitrary threshold for position)
        dist_x_obj1 = torch.abs(part_pos_x - object1.data.root_pos_w[:, 0])
        dist_y_obj1 = torch.abs(part_pos_y - object1.data.root_pos_w[:, 1])
        dist_z_obj1 = torch.abs(part_pos_z - (object1.data.root_pos_w[:, 2] + OBJECT1_HEIGHT / 2))
        
        # Penalize if part is too close to the block's volume (excluding feet on top)
        # Use a small buffer (e.g., 0.1m) around the block dimensions for collision detection
        collision_condition_obj1 = (dist_x_obj1 < BLOCK_X_SIZE / 2 + 0.1) & \
                                   (dist_y_obj1 < BLOCK_Y_SIZE / 2 + 0.1) & \
                                   (dist_z_obj1 < OBJECT1_HEIGHT / 2 + 0.1)
        
        # This reward is for *non-foot* parts colliding.
        collision_penalty += torch.where(collision_condition_obj1, -0.5, 0.0)

        # Check proximity to Object2
        dist_x_obj2 = torch.abs(part_pos_x - object2.data.root_pos_w[:, 0])
        dist_y_obj2 = torch.abs(part_pos_y - object2.data.root_pos_w[:, 1])
        dist_z_obj2 = torch.abs(part_pos_z - (object2.data.root_pos_w[:, 2] + OBJECT2_HEIGHT / 2))
        collision_condition_obj2 = (dist_x_obj2 < BLOCK_X_SIZE / 2 + 0.1) & \
                                   (dist_y_obj2 < BLOCK_Y_SIZE / 2 + 0.1) & \
                                   (dist_z_obj2 < OBJECT2_HEIGHT / 2 + 0.1)
        collision_penalty += torch.where(collision_condition_obj2, -0.5, 0.0)

        # Check proximity to Object3
        dist_x_obj3 = torch.abs(part_pos_x - object3.data.root_pos_w[:, 0])
        dist_y_obj3 = torch.abs(part_pos_y - object3.data.root_pos_w[:, 1])
        dist_z_obj3 = torch.abs(part_pos_z - (object3.data.root_pos_w[:, 2] + OBJECT3_HEIGHT / 2))
        collision_condition_obj3 = (dist_x_obj3 < BLOCK_X_SIZE / 2 + 0.1) & \
                                   (dist_y_obj3 < BLOCK_Y_SIZE / 2 + 0.1) & \
                                   (dist_z_obj3 < OBJECT3_HEIGHT / 2 + 0.1)
        collision_penalty += torch.where(collision_condition_obj3, -0.5, 0.0)

    shaping_reward_2 = collision_penalty

    # Rule: MANDATORY REWARD NORMALIZATION - Complete normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward_2)
        RewNormalizer.update_stats(normaliser_name, shaping_reward_2)
        return scaled_reward
    return shaping_reward_2

@configclass
class TaskRewardsCfg:
    # Rule: Main reward with weight ~1.0
    ClimbStairsPrimaryReward = RewTerm(func=climb_stairs_primary_reward, weight=1.0, 
                                       params={"normalise": True, "normaliser_name": "climb_stairs_primary_reward"})
    
    # Rule: Supporting rewards with lower weights (typically 0.1-0.5)
    ClimbStairsPelvisStabilityReward = RewTerm(func=climb_stairs_pelvis_stability_reward, weight=0.4,
                                               params={"normalise": True, "normaliser_name": "climb_stairs_pelvis_stability_reward"})
    
    ClimbStairsCollisionAvoidanceReward = RewTerm(func=climb_stairs_collision_avoidance_reward, weight=0.3,
                                                  params={"normalise": True, "normaliser_name": "climb_stairs_collision_avoidance_reward"})