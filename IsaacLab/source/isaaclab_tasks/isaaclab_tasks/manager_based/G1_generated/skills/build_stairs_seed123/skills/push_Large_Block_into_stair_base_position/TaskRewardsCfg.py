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


def main_push_large_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for pushing the Large Block (Object3) to its designated stair base position.

    This reward encourages Object3 to reach a specific target X and Y coordinate, and to maintain its
    stable base Z height. The target coordinates are conceptual fixed points in the environment,
    derived from the task description for the stair base.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required object: Object3 (Large Block)
    # REASONING: Accessing object directly using approved pattern env.scene['ObjectN']
    object3 = env.scene['Object3']
    object3_pos = object3.data.root_pos_w

    # Define target position for Object3. These are fixed conceptual target coordinates for the object.
    # REASONING: Hardcoding target values derived from environment setup, as allowed for fixed environment targets.
    # Object3 starts at (4.0, 0.0, 0.45) and needs to be pushed to (2.0, 0.0, 0.45) based on task description.
    target_object3_x_pos = 2.0
    target_object3_y_pos = 0.0
    # Object3's height is 0.9m, so its stable base Z position (center) is 0.9 / 2.0 = 0.45m.
    # REASONING: Hardcoding object dimension (height) from object configuration, as required.
    object3_height = 0.9
    object3_base_z = object3_height / 2.0

    # Calculate the distance components between Object3's current position and its target position
    # REASONING: Using relative distances between object position and fixed target coordinates.
    # All operations are batched for multiple environments.
    distance_x = object3_pos[:, 0] - target_object3_x_pos
    distance_y = object3_pos[:, 1] - target_object3_y_pos
    distance_z = object3_pos[:, 2] - object3_base_z

    # Reward for Object3 reaching its target X, Y, and maintaining Z
    # REASONING: Using negative absolute differences for continuous, smooth rewards.
    reward_x = -torch.abs(distance_x)
    reward_y = -torch.abs(distance_y)
    reward_z = -torch.abs(distance_z) # Penalize deviation from base Z

    reward = reward_x + reward_y + reward_z

    # Mandatory reward normalization
    # REASONING: Applying the required normalization boilerplate.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def hand_to_block_proximity_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "hand_proximity_reward") -> torch.Tensor:
    '''Shaping reward: Encourages the robot's right hand to be close to Object3 and at an appropriate
    height for pushing, and positioned behind Object3 relative to the pushing direction.
    This reward is active only when Object3 still needs to be pushed towards its target.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts
    # REASONING: Accessing object and robot parts using approved patterns.
    robot = env.scene["robot"]
    object3 = env.scene['Object3']
    object3_pos = object3.data.root_pos_w

    right_palm_idx = robot.body_names.index('right_palm_link')
    right_palm_pos = robot.data.body_pos_w[:, right_palm_idx]

    # Object3 dimensions for relative height and pushing face
    # REASONING: Hardcoding object dimensions from object configuration, as required.
    object3_height = 0.9 # From object configuration: z=0.9m
    object3_width_x = 1.0 # From object configuration: x=1.0m
    object3_half_x = object3_width_x / 2.0

    # Define target hand position relative to Object3 for pushing along the -X axis
    # Hand should be slightly behind Object3's X-max face (e.g., 0.1m behind)
    # Target hand height for pushing (e.g., mid-height of the block, slightly lower for push)
    # REASONING: Using relative distances for target hand position.
    target_hand_x_relative = object3_pos[:, 0] + object3_half_x + 0.1 # 0.1m behind the block's +X face
    target_hand_y_relative = object3_pos[:, 1] # Aligned with block's Y center
    target_hand_z_relative = object3_pos[:, 2] + (object3_height / 2.0) - 0.1 # Slightly below center for pushing

    # Calculate distances from hand to the target relative position
    # REASONING: Using relative distances between robot part and object-relative target.
    distance_hand_object3_x = right_palm_pos[:, 0] - target_hand_x_relative
    distance_hand_object3_y = right_palm_pos[:, 1] - target_hand_y_relative
    distance_hand_object3_z = right_palm_pos[:, 2] - target_hand_z_relative

    # Reward for hand being close to Object3's back face and aligned in Y and Z
    # REASONING: Using negative absolute differences for continuous, smooth rewards.
    reward_hand_x = -torch.abs(distance_hand_object3_x)
    reward_hand_y = -torch.abs(distance_hand_object3_y)
    reward_hand_z = -torch.abs(distance_hand_object3_z)

    reward = reward_hand_x + reward_hand_y + reward_hand_z

    # Activation condition: Hand is behind Object3 (relative to push direction) AND Object3 is not yet at target X
    # REASONING: Conditional reward activation based on relative positions and task progress.
    # Target X for Object3 (same as in primary reward)
    target_object3_x_pos = 2.0
    # Condition 1: Hand's X position is greater than Object3's X position (assuming pushing along -X)
    is_hand_behind_block = right_palm_pos[:, 0] > object3_pos[:, 0]
    # Condition 2: Object3 is still far from its target X position (needs pushing)
    is_block_not_at_target = object3_pos[:, 0] > target_object3_x_pos + 0.1 # 0.1m buffer

    activation_condition = is_hand_behind_block & is_block_not_at_target
    # REASONING: Using torch.where for conditional reward application, ensuring batch compatibility.
    reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device))

    # Mandatory reward normalization
    # REASONING: Applying the required normalization boilerplate.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def robot_post_push_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "post_push_stability_reward") -> torch.Tensor:
    '''Shaping reward: Encourages the robot to maintain a stable standing posture (pelvis at default height)
    and to move away from Object3 once it has reached its target position, preparing for the next skill.
    This reward activates when Object3 is at or very close to its target X position.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts
    # REASONING: Accessing object and robot parts using approved patterns.
    robot = env.scene["robot"]
    object3 = env.scene['Object3']
    object3_pos = object3.data.root_pos_w

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Define default pelvis height for stability
    # REASONING: Hardcoding a conceptual default height for the robot's pelvis, as allowed for fixed robot posture targets.
    default_pelvis_z = 0.7

    # Define target position for Object3 (same as primary reward)
    # REASONING: Reusing the same fixed target coordinate for Object3, as allowed.
    target_object3_x_pos = 2.0

    # Reward for pelvis height
    # REASONING: Using negative absolute difference for continuous, smooth reward for pelvis height.
    reward_pelvis_z = -torch.abs(pelvis_pos[:, 2] - default_pelvis_z)

    # Reward for robot moving away from Object3 after push
    # Assuming robot pushes along -X, it should move to X < Object3.X after push
    # A safe distance, e.g., 1.0m behind Object3's target X
    # REASONING: Using relative distance for robot's X position relative to Object3's target, as allowed for fixed robot targets.
    target_robot_x_after_push = target_object3_x_pos - 1.0 # Robot should move to X=1.0 after push

    reward_robot_distance_x = -torch.abs(pelvis_pos[:, 0] - target_robot_x_after_push)

    reward = reward_pelvis_z + reward_robot_distance_x

    # Activation condition: Object3 is at or very close to its target X position
    # This reward activates when the pushing phase is mostly complete.
    # REASONING: Conditional reward activation based on object's proximity to its final target.
    activation_condition = (torch.abs(object3_pos[:, 0] - target_object3_x_pos) < 0.1) # Within 0.1m of target X
    # REASONING: Using torch.where for conditional reward application, ensuring batch compatibility.
    reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device))

    # Mandatory reward normalization
    # REASONING: Applying the required normalization boilerplate.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Primary reward for pushing the large block to its target position
    # REASONING: Main reward with weight 1.0 as it's the primary objective.
    MainPushLargeBlockReward = RewTerm(func=main_push_large_block_reward, weight=1.0,
                                       params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for guiding the hand to the block for pushing
    # REASONING: Shaping reward with lower weight (0.5) to guide behavior.
    HandToBlockProximityReward = RewTerm(func=hand_to_block_proximity_reward, weight=0.5,
                                         params={"normalise": True, "normaliser_name": "hand_proximity_reward"})

    # Shaping reward for robot stability and moving away after the push
    # REASONING: Shaping reward with lower weight (0.3) for post-action behavior.
    RobotPostPushStabilityReward = RewTerm(func=robot_post_push_stability_reward, weight=0.3,
                                           params={"normalise": True, "normaliser_name": "post_push_stability_reward"})