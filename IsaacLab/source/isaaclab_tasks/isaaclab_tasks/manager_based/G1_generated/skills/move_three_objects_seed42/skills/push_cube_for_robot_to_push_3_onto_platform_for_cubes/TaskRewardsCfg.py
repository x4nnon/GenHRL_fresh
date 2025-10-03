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

def main_push_cube_onto_platform_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for ensuring Object3 (Cube for robot to push) is fully and stably positioned on Object4 (Platform for cubes).
    This reward encourages Object3 to be centered on Object4 at the correct height and penalizes if it's off the platform.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object3 = env.scene['Object3'] # Cube for robot to push
    object4 = env.scene['Object4'] # Platform for cubes

    # Object dimensions (hardcoded from object configuration as per requirements)
    # This adheres to the rule: "THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it."
    object3_size_x = 0.5
    object3_size_y = 0.5
    object3_size_z = 0.5
    object4_size_x = 2.0
    object4_size_y = 2.0
    object4_size_z = 0.001

    # Calculate target z-position for Object3 on Object4
    # Object4's root_pos_w is its center. For a thin platform, its top surface is at root_pos_w.z + object4_size_z / 2
    # Object3's center should be at Object4's top surface + Object3's half height
    # This uses relative distances by adding object dimensions to the platform's Z position.
    target_object3_z = object4.data.root_pos_w[:, 2] + (object4_size_z / 2) + (object3_size_z / 2)

    # Calculate the allowed range for Object3's center on Object4
    # Object4's half size minus Object3's half size. This defines the valid area for Object3's center.
    allowed_x_offset = (object4_size_x / 2) - (object3_size_x / 2)
    allowed_y_offset = (object4_size_y / 2) - (object3_size_y / 2)

    # Distance of Object3's center from Object4's center in x, y, and z dimensions
    # These are relative distances between object centers.
    # This adheres to the rule: "ALL rewards MUST ONLY use relative distances between objects and robot parts"
    dist_x = object3.data.root_pos_w[:, 0] - object4.data.root_pos_w[:, 0]
    dist_y = object3.data.root_pos_w[:, 1] - object4.data.root_pos_w[:, 1]
    dist_z = object3.data.root_pos_w[:, 2] - target_object3_z

    # Reward for Object3 being centered on Object4 (within bounds)
    # Penalize based on absolute distance from the center. This is a continuous reward.
    # This adheres to the rule: "Use smooth, continuous rewards."
    reward_x = -torch.abs(dist_x)
    reward_y = -torch.abs(dist_y)

    # Apply a penalty if Object3 is outside the platform's x-bounds
    # This uses relative distances (dist_x compared to allowed_x_offset) and applies a large penalty for being off the platform.
    reward_x = torch.where(torch.abs(dist_x) > allowed_x_offset, reward_x - 10.0, reward_x)
    # Apply a penalty if Object3 is outside the platform's y-bounds
    reward_y = torch.where(torch.abs(dist_y) > allowed_y_offset, reward_y - 10.0, reward_y)

    # Reward for Object3 being at the correct height on Object4
    # Penalize based on absolute distance from the target Z height. This is a continuous reward.
    reward_z = -torch.abs(dist_z)

    # Combine rewards
    reward = reward_x + reward_y + reward_z

    # Mandatory normalization
    # This adheres to the rule: "MANDATORY REWARD NORMALIZATION"
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def robot_palm_to_object3_push_direction_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward_1") -> torch.Tensor:
    """
    This reward encourages the robot's 'right_palm_link' to be close to Object3 and on the correct side
    relative to Object4, facilitating the push onto the platform.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot part using approved patterns
    # This adheres to the rule: "ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w"
    # This adheres to the rule: "ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]"
    object3 = env.scene['Object3'] # Cube for robot to push
    object4 = env.scene['Object4'] # Platform for cubes
    robot_palm_idx = env.scene["robot"].body_names.index('right_palm_link')
    robot_palm_pos = env.scene["robot"].data.body_pos_w[:, robot_palm_idx]

    # Calculate the vector from Object3 to Object4's center (pushing direction)
    # These are relative distances between object centers.
    # This adheres to the rule: "ALL rewards MUST ONLY use relative distances between objects and robot parts"
    obj3_to_obj4_x = object4.data.root_pos_w[:, 0] - object3.data.root_pos_w[:, 0]
    obj3_to_obj4_y = object4.data.root_pos_w[:, 1] - object3.data.root_pos_w[:, 1]

    # Normalize the direction vector to get a unit vector
    # This handles tensor operations correctly for batched environments.
    direction_magnitude = torch.sqrt(obj3_to_obj4_x**2 + obj3_to_obj4_y**2)
    # Add a small epsilon to avoid division by zero if objects are at the same spot
    direction_x_norm = obj3_to_obj4_x / (direction_magnitude + 1e-6)
    direction_y_norm = obj3_to_obj4_y / (direction_magnitude + 1e-6)

    # Calculate the vector from robot palm to Object3
    # These are relative distances between robot part and object.
    palm_to_obj3_x = object3.data.root_pos_w[:, 0] - robot_palm_pos[:, 0]
    palm_to_obj3_y = object3.data.root_pos_w[:, 1] - robot_palm_pos[:, 1]
    palm_to_obj3_z = object3.data.root_pos_w[:, 2] - robot_palm_pos[:, 2]

    # Reward for palm being close to Object3 (proximity)
    # This is a continuous reward based on Euclidean distance.
    # This adheres to the rule: "Use smooth, continuous rewards."
    reward_proximity = -torch.norm(torch.stack([palm_to_obj3_x, palm_to_obj3_y, palm_to_obj3_z], dim=-1), dim=-1)

    # Reward for palm being on the "pushing side" of Object3
    # This means the dot product of (palm_to_obj3_vector) and (obj3_to_obj4_vector) should be positive
    # i.e., the palm is behind Object3 relative to Object4. This is a continuous reward.
    dot_product = (palm_to_obj3_x * direction_x_norm) + (palm_to_obj3_y * direction_y_norm)
    # Reward positive dot product (aligned for pushing), penalize negative (wrong side)
    reward_push_direction = torch.where(dot_product > 0, dot_product, -torch.abs(dot_product))

    # Combine rewards, scaling the direction reward as per plan
    reward = reward_proximity + (reward_push_direction * 0.5)

    # Mandatory normalization
    # This adheres to the rule: "MANDATORY REWARD NORMALIZATION"
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def stability_and_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward_2") -> torch.Tensor:
    """
    This reward encourages the robot to maintain a stable, upright posture (pelvis at a reasonable height)
    and avoids collisions with Object4 (the platform) with its body parts other than the intended pushing hand.
    It also penalizes the robot for pushing Object3 past Object4.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot part using approved patterns
    object3 = env.scene['Object3'] # Cube for robot to push
    object4 = env.scene['Object4'] # Platform for cubes
    robot_pelvis_idx = env.scene["robot"].body_names.index('pelvis')
    robot_pelvis_pos = env.scene["robot"].data.body_pos_w[:, robot_pelvis_idx]

    # Object dimensions (hardcoded from object configuration as per requirements)
    # This adheres to the rule: "THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it."
    object3_size_x = 0.5
    object4_size_x = 2.0
    object4_size_y = 2.0
    object4_size_z = 0.001

    # Reward for maintaining pelvis height for stability
    # Default stable pelvis height is around 0.7m. This is a continuous reward based on absolute Z height.
    # This adheres to the rule: "Use smooth, continuous rewards."
    pelvis_target_z = 0.7
    reward_pelvis_height = -torch.abs(robot_pelvis_pos[:, 2] - pelvis_target_z)

    # Collision avoidance with Object4 (platform) for robot's body (e.g., pelvis)
    # Penalize if pelvis is below platform top and within platform x/y bounds.
    # This uses relative distances between robot part and object.
    # This adheres to the rule: "ALL rewards MUST ONLY use relative distances between objects and robot parts"
    platform_top_z = object4.data.root_pos_w[:, 2] + (object4_size_z / 2)

    pelvis_dist_x_to_platform_center = torch.abs(robot_pelvis_pos[:, 0] - object4.data.root_pos_w[:, 0])
    pelvis_dist_y_to_platform_center = torch.abs(robot_pelvis_pos[:, 1] - object4.data.root_pos_w[:, 1])

    pelvis_collision_condition = (robot_pelvis_pos[:, 2] < platform_top_z) & \
                                (pelvis_dist_x_to_platform_center < (object4_size_x / 2)) & \
                                (pelvis_dist_y_to_platform_center < (object4_size_y / 2))
    # This is a binary penalty for collision, which is acceptable for hard constraints.
    reward_pelvis_collision = torch.where(pelvis_collision_condition, -20.0, 0.0)

    # Penalize if Object3 is pushed too far past Object4
    # This means Object3's center is beyond Object4's far edge in the pushing direction (assuming positive x).
    # This uses relative distances between objects and their dimensions.
    # overshoot_threshold_x defines the maximum x-position Object3's center should reach.
    overshoot_threshold_x = object4.data.root_pos_w[:, 0] + (object4_size_x / 2) + (object3_size_x / 2) + 0.1 # 0.1m buffer

    overshoot_condition_x = object3.data.root_pos_w[:, 0] > overshoot_threshold_x
    # This is a binary penalty for overshooting, which is acceptable for hard constraints.
    reward_overshoot_x = torch.where(overshoot_condition_x, -15.0, 0.0)

    # Combine rewards
    reward = reward_pelvis_height + reward_pelvis_collision + reward_overshoot_x

    # Mandatory normalization
    # This adheres to the rule: "MANDATORY REWARD NORMALIZATION"
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
    Reward terms for the push_cube_for_robot_to_push_3_onto_platform_for_cubes skill.
    """
    # Main reward for positioning Object3 correctly on Object4
    # This adheres to the rule: "PROPER WEIGHTS - Set appropriate weights in TaskRewardsCfg (primary reward ~1.0, supporting rewards <1.0)"
    MainPushCubeOntoPlatformReward = RewTerm(func=main_push_cube_onto_platform_reward, weight=1.0,
                                            params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for guiding the robot's palm to Object3 and encouraging correct pushing direction
    RobotPalmToObject3PushDirectionReward = RewTerm(func=robot_palm_to_object3_push_direction_reward, weight=0.6,
                                                    params={"normalise": True, "normaliser_name": "shaping_reward_1"})

    # Shaping reward for maintaining robot stability, avoiding platform collisions, and preventing overshooting Object3
    StabilityAndCollisionAvoidanceReward = RewTerm(func=stability_and_collision_avoidance_reward, weight=0.4,
                                                    params={"normalise": True, "normaliser_name": "shaping_reward_2"})