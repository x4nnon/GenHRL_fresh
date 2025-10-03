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


def knock_over_cylinder3_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the 'knock_over_cylinderColumn3' skill.
    Encourages the robot to approach Object3 and knock it over until it falls completely.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts using approved patterns
    object3 = env.scene['Object3'] # Accessing Object3 directly as per requirements
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern

    # Hardcode object dimensions from the description (z=2m, radius=0.3m)
    # This adheres to the rule of hardcoding dimensions from the object configuration, not accessing from RigidObject.
    object3_radius = 0.3
    object3_height = 2.0
    object3_standing_z = object3_height / 2.0 # Center of mass when standing (1.0m for 2m height)
    object3_fallen_z = object3_radius # Center of mass when fallen (0.3m for 0.3m radius)

    # Calculate horizontal distance from pelvis to Object3
    # Using relative distances between object and robot part positions, adhering to requirement 1.
    dist_pelvis_obj3_x = object3.data.root_pos_w[:, 0] - pelvis_pos[:, 0]
    dist_pelvis_obj3_y = object3.data.root_pos_w[:, 1] - pelvis_pos[:, 1]
    horizontal_dist_pelvis_obj3 = torch.sqrt(dist_pelvis_obj3_x**2 + dist_pelvis_obj3_y**2)

    # Reward for approaching Object3 horizontally
    # This reward is maximized when the horizontal distance is equal to the object's radius,
    # encouraging the robot to get just close enough for contact. This is a continuous reward.
    approach_reward = -torch.abs(horizontal_dist_pelvis_obj3 - object3_radius)

    # Reward for knocking Object3 over (decreasing its Z-position)
    # This reward is positive and increases as Object3's Z-position decreases from its standing height.
    # It's scaled to make it significant. This is a continuous reward.
    knock_over_reward = (object3_standing_z - object3.data.root_pos_w[:, 2]) * 2.0

    # Blend approach and knock-over rewards based on horizontal distance
    # This ensures a smooth transition: approach is prioritized when far, knock-over when close.
    # This uses relative distance for blending, adhering to requirement 1.
    transition_start_dist = 1.0 # Robot starts focusing on knock-over when within 1.0m
    transition_end_dist = object3_radius # Fully focused on knock-over when at contact distance

    # Calculate interpolation factor (0 when far, 1 when close)
    # Clamped between 0 and 1 to ensure valid blending weights.
    interpolation_factor = torch.clamp(
        (transition_start_dist - horizontal_dist_pelvis_obj3) / (transition_start_dist - transition_end_dist),
        0.0, 1.0
    )

    # Primary reward is a blend: more approach when far, more knock_over when close
    reward = (1.0 - interpolation_factor) * approach_reward + interpolation_factor * knock_over_reward

    # Add a bonus for Object3 being completely fallen
    # This is a continuous bonus, maximized when Object3's Z-position is at its fallen height.
    # A stronger penalty is applied if it's not at the fallen height.
    fallen_bonus = -torch.abs(object3.data.root_pos_w[:, 2] - object3_fallen_z) * 5.0

    reward += fallen_bonus

    # Mandatory normalization, adhering to requirement 6.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_reward") -> torch.Tensor:
    """
    Shaping reward to penalize collisions between robot body parts and any of the cylinder columns.
    Encourages safe and controlled movement.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts using approved patterns
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']
    object4 = env.scene['Object4']
    object5 = env.scene['Object5']
    robot = env.scene["robot"]

    # Robot body parts to monitor for collisions
    robot_parts_to_monitor = ['left_palm_link', 'right_palm_link', 'left_ankle_roll_link', 'right_ankle_roll_link', 'pelvis', 'head_link']
    collision_reward = torch.zeros(env.num_envs, device=env.device) # Initialize reward tensor for batch processing

    # Hardcode cylinder dimensions from the description (radius=0.3m, height=2m)
    # Adheres to the rule of hardcoding dimensions.
    cylinder_radius = 0.3
    cylinder_half_height = 1.0 # Half of 2m height

    for part_name in robot_parts_to_monitor:
        part_idx = robot.body_names.index(part_name) # Accessing robot part index
        part_pos = robot.data.body_pos_w[:, part_idx] # Accessing robot part position

        for obj in [object1, object2, object3, object4, object5]:
            obj_pos = obj.data.root_pos_w # Accessing object position

            # Calculate relative distance vector between robot part and object center, adhering to requirement 1.
            dist_x = obj_pos[:, 0] - part_pos[:, 0]
            dist_y = obj_pos[:, 1] - part_pos[:, 1]
            dist_z = obj_pos[:, 2] - part_pos[:, 2]

            # Horizontal distance to cylinder center
            horizontal_dist = torch.sqrt(dist_x**2 + dist_y**2)

            # Approximate vertical extent of cylinder (from root_pos_w)
            obj_min_z = obj_pos[:, 2] - cylinder_half_height
            obj_max_z = obj_pos[:, 2] + cylinder_half_height

            # Condition for vertical overlap: robot part Z is within cylinder's vertical range
            vertical_overlap_condition = (part_pos[:, 2] > obj_min_z) & (part_pos[:, 2] < obj_max_z)

            # Calculate penetration depth: how much the robot part is inside the cylinder's horizontal radius
            penetration_depth = cylinder_radius - horizontal_dist

            # Apply continuous penalty if horizontal penetration and vertical overlap
            # Penalty increases linearly with penetration depth. This is a continuous reward.
            collision_penalty = torch.where(
                (penetration_depth > 0) & vertical_overlap_condition,
                -penetration_depth * 5.0, # Stronger penalty for deeper penetration
                torch.zeros_like(penetration_depth) # No penalty if no penetration or no vertical overlap
            )
            collision_reward += collision_penalty

    # Mandatory normalization, adhering to requirement 6.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, collision_reward)
        RewNormalizer.update_stats(normaliser_name, collision_reward)
        return scaled_reward
    return collision_reward


def stability_and_next_skill_readiness_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stability_readiness_reward") -> torch.Tensor:
    """
    Shaping reward for robot stability and readiness for the next skill.
    Encourages maintaining a stable pelvis height and not overshooting Object3 relative to Object4.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts using approved patterns
    object3 = env.scene['Object3']
    object4 = env.scene['Object4'] # Used for relative positioning for next skill
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Target pelvis Z for stability (common standing height for humanoid robots)
    # This is a hardcoded target height, which is acceptable for a stability reward.
    target_pelvis_z = 0.7

    # Reward for maintaining pelvis height (stability)
    # Penalizes deviation from the target Z-height. This is a continuous reward.
    stability_reward = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z) * 0.5

    # Reward for not overshooting Object3 relative to Object4
    # This encourages the robot to stop near Object3 and be oriented towards Object4.
    # All calculations use relative distances between objects and robot parts, adhering to requirement 1.
    obj3_pos = object3.data.root_pos_w
    obj4_pos = object4.data.root_pos_w

    # Calculate the vector from Object3 to Object4 (relative distance)
    obj3_to_obj4_x = obj4_pos[:, 0] - obj3_pos[:, 0]
    obj3_to_obj4_y = obj4_pos[:, 1] - obj3_pos[:, 1]

    # Calculate the vector from Object3 to robot pelvis (relative distance)
    obj3_to_pelvis_x = pelvis_pos[:, 0] - obj3_pos[:, 0]
    obj3_to_pelvis_y = pelvis_pos[:, 1] - obj3_pos[:, 1]

    # Calculate squared magnitude of obj3_to_obj4 vector
    obj3_to_obj4_mag_sq = obj3_to_obj4_x**2 + obj3_to_obj4_y**2
    # Add a small epsilon to avoid division by zero if objects are coincident (unlikely but good practice)
    obj3_to_obj4_mag_sq = torch.where(obj3_to_obj4_mag_sq < 1e-6, torch.ones_like(obj3_to_obj4_mag_sq) * 1e-6, obj3_to_obj4_mag_sq)

    # Project robot's position onto the line from Object3 to Object4
    # This scalar indicates how far along the line the robot is past Object3.
    projection_scalar = (obj3_to_pelvis_x * obj3_to_obj4_x + obj3_to_pelvis_y * obj3_to_obj4_y) / obj3_to_obj4_mag_sq

    # Penalize if projection_scalar is significantly greater than a small buffer (overshoot)
    overshoot_threshold = 0.5 # Robot can be 0.5m past Object3 along the line to Object4 without penalty

    overshoot_penalty = torch.where(
        projection_scalar > overshoot_threshold,
        -(projection_scalar - overshoot_threshold) * 2.0, # Penalty increases with overshoot. This is a continuous reward.
        torch.zeros_like(projection_scalar) # No penalty if within threshold or behind Object3
    )

    reward = stability_reward + overshoot_penalty

    # Mandatory normalization, adhering to requirement 6.
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
    Reward terms for the 'knock_over_cylinderColumn3' skill.
    """
    # Main reward for knocking over Cylinder Column 3
    # Weight is 1.0 as per requirements for primary reward.
    main_knock_over_cylinder3_reward = RewTerm(
        func=knock_over_cylinder3_reward,
        weight=1.0,
        params={"normalise": True, "normaliser_name": "main_knock_over_cylinder3_reward"}
    )

    # Shaping reward for collision avoidance with any cylinder column
    # Weight is 0.6 as per requirements for significant shaping reward.
    collision_avoidance_reward = RewTerm(
        func=collision_avoidance_reward,
        weight=0.6,
        params={"normalise": True, "normaliser_name": "collision_avoidance_reward"}
    )

    # Shaping reward for robot stability and readiness for the next skill
    # Weight is 0.3 as per requirements for moderate shaping reward.
    stability_and_next_skill_readiness_reward = RewTerm(
        func=stability_and_next_skill_readiness_reward,
        weight=0.3,
        params={"normalise": True, "normaliser_name": "stability_and_next_skill_readiness_reward"}
    )