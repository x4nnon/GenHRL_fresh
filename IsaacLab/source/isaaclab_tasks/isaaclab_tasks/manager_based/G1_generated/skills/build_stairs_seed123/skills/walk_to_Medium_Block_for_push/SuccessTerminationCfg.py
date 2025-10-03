
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_Medium_Block_for_push_success(env) -> torch.Tensor:
    '''Determine if the walk_to_Medium_Block_for_push skill has been successfully completed.'''

    # Access the required objects using approved patterns
    object1 = env.scene['Object1'] # Small Block
    object2 = env.scene['Object2'] # Medium Block
    object3 = env.scene['Object3'] # Large Block

    # Access the required robot part(s) using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')

    # Get positions of robot parts and objects (these are world frame positions)
    # All positions are accessed using approved patterns for batched environments.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    object1_pos = object1.data.root_pos_w
    object2_pos = object2.data.root_pos_w
    object3_pos = object3.data.root_pos_w

    # Hardcode object half-dimensions based on the provided object configuration.
    # This is compliant with the rule that object dimensions cannot be accessed dynamically.
    # Object2 (Medium Block) dimensions: x=1m, y=1m, z=0.6m. Half-dimensions: 0.5m, 0.5m, 0.3m.
    object2_half_x = 0.5
    object2_half_y = 0.5
    object2_half_z = 0.3

    # Object1 (Small Block) dimensions: x=1m, y=1m, z=0.3m. Half-dimensions: 0.5m, 0.5m, 0.15m.
    object1_half_x = 0.5
    object1_half_y = 0.5
    object1_half_z = 0.15

    # Object3 (Large Block) dimensions: x=1m, y=1m, z=0.9m. Half-dimensions: 0.5m, 0.5m, 0.45m.
    object3_half_x = 0.5
    object3_half_y = 0.5
    object3_half_z = 0.45

    # Clearance buffer for collision checks (a lenient threshold)
    # This is a reasonable tolerance for collision avoidance.
    clearance_buffer = 0.1

    # Condition 1: Pelvis is adjacent to Object2 (Medium Block)
    # Calculate relative distances between pelvis and Object2 center for X and Y axes.
    # These are relative distances, compliant with the rules.
    dist_x_pelvis_obj2 = torch.abs(pelvis_pos[:, 0] - object2_pos[:, 0])
    dist_y_pelvis_obj2 = torch.abs(pelvis_pos[:, 1] - object2_pos[:, 1])
    pelvis_z = pelvis_pos[:, 2] # Z-position is allowed as an absolute value for height checks.

    # Target distance from Object2 center for adjacency (0.5m from surface + 0.5m half-dimension)
    # This value is derived from object dimensions and skill requirements, not an arbitrary threshold.
    target_dist_from_center = object2_half_x + 0.5 # This is 1.0m

    # Define tolerances for adjacency and alignment. These are reasonable thresholds.
    adjacency_tolerance = 0.1 # +/- 0.1m for the target distance
    alignment_tolerance = 0.2 # +/- 0.2m for alignment along the non-approaching axis

    # Check if adjacent along X and aligned along Y
    # Pelvis X-distance is within the target range AND Pelvis Y-distance is within the block's Y-width plus tolerance
    adjacent_x = (dist_x_pelvis_obj2 > target_dist_from_center - adjacency_tolerance) & \
                 (dist_x_pelvis_obj2 < target_dist_from_center + adjacency_tolerance)
    aligned_y_for_x_approach = (dist_y_pelvis_obj2 < object2_half_y + alignment_tolerance)

    # Check if adjacent along Y and aligned along X
    # Pelvis Y-distance is within the target range AND Pelvis X-distance is within the block's X-length plus tolerance
    adjacent_y = (dist_y_pelvis_obj2 > target_dist_from_center - adjacency_tolerance) & \
                 (dist_y_pelvis_obj2 < target_dist_from_center + adjacency_tolerance)
    aligned_x_for_y_approach = (dist_x_pelvis_obj2 < object2_half_x + alignment_tolerance)

    # Pelvis is adjacent if it satisfies either X-approach or Y-approach conditions.
    # All conditions are handled with tensor operations for batched environments.
    pelvis_adjacent_to_obj2 = (adjacent_x & aligned_y_for_x_approach) | (adjacent_y & aligned_x_for_y_approach)

    # Condition 2: Pelvis is at a stable standing height
    # Pelvis Z-position should be within a reasonable range for standing (0.6m to 0.8m).
    # This uses the absolute Z-position, which is allowed for height checks.
    pelvis_stable_height = (pelvis_z > 0.6) & (pelvis_z < 0.8)

    # Helper function to check for collision between a robot part and an object.
    # This function uses relative distances for collision detection, compliant with rules.
    def is_colliding(part_pos, obj_pos, obj_half_x, obj_half_y, obj_half_z, buffer):
        dist_x = torch.abs(part_pos[:, 0] - obj_pos[:, 0])
        dist_y = torch.abs(part_pos[:, 1] - obj_pos[:, 1])
        dist_z = torch.abs(part_pos[:, 2] - obj_pos[:, 2])
        # Collision occurs if the part is within the object's bounds plus a buffer in all three dimensions
        return (dist_x < obj_half_x + buffer) & \
               (dist_y < obj_half_y + buffer) & \
               (dist_z < obj_half_z + buffer)

    # Condition 3: No collision with Object2 (Medium Block)
    # Check pelvis, left foot, and right foot for collision with Object2.
    # All checks use relative distances and the defined clearance buffer.
    no_collision_obj2_pelvis = ~is_colliding(pelvis_pos, object2_pos, object2_half_x, object2_half_y, object2_half_z, clearance_buffer)
    no_collision_obj2_left_foot = ~is_colliding(left_foot_pos, object2_pos, object2_half_x, object2_half_y, object2_half_z, clearance_buffer)
    no_collision_obj2_right_foot = ~is_colliding(right_foot_pos, object2_pos, object2_half_x, object2_half_y, object2_half_z, clearance_buffer)
    no_collision_obj2 = no_collision_obj2_pelvis & no_collision_obj2_left_foot & no_collision_obj2_right_foot

    # Condition 4: No collision with Object1 (Small Block)
    # Check pelvis, left foot, and right foot for collision with Object1.
    no_collision_obj1_pelvis = ~is_colliding(pelvis_pos, object1_pos, object1_half_x, object1_half_y, object1_half_z, clearance_buffer)
    no_collision_obj1_left_foot = ~is_colliding(left_foot_pos, object1_pos, object1_half_x, object1_half_y, object1_half_z, clearance_buffer)
    no_collision_obj1_right_foot = ~is_colliding(right_foot_pos, object1_pos, object1_half_x, object1_half_y, object1_half_z, clearance_buffer)
    no_collision_obj1 = no_collision_obj1_pelvis & no_collision_obj1_left_foot & no_collision_obj1_right_foot

    # Condition 5: No collision with Object3 (Large Block)
    # Check pelvis, left foot, and right foot for collision with Object3.
    no_collision_obj3_pelvis = ~is_colliding(pelvis_pos, object3_pos, object3_half_x, object3_half_y, object3_half_z, clearance_buffer)
    no_collision_obj3_left_foot = ~is_colliding(left_foot_pos, object3_pos, object3_half_x, object3_half_y, object3_half_z, clearance_buffer)
    no_collision_obj3_right_foot = ~is_colliding(right_foot_pos, object3_pos, object3_half_x, object3_half_y, object3_half_z, clearance_buffer)
    no_collision_obj3 = no_collision_obj3_pelvis & no_collision_obj3_left_foot & no_collision_obj3_right_foot

    # Combine all conditions for overall success.
    # All conditions must be met simultaneously for success.
    overall_condition = pelvis_adjacent_to_obj2 & \
                        pelvis_stable_height & \
                        no_collision_obj2 & \
                        no_collision_obj1 & \
                        no_collision_obj3

    # Check duration and save success states.
    # The skill requires the conditions to be met for a duration of 1.0 seconds.
    # This uses check_success_duration and save_success_state as required.
    success = check_success_duration(env, overall_condition, "walk_to_Medium_Block_for_push", duration=1.0)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_Medium_Block_for_push")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_to_Medium_Block_for_push_success)
