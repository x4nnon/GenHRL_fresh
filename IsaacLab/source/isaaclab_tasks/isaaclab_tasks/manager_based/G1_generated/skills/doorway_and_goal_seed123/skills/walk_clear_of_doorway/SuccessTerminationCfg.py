
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_clear_of_doorway_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_clear_of_doorway skill has been successfully completed.'''

    # Access the robot object
    robot = env.scene["robot"]

    # Access the required objects based on the object configuration
    # Object1: Heavy Cube (Wall 1) forming the left wall of the doorway
    object1 = env.scene['Object1']
    # Object2: Heavy Cube (Wall 2) forming the right wall of the doorway
    object2 = env.scene['Object2']
    # Object3: Small Block for the robot to walk to
    object3 = env.scene['Object3']

    # Access the pelvis position, which is used as the primary reference point for the robot's body
    # This follows the approved pattern for accessing robot body parts.
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Hardcoded dimensions from the task description for the walls.
    # These values are obtained from the environment setup description, not from object attributes,
    # adhering to the rule that object sizes must be hardcoded from the config.
    wall_length_y = 5.0 # Length of the wall along the y-axis
    wall_thickness_x = 0.5 # Thickness of the wall along the x-axis

    # Calculate the y-coordinate of the doorway exit.
    # This is relative to Object1's y-center, ensuring it adapts to scene variations,
    # and uses the approved object position access pattern.
    doorway_y_exit = object1.data.root_pos_w[:, 1] + (wall_length_y / 2.0)

    # Calculate the inner x-edges of the walls and the center of the doorway gap.
    # This ensures the x-centering condition is relative to the actual doorway position,
    # and uses the approved object position access pattern.
    wall1_inner_x = object1.data.root_pos_w[:, 0] + (wall_thickness_x / 2.0)
    wall2_inner_x = object2.data.root_pos_w[:, 0] - (wall_thickness_x / 2.0)
    doorway_x_center = (wall1_inner_x + wall2_inner_x) / 2.0

    # Calculate the target y-position for the robot, which is before Object3.
    # This is relative to Object3's position, ensuring the robot doesn't overshoot,
    # and uses the approved object position access pattern.
    # The 0.3m buffer is a reasonable tolerance.
    target_y_before_object3 = object3.data.root_pos_w[:, 1] - 0.3 # 0.3m buffer before Object3

    # Condition 1: Pelvis is past the doorway exit.
    # This uses a relative distance check between the pelvis and the calculated doorway exit.
    # A small buffer (0.1m) is added to ensure the robot is clearly past the doorway,
    # which is a reasonable tolerance (0.05-0.1m).
    condition_past_doorway_y = pelvis_pos_y > (doorway_y_exit + 0.1)

    # Condition 2: Pelvis is not too far past the doorway, within range for Object3.
    # This uses a relative distance check between the pelvis and the target y-position before Object3.
    # This ensures the robot is in the correct area for the next objective.
    condition_before_object3_y = pelvis_pos_y < target_y_before_object3

    # Condition 3: Pelvis is centered in x within the doorway's x-range.
    # This uses a relative distance check between the pelvis's x-position and the doorway's x-center.
    # A tolerance of 0.2m is used for centering, which is a reasonable threshold.
    condition_centered_x = torch.abs(pelvis_pos_x - doorway_x_center) < 0.2

    # Condition 4: Pelvis is at a stable standing height.
    # The z-component is the only absolute position allowed for success criteria, used here for stability.
    # A target height of 0.7m and a tolerance of 0.1m are used, which are reasonable values for a standing robot.
    pelvis_target_z = 0.7
    condition_stable_z = torch.abs(pelvis_pos_z - pelvis_target_z) < 0.1

    # Combine all conditions for overall success. All conditions must be met.
    # All operations work with batched environments.
    overall_success_condition = condition_past_doorway_y & \
                                condition_before_object3_y & \
                                condition_centered_x & \
                                condition_stable_z

    # Check success duration and save success states.
    # A duration of 0.5 seconds is used to ensure the robot maintains the successful state for a short period.
    # This adheres to the requirement to use check_success_duration.
    success = check_success_duration(env, overall_success_condition, "walk_clear_of_doorway", duration=0.5)

    # Save success states for environments that have successfully completed the skill.
    # This adheres to the requirement to use save_success_state.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_clear_of_doorway")

    return success

@configclass
class SuccessTerminationCfg:
    # Define the success termination term using the implemented success function.
    success = DoneTerm(func=walk_clear_of_doorway_success)
