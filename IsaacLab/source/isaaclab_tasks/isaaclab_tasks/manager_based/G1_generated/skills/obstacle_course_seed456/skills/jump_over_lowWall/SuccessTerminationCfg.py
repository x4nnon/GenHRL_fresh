
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def jump_over_lowWall_success(env) -> torch.Tensor:
    '''Determine if the jump_over_lowWall skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Access the robot object using the approved pattern.
    robot = env.scene["robot"]

    # Get the pelvis position, which is crucial for determining the robot's overall position and stability.
    # This uses the approved pattern for accessing robot body part positions.
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]

    # Access the low wall (Object3) and large sphere (Object1) objects.
    # These are accessed directly using their scene names as per requirements.
    low_wall = env.scene['Object3']
    large_sphere = env.scene['Object1']

    # Get the root positions of the low wall and large sphere.
    # This uses the approved pattern for accessing object positions.
    low_wall_pos = low_wall.data.root_pos_w # Shape: [num_envs, 3]
    large_sphere_pos = large_sphere.data.root_pos_w # Shape: [num_envs, 3]

    # Hardcode object dimensions as per the task description and "THERE IS NO way to access the SIZE of an object" rule.
    # Object dimensions cannot be accessed from the RigidObject, so they must be hardcoded from the object configuration.
    low_wall_thickness = 0.3 # 0.3m in x-axis for the low wall, from task description
    low_wall_height = 0.5 # 0.5m in z-axis for the low wall, from task description

    # Calculate the x-position of the back edge of the low wall.
    # This is a relative calculation based on the wall's root position and its thickness.
    low_wall_x = low_wall_pos[:, 0]
    low_wall_back_x = low_wall_x + (low_wall_thickness / 2.0)

    # Define the target stable standing height for the robot's pelvis.
    # This value is taken from the success criteria plan and reward functions, and is an allowed absolute Z position.
    target_pelvis_z = 0.7 # meters

    # Condition 1: Robot's pelvis is past the low wall in the x-axis.
    # This ensures the robot has successfully cleared the wall.
    # A buffer of 0.1m is added to ensure full clearance beyond the wall's back edge, as per plan.
    # This uses relative distance between pelvis x and wall's back edge x.
    condition_past_wall_x = pelvis_pos[:, 0] > (low_wall_back_x + 0.1)

    # Condition 2: Robot's pelvis is within a reasonable distance of the large sphere in the x-axis.
    # This ensures the robot lands in a good position for the next skill (pushing the large sphere).
    # A buffer of 0.5m is used to ensure the robot is not past the sphere, as per plan.
    # This uses relative distance between pelvis x and large sphere x.
    large_sphere_x = large_sphere_pos[:, 0]
    condition_before_sphere_x = pelvis_pos[:, 0] < (large_sphere_x - 0.5)

    # Condition 3: Robot's pelvis is at a stable standing height in the z-axis.
    # This checks for stability after landing.
    # An absolute difference threshold of 0.15m allows for slight variations in landing height, as per plan.
    # This uses relative distance between pelvis z and the target pelvis z.
    condition_stable_z = torch.abs(pelvis_pos[:, 2] - target_pelvis_z) < 0.15

    # Condition 4: Robot's pelvis is centered in the y-axis.
    # This ensures the robot maintains its alignment with the obstacle course.
    # An absolute difference threshold of 0.2m allows for minor lateral deviations, as per plan.
    # This uses relative distance between pelvis y and the global y-axis center (0.0).
    condition_centered_y = torch.abs(pelvis_pos[:, 1] - 0.0) < 0.2

    # Combine all conditions. All conditions must be true for success.
    # All operations are performed on tensors, handling batched environments correctly.
    all_conditions_met = condition_past_wall_x & condition_before_sphere_x & condition_stable_z & condition_centered_y

    # Check success duration and save success states.
    # The duration is set to 0.5 seconds as specified in the plan to ensure stability.
    # This follows the "CHECK SUCCESS DURATION" and "SAVE SUCCESS STATES" rules.
    success = check_success_duration(env, all_conditions_met, "jump_over_lowWall", duration=0.5)

    # Save success states for environments that have met the success criteria for the required duration.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "jump_over_lowWall")

    return success

class SuccessTerminationCfg:
    # Register the success function as a termination condition.
    success = DoneTerm(func=jump_over_lowWall_success)
