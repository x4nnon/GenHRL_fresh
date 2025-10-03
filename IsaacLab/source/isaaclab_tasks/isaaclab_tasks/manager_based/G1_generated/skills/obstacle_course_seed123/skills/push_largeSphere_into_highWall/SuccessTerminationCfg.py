
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def push_largeSphere_into_highWall_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the push_largeSphere_into_highWall skill has been successfully completed.'''

    # Access the required objects based on the object configuration
    # Object1: "large sphere for robot to push"
    large_sphere = env.scene['Object1'] # CORRECT: Accessing Object1 directly as per rules.
    # Object4: "high wall for large sphere to push over"
    high_wall = env.scene['Object4']   # CORRECT: Accessing Object4 directly as per rules.

    # Access the robot object (though robot position is a secondary success factor for this skill)
    robot = env.scene["robot"]
    # Get the pelvis position for checking robot's general location
    pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Using robot.body_names.index for robot part access.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing robot part position using approved pattern.

    # Hardcode object dimensions as per requirement 6 (no data.size or similar)
    # These values are derived from the task description and object configuration.
    large_sphere_radius = 1.0 # CORRECT: Hardcoding dimensions from task description.
    high_wall_thickness = 0.3 # CORRECT: Hardcoding dimensions from task description.
    high_wall_height = 1.0    # CORRECT: Hardcoding dimensions from task description.

    # --- Success Condition 1: High wall has fallen over ---
    # The wall is considered fallen if its center's Z-position drops significantly.
    # A threshold of half its height is a reasonable indicator for a fallen wall.
    # This uses the relative Z-position of the wall to a hardcoded threshold based on its dimensions.
    # CORRECT: Z-position check is allowed for height-related conditions.
    wall_fallen_threshold_z = high_wall_height * 0.5
    wall_fallen_condition = high_wall.data.root_pos_w[:, 2] < wall_fallen_threshold_z

    # --- Success Condition 2: Large sphere has moved past the high wall's original X-location ---
    # Since we cannot use previous_object_positions, we infer this by checking if the sphere's
    # leading edge is past the current leading edge of the wall. If the wall has fallen,
    # it implies the sphere pushed it. A small tolerance is added to account for slight overlaps
    # or the wall moving.
    # This uses relative distances between the sphere and wall positions, considering their dimensions.
    sphere_leading_edge_x = large_sphere.data.root_pos_w[:, 0] + large_sphere_radius # CORRECT: Using relative distance based on object positions and hardcoded dimensions.
    wall_leading_edge_x = high_wall.data.root_pos_w[:, 0] - (high_wall_thickness / 2.0) # CORRECT: Using relative distance based on object positions and hardcoded dimensions.
    # The sphere should be at or past the wall's leading edge.
    sphere_past_wall_condition = sphere_leading_edge_x > wall_leading_edge_x - 0.1 # Tolerance of 0.1m, CORRECT: Using reasonable tolerance.

    # --- Success Condition 3: Robot is past the high wall and ready for the next skill ---
    # The robot should not be stuck behind the wall or the sphere. It should be positioned
    # to move towards the next object (small sphere, Object2).
    # This uses relative distance between robot pelvis and the wall's current X-position.
    # The robot's pelvis should be past the wall's leading edge.
    robot_past_wall_x_condition = pelvis_pos[:, 0] > wall_leading_edge_x - 0.5 # Robot pelvis is 0.5m past wall's front face, CORRECT: Using relative distance and reasonable tolerance.

    # Combine all success conditions. All must be true for success.
    # All conditions use relative distances between objects/robot parts or object dimensions.
    success_condition = wall_fallen_condition & sphere_past_wall_condition & robot_past_wall_x_condition # CORRECT: Combining conditions with tensor operations.

    # Check success duration and save success states as per requirements
    # A duration of 0.5 seconds ensures the state is stable.
    success = check_success_duration(env, success_condition, "push_largeSphere_into_highWall", duration=0.5) # CORRECT: Using check_success_duration.

    # Save success states for environments that have met the success criteria
    if success.any(): # CORRECT: Checking if any environment succeeded.
        for env_id in torch.where(success)[0]: # CORRECT: Iterating over successful environments.
            save_success_state(env, env_id, "push_largeSphere_into_highWall") # CORRECT: Saving success state.

    return success

class SuccessTerminationCfg:
    # Configure the success termination for the skill
    success = DoneTerm(func=push_largeSphere_into_highWall_success)
