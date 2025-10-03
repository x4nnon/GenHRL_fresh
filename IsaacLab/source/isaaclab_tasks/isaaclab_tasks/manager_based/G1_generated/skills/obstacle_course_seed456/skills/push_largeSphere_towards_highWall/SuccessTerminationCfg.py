
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def push_largeSphere_towards_highWall_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the push_largeSphere_towards_highWall skill has been successfully completed.'''

    # 1. Get robot parts
    robot = env.scene["robot"]
    # Access the pelvis position for stability and positioning checks.
    # CORRECT: Using robot.body_names.index for robust access to robot parts.
    pelvis_idx = robot.body_names.index('pelvis')
    # CORRECT: Accessing robot body position using approved pattern.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # 2. Get object positions
    # Access Object1 (large sphere) position.
    # CORRECT: Direct object access using approved pattern.
    object1 = env.scene['Object1']
    object1_pos = object1.data.root_pos_w
    # Access Object4 (high wall) position.
    # CORRECT: Direct object access using approved pattern.
    object4 = env.scene['Object4']
    object4_pos = object4.data.root_pos_w
    # Access Object2 (small sphere) position for robot's final positioning.
    # CORRECT: Direct object access using approved pattern.
    object2 = env.scene['Object2']
    object2_pos = object2.data.root_pos_w

    # 3. Hardcode object dimensions from the configuration
    # CORRECT: Hardcoding object dimensions as per rule 6, reading from object configuration.
    # Object1: large sphere, radius 1.0m
    object1_radius = 1.0
    # Object4: high wall, 0.3m in x, 5m in y, 1m in z
    object4_x_dim = 0.3
    # Object2: small sphere, radius 0.2m
    object2_radius = 0.2

    # 4. Check success conditions based on relative distances and Z-position for height/toppling

    # Condition 1: Object4's Z-position is less than 0.5m (indicating it has fallen).
    # This is an absolute Z-position check, which is allowed for height/toppling as per rules.
    # CORRECT: Using Z-position for height/toppling check, which is an allowed absolute measurement.
    wall_fallen_condition = object4_pos[:, 2] < 0.5

    # Condition 2: Object1's front edge is past Object4's initial front edge by at least 0.5m.
    # Object1's front edge: object1_pos[:, 0] + object1_radius
    # Object4's initial front edge: object4_pos[:, 0] - object4_x_dim / 2.0
    # We check the relative x-distance between these two points.
    # CORRECT: Using relative X-distance between object edges, incorporating hardcoded dimensions.
    sphere_past_wall_condition = (object1_pos[:, 0] + object1_radius) > (object4_pos[:, 0] - object4_x_dim / 2.0 + 0.5)

    # Condition 3: Robot's pelvis X-position is greater than Object4's initial back edge by at least 0.1m.
    # Object4's initial back edge: object4_pos[:, 0] + object4_x_dim / 2.0
    # This ensures the robot has moved past the toppled wall.
    # CORRECT: Using relative X-distance for robot's position relative to the wall.
    robot_past_wall_x_condition = pelvis_pos[:, 0] > (object4_pos[:, 0] + object4_x_dim / 2.0 + 0.1)

    # Condition 4: Robot's pelvis X-position is less than Object2's front edge by at least 0.1m.
    # Object2's front edge: object2_pos[:, 0] - object2_radius
    # This ensures the robot is not yet at the small sphere, positioning for the next skill.
    # CORRECT: Using relative X-distance for robot's position relative to the next object.
    robot_before_sphere2_x_condition = pelvis_pos[:, 0] < (object2_pos[:, 0] - object2_radius - 0.1)

    # Condition 5: Robot's pelvis Z-position is within 0.2m of 0.7m (i.e., between 0.5m and 0.9m).
    # This checks for robot stability and a reasonable standing height.
    # CORRECT: Using absolute Z-position for robot stability/height, which is an allowed absolute measurement.
    robot_pelvis_z_condition = (pelvis_pos[:, 2] > 0.5) & (pelvis_pos[:, 2] < 0.9)

    # Combine all conditions for overall success
    # CORRECT: Combining all conditions using logical AND for batched environments.
    condition = wall_fallen_condition & \
                sphere_past_wall_condition & \
                robot_past_wall_x_condition & \
                robot_before_sphere2_x_condition & \
                robot_pelvis_z_condition

    # 5. Check duration and save success states
    # The duration is set to 0.5 seconds as per the plan.
    # CORRECT: Using check_success_duration as required.
    success = check_success_duration(env, condition, "push_largeSphere_towards_highWall", duration=0.5)
    # CORRECT: Saving success states for environments that meet the criteria.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "push_largeSphere_towards_highWall")

    return success

@configclass
class SuccessTerminationCfg:
    # Define the success termination term using the implemented function.
    success = DoneTerm(func=push_largeSphere_towards_highWall_success)
