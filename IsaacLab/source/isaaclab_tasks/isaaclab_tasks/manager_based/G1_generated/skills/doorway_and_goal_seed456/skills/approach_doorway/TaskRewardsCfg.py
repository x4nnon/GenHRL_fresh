from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.reward_normalizer import get_normalizer, RewardStats # this automatically sets up the RewNormalizer instance.
from genhrl.generation.objects import get_object_volume
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv # Corrected: Added ManagerBasedRLEnv import
import torch

from isaaclab.envs import mdp
# Import custom MDP functions from genhrl
import genhrl.generation.mdp.rewards as custom_rewards
import genhrl.generation.mdp.terminations as custom_terminations  
import genhrl.generation.mdp.observations as custom_observations
import genhrl.generation.mdp.events as custom_events
import genhrl.generation.mdp.curriculums as custom_curriculums

# Helper function to convert quaternion to a rotated vector
# This function is not part of Isaac Lab's mdp, so it's defined here.
def quat_to_vec(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotates a vector by a quaternion.
    Args:
        quat (torch.Tensor): Quaternion (w, x, y, z) or (x, y, z, w).
        vec (torch.Tensor): Vector to rotate.
    Returns:
        torch.Tensor: Rotated vector.
    """
    # Ensure quaternion is (w, x, y, z) for standard multiplication
    # Assuming input quat is (x, y, z, w) as is common in Isaac Lab data
    # Convert to (w, x, y, z) for standard quaternion multiplication
    w = quat[..., 3]
    x = quat[..., 0]
    y = quat[..., 1]
    z = quat[..., 2]
    
    # Create a quaternion from the vector (0, vec_x, vec_y, vec_z)
    vec_quat = torch.cat([torch.zeros_like(w).unsqueeze(-1), vec], dim=-1)

    # Conjugate of the quaternion
    quat_conj = torch.cat([w.unsqueeze(-1), -x.unsqueeze(-1), -y.unsqueeze(-1), -z.unsqueeze(-1)], dim=-1)

    # Quaternion multiplication: q * v_quat * q_conj
    # First, q * v_quat
    # (w1*w2 - x1*x2 - y1*y2 - z1*z2,
    #  w1*x2 + x1*w2 + y1*z2 - z1*y2,
    #  w1*y2 - x1*z2 + y1*w2 + z1*x2,
    #  w1*z2 + x1*y2 - y1*x2 + z1*w2)
    
    # For q * v_quat, w2=0, x2=vec_x, y2=vec_y, z2=vec_z
    q_v_w = -x * vec[..., 0] - y * vec[..., 1] - z * vec[..., 2]
    q_v_x = w * vec[..., 0] + y * vec[..., 2] - z * vec[..., 1]
    q_v_y = w * vec[..., 1] - x * vec[..., 2] + z * vec[..., 0]
    q_v_z = w * vec[..., 2] + x * vec[..., 1] - y * vec[..., 0]
    
    q_v = torch.stack([q_v_w, q_v_x, q_v_y, q_v_z], dim=-1)

    # Now, (q * v_quat) * q_conj
    # (w1*w2 - x1*x2 - y1*y2 - z1*z2, ...)
    # Here, q1 = q_v, q2 = quat_conj
    
    w1, x1, y1, z1 = q_v[..., 0], q_v[..., 1], q_v[..., 2], q_v[..., 3]
    w2, x2, y2, z2 = quat_conj[..., 0], quat_conj[..., 1], quat_conj[..., 2], quat_conj[..., 3]

    final_w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    final_x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    final_y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    final_z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return torch.stack([final_x, final_y, final_z], dim=-1)


def approach_doorway_primary_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_doorway_primary_reward") -> torch.Tensor:
    '''
    This reward guides the robot to approach the doorway, center itself, and align its orientation.
    It combines three components: approaching the target Y-position of the doorway, centering in the X-axis,
    and aligning its orientation. The reward is designed to be continuous and provide a clear gradient towards the goal.
    The target Y-position is the front face of the walls.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects - USE ONLY Object1, Object2, Object3, Object4, Object5
    # Object1: Heavy Cube (Wall 1) forming the left side of the doorway.
    # Object2: Heavy Cube (Wall 2) forming the right side of the doorway.
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']

    # Access the required robot part(s)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing robot part index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing robot part position using approved pattern
    pelvis_pos_x = pelvis_pos[:, 0] # CORRECT: Separating x component
    pelvis_pos_y = pelvis_pos[:, 1] # CORRECT: Separating y component

    pelvis_quat = robot.data.body_quat_w[:, pelvis_idx] # CORRECT: Accessing robot part quaternion
    # CORRECT: Using helper function to get forward vector, aligning with global Y-axis [0, 1, 0]
    pelvis_forward_vec = quat_to_vec(pelvis_quat, torch.tensor([0.0, 1.0, 0.0], device=robot.device).repeat(env.num_envs, 1))

    # Object dimensions (from description: x of 0.5m for walls)
    # CORRECT: Hardcoding object dimension from description, not accessing from object directly
    wall_x_size = 0.5 # Depth of the wall in X-axis (this is the dimension along the object's local X, which is global Y for the doorway depth)

    # Calculate target Y-position (front edge of the doorway)
    # Assuming Object1 and Object2 are aligned in Y, and their root_pos_w is their center.
    # The front face would be at object_y_pos - wall_x_size / 2
    # CORRECT: Relative distance calculation based on object position and known dimension
    target_doorway_y = object1.data.root_pos_w[:, 1] - wall_x_size / 2.0

    # Calculate target X-position (center of the doorway)
    # CORRECT: Relative distance calculation based on object positions
    target_doorway_x = (object1.data.root_pos_w[:, 0] + object2.data.root_pos_w[:, 0]) / 2.0

    # 1. Approach Y-position reward
    # Penalize absolute difference from target Y. Reward is max when difference is 0.
    # CORRECT: Continuous reward based on absolute relative distance
    reward_approach_y = -torch.abs(pelvis_pos_y - target_doorway_y)

    # 2. Centering X-position reward
    # Penalize absolute difference from target X. Reward is max when difference is 0.
    # CORRECT: Continuous reward based on absolute relative distance
    reward_center_x = -torch.abs(pelvis_pos_x - target_doorway_x)

    # 3. Orientation reward (aligning with global Y-axis)
    # Dot product of pelvis forward vector with global Y-axis [0, 1, 0]
    # Max reward when aligned (dot product = 1), min when opposite (-1)
    # CORRECT: Continuous reward based on vector alignment (dot product)
    reward_orientation = torch.sum(pelvis_forward_vec * torch.tensor([0.0, 1.0, 0.0], device=robot.device), dim=-1)

    # Combine rewards with specified weights
    reward = reward_approach_y * 0.5 + reward_center_x * 0.3 + reward_orientation * 0.2

    # CORRECT: Complete normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def approach_doorway_pelvis_height_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_doorway_pelvis_height_reward") -> torch.Tensor:
    '''
    This reward encourages the robot to maintain a stable, upright posture by keeping its pelvis at a desired height (around 0.7m).
    This is crucial for stability and preparing for the next skill.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot part(s)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing robot part index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing robot part position using approved pattern
    pelvis_pos_z = pelvis_pos[:, 2] # CORRECT: Separating z component

    # Target pelvis height for stability
    # CORRECT: Hardcoding target height as per description, this is an absolute Z position, which is allowed for height.
    target_pelvis_z = 0.7

    # Reward for maintaining pelvis height
    # Penalize absolute difference from target Z. Reward is max when difference is 0.
    # CORRECT: Continuous reward based on absolute relative distance (from target height)
    reward = -torch.abs(pelvis_pos_z - target_pelvis_z)

    # CORRECT: Complete normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def approach_doorway_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_doorway_collision_avoidance_reward") -> torch.Tensor:
    '''
    This reward encourages collision avoidance with the doorway walls (Object1 and Object2).
    It penalizes the robot if its pelvis gets too close to the walls in the X-dimension,
    ensuring it stays within the doorway gap. It also penalizes if the robot goes too far past the doorway in Y.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects - USE ONLY Object1, Object2, Object3, Object4, Object5
    # Object1: Heavy Cube (Wall 1) forming the left side of the doorway.
    # Object2: Heavy Cube (Wall 2) forming the right side of the doorway.
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']

    # Access the required robot part(s)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing robot part index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing robot part position using approved pattern
    pelvis_pos_x = pelvis_pos[:, 0] # CORRECT: Separating x component
    pelvis_pos_y = pelvis_pos[:, 1] # CORRECT: Separating y component

    # Object dimensions (from description: x of 0.5m for walls)
    # CORRECT: Hardcoding object dimension from description
    wall_x_size = 0.5 # Depth of the wall in X-axis (this is the dimension along the object's local X, which is global Y for the doorway depth)
    doorway_gap_width = 0.5 # From description: "doorway gap of 0.5m"

    # Calculate the X-boundaries of the doorway based on wall positions and their X-size
    # The doorway is centered between Object1 and Object2.
    # CORRECT: Relative distance calculation based on object positions and known dimension
    doorway_center_x = (object1.data.root_pos_w[:, 0] + object2.data.root_pos_w[:, 0]) / 2.0
    doorway_half_width = doorway_gap_width / 2.0

    # Define safe zone boundaries for the pelvis in X
    safe_x_min = doorway_center_x - doorway_half_width
    safe_x_max = doorway_center_x + doorway_half_width

    # Penalize if pelvis_x is outside the safe zone
    # CORRECT: Continuous penalty based on relative distance from safe boundaries
    reward_collision_x = torch.where(
        (pelvis_pos_x < safe_x_min),
        -(safe_x_min - pelvis_pos_x), # Penalize for being too far left
        torch.zeros_like(pelvis_pos_x)
    )
    reward_collision_x += torch.where(
        (pelvis_pos_x > safe_x_max),
        -(pelvis_pos_x - safe_x_max), # Penalize for being too far right
        torch.zeros_like(pelvis_pos_x)
    )

    # Also penalize if the robot goes too far into the walls in Y (past the front face)
    # The front face of the walls is at target_doorway_y (calculated in primary reward)
    # CORRECT: Relative distance calculation based on object position and known dimension
    target_doorway_y = object1.data.root_pos_w[:, 1] - wall_x_size / 2.0 # Assuming wall_x_size is depth in Y

    # A small buffer (e.g., 0.1m) might be acceptable to be "in" the doorway, but not too far.
    # This ensures the robot stops *at* the doorway, not through it.
    # CORRECT: Hardcoding buffer value, which is an acceptable threshold for a small clearance.
    buffer_y = 0.1
    # Penalize if pelvis_y goes significantly past the front of the doorway
    # CORRECT: Continuous penalty based on relative distance from target Y with buffer
    reward_collision_y = torch.where(
        (pelvis_pos_y > target_doorway_y + buffer_y),
        -(pelvis_pos_y - (target_doorway_y + buffer_y)), # Penalize for going too far past the doorway
        torch.zeros_like(pelvis_pos_y)
    )

    reward = reward_collision_x + reward_collision_y

    # CORRECT: Complete normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Primary reward for approaching, centering, and orienting at the doorway
    ApproachDoorwayPrimaryReward = RewTerm(func=approach_doorway_primary_reward, weight=1.0,
                                           params={"normalise": True, "normaliser_name": "approach_doorway_primary_reward"})

    # Shaping reward for maintaining stable pelvis height
    PelvisHeightReward = RewTerm(func=approach_doorway_pelvis_height_reward, weight=0.4,
                                 params={"normalise": True, "normaliser_name": "approach_doorway_pelvis_height_reward"})

    # Shaping reward for avoiding collisions with doorway walls and not overshooting in Y
    CollisionAvoidanceReward = RewTerm(func=approach_doorway_collision_avoidance_reward, weight=0.6,
                                       params={"normalise": True, "normaliser_name": "approach_doorway_collision_avoidance_reward"})