"""Object generation helpers.

This module is imported both during **task generation** (pure JSON/text) and
later during **runtime simulation** inside Isaac Sim.  Only the latter actually
requires the heavy IsaacLab / Omniverse dependencies.  To ensure that the
light-weight generation pipeline can run on machines without those libraries we
guard the imports below.
"""

from dataclasses import MISSING
from typing import Dict, List, Any
import os
import json

import isaaclab.sim as sim_utils

from isaaclab.assets import RigidObjectCfg

from . import mdp
from .mdp import *
import isaacsim.core.utils.prims as prim_utils
import random

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab_assets import G1_MINIMAL_CFG, G1_CFG   # isort: skip

# Define default values for object properties
# These are sensible defaults that can be overridden by the LLM
DEFAULT_SIZES = {
    "sphere": 0.25,  # Default radius
    "cube": (0.5, 0.5, 0.5),  # Default size
    "cylinder": (0.25, 0.5),  # Default radius, height
    "cone": (0.25, 0.5),  # Default radius, height
    "capsule": (0.2, 0.4)  # Default radius, height
}

def load_object_config(json_path=None):
        """Load object configuration from JSON file."""
        if json_path is None:
            json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'object_config.json')
        
        if os.path.exists(json_path):
            print(f"\n\n Loading object config from: {json_path} \n\n")
            with open(json_path, 'r') as f:
                config = json.load(f)
                # Filter out comment fields (those starting with '_')
                return {k: v for k, v in config.items() if not k.startswith('_')}
        else:
            print(f"\n\n WARNING! Object config file not found at: {json_path} \n\n")
        return {}  # Return empty dict if file doesn't exist


def generate_object_configs(
    # Sphere parameters
    num_spheres: int = 0,
    mass_spheres: list[float] = [],
    position_spheres: list[list[float]] = [],
    rotation_spheres: list[list[float]] = [],
    radius_spheres: list[float] = [],
    
    # Cube parameters
    num_cubes: int = 0,
    mass_cubes: list[float] = [],
    position_cubes: list[list[float]] = [],
    rotation_cubes: list[list[float]] = [],
    size_cubes: list[list[float]] = [],
    
    # Cylinder parameters
    num_cylinders: int = 0,
    mass_cylinders: list[float] = [],
    position_cylinders: list[list[float]] = [],
    rotation_cylinders: list[list[float]] = [],
    radius_cylinders: list[float] = [],
    height_cylinders: list[float] = [],
    
    # Cone parameters
    num_cones: int = 0,
    mass_cones: list[float] = [],
    position_cones: list[list[float]] = [],
    rotation_cones: list[list[float]] = [],
    radius_cones: list[float] = [],
    height_cones: list[float] = [],
    
    # Capsule parameters
    num_capsules: int = 0,
    mass_capsules: list[float] = [],
    position_capsules: list[list[float]] = [],
    rotation_capsules: list[list[float]] = [],
    radius_capsules: list[float] = [],
    height_capsules: list[float] = [],
    
    # Comment parameters (not used for object creation but part of the interface)
    _sphere_comments: list[str] = [],
    _cube_comments: list[str] = [],
    _cylinder_comments: list[str] = [],
    _cone_comments: list[str] = [],
    _capsule_comments: list[str] = [],
    _comment: str = "",
    object_order: list[str] = [],
    **kwargs  # Catch any additional parameters
) -> tuple[RigidObjectCfg, ...]:
    """Generate configurations for exactly 5 objects."""
    total_objects = num_spheres + num_cubes + num_cylinders + num_cones + num_capsules
    if total_objects > 5:
        raise ValueError("Total number of objects cannot exceed 5")

    configs = []
    object_idx = 1

    # Helper to create object configs
    def create_object_cfg(obj_type: str, mass: float, position: list[float], rotation: list[float], obj_idx: int, **object_params) -> RigidObjectCfg:
        prim_path = f"{{ENV_REGEX_NS}}/Object{obj_idx}"
        
        # Create spawn configuration based on object type
        if obj_type == "sphere":
            spawn_cfg = sim_utils.SphereCfg(
                radius=object_params.get("radius", DEFAULT_SIZES["sphere"]),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(),
            )
        elif obj_type == "cube":
            size = object_params.get("size", DEFAULT_SIZES["cube"])
            if isinstance(size, (list, tuple)) and len(size) == 3:
                size = tuple(size)
            else:
                size = DEFAULT_SIZES["cube"]
            spawn_cfg = sim_utils.CuboidCfg(
                size=size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(),
            )
        elif obj_type == "cylinder":
            spawn_cfg = sim_utils.CylinderCfg(
                radius=object_params.get("radius", DEFAULT_SIZES["cylinder"][0]),
                height=object_params.get("height", DEFAULT_SIZES["cylinder"][1]),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(),
            )
        elif obj_type == "cone":
            spawn_cfg = sim_utils.ConeCfg(
                radius=object_params.get("radius", DEFAULT_SIZES["cone"][0]),
                height=object_params.get("height", DEFAULT_SIZES["cone"][1]),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(),
            )
        elif obj_type == "capsule":
            spawn_cfg = sim_utils.CapsuleCfg(
                radius=object_params.get("radius", DEFAULT_SIZES["capsule"][0]),
                height=object_params.get("height", DEFAULT_SIZES["capsule"][1]),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(),
            )
        else:
            # Default to cube for unknown types
            spawn_cfg = sim_utils.CuboidCfg(
                size=DEFAULT_SIZES["cube"],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(),
            )

        # Ensure position and rotation are properly formatted
        if isinstance(position, list) and len(position) == 3:
            pos = tuple(position)
        else:
            pos = (0.0, 0.0, 0.0)
            
        if isinstance(rotation, list) and len(rotation) == 4:
            rot = tuple(rotation)
        else:
            rot = (1.0, 0.0, 0.0, 0.0)  # Default quaternion (no rotation)

        return RigidObjectCfg(
            prim_path=prim_path,
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
        )

    # Process spheres
    for i in range(num_spheres):
        mass = mass_spheres[i] if i < len(mass_spheres) else 1.0
        position = position_spheres[i] if i < len(position_spheres) else [0.0, 0.0, 0.0]
        rotation = rotation_spheres[i] if i < len(rotation_spheres) else [1.0, 0.0, 0.0, 0.0]
        radius = radius_spheres[i] if i < len(radius_spheres) else DEFAULT_SIZES["sphere"]
        
        config = create_object_cfg("sphere", mass, position, rotation, object_idx, radius=radius)
        configs.append(config)
        object_idx += 1

    # Process cubes
    for i in range(num_cubes):
        mass = mass_cubes[i] if i < len(mass_cubes) else 1.0
        position = position_cubes[i] if i < len(position_cubes) else [0.0, 0.0, 0.0]
        rotation = rotation_cubes[i] if i < len(rotation_cubes) else [1.0, 0.0, 0.0, 0.0]
        size = size_cubes[i] if i < len(size_cubes) else DEFAULT_SIZES["cube"]
        
        config = create_object_cfg("cube", mass, position, rotation, object_idx, size=size)
        configs.append(config)
        object_idx += 1

    # Process cylinders
    for i in range(num_cylinders):
        mass = mass_cylinders[i] if i < len(mass_cylinders) else 1.0
        position = position_cylinders[i] if i < len(position_cylinders) else [0.0, 0.0, 0.0]
        rotation = rotation_cylinders[i] if i < len(rotation_cylinders) else [1.0, 0.0, 0.0, 0.0]
        radius = radius_cylinders[i] if i < len(radius_cylinders) else DEFAULT_SIZES["cylinder"][0]
        height = height_cylinders[i] if i < len(height_cylinders) else DEFAULT_SIZES["cylinder"][1]
        
        config = create_object_cfg("cylinder", mass, position, rotation, object_idx, radius=radius, height=height)
        configs.append(config)
        object_idx += 1

    # Process cones
    for i in range(num_cones):
        mass = mass_cones[i] if i < len(mass_cones) else 1.0
        position = position_cones[i] if i < len(position_cones) else [0.0, 0.0, 0.0]
        rotation = rotation_cones[i] if i < len(rotation_cones) else [1.0, 0.0, 0.0, 0.0]
        radius = radius_cones[i] if i < len(radius_cones) else DEFAULT_SIZES["cone"][0]
        height = height_cones[i] if i < len(height_cones) else DEFAULT_SIZES["cone"][1]
        
        config = create_object_cfg("cone", mass, position, rotation, object_idx, radius=radius, height=height)
        configs.append(config)
        object_idx += 1

    # Process capsules
    for i in range(num_capsules):
        mass = mass_capsules[i] if i < len(mass_capsules) else 1.0
        position = position_capsules[i] if i < len(position_capsules) else [0.0, 0.0, 0.0]
        rotation = rotation_capsules[i] if i < len(rotation_capsules) else [1.0, 0.0, 0.0, 0.0]
        radius = radius_capsules[i] if i < len(radius_capsules) else DEFAULT_SIZES["capsule"][0]
        height = height_capsules[i] if i < len(height_capsules) else DEFAULT_SIZES["capsule"][1]
        
        config = create_object_cfg("capsule", mass, position, rotation, object_idx, radius=radius, height=height)
        configs.append(config)
        object_idx += 1

    # Fill remaining slots with placeholder objects (tiny cubes placed far below ground)
    while len(configs) < 5:
        prim_path = f"{{ENV_REGEX_NS}}/Object{object_idx}"
        # Create a minimal cube placeholder placed far below ground so it doesn't interfere
        placeholder_config = RigidObjectCfg(
            prim_path=prim_path,
            spawn=sim_utils.CuboidCfg(
                size=(0.01, 0.01, 0.01),  # Very small cube
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),  # Place far below ground
        )
        configs.append(placeholder_config)
        object_idx += 1

    return tuple(configs)

def generate_config_from_llm(objects_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a JSON-like dictionary for scene object configurations based on a list of objects
    provided by the LLM. This now contains the logic from the previously broken function.
    """
    config = {
        "num_spheres": 0, "num_cubes": 0, "num_cylinders": 0, "num_cones": 0, "num_capsules": 0,
        "position_spheres": [], "radius_spheres": [], "_sphere_comments": [],
        "position_cubes": [], "size_cubes": [], "_cube_comments": [],
        "position_cylinders": [], "radius_cylinders": [], "height_cylinders": [], "_cylinder_comments": [],
        "position_cones": [], "radius_cones": [], "height_cones": [], "_cone_comments": [],
        "position_capsules": [], "radius_capsules": [], "height_capsules": [], "_capsule_comments": [],
        "object_order": []
    }

    object_order_list = []
    for obj in objects_list:
        if "object_order" in obj:
            object_order_list = obj["object_order"]
            break

    config["object_order"] = object_order_list
    objects_to_process = [obj for obj in objects_list if "object_order" not in obj]

    if not object_order_list: # Fallback if no order is specified
        object_order_list = [obj.get("type", "cube") for obj in objects_to_process]

    for i, object_type in enumerate(object_order_list):
        if i < len(objects_to_process):
            obj_data = objects_to_process[i]
            comment = obj_data.get("comment", f"a {object_type}")
            position = obj_data.get("position", [0, 0, 0])
            
            if object_type == "sphere":
                config["num_spheres"] += 1
                config["position_spheres"].extend(position)
                config["radius_spheres"].append(obj_data.get("radius", 0.25))
                config["_sphere_comments"].append(comment)
            elif object_type == "cube":
                config["num_cubes"] += 1
                config["position_cubes"].extend(position)
                config["size_cubes"].extend(obj_data.get("size", [0.5, 0.5, 0.5]))
                config["_cube_comments"].append(comment)
            # ... and so on for other object types
            elif object_type == "cylinder":
                config["num_cylinders"] += 1
                config["position_cylinders"].extend(position)
                radius, height = obj_data.get("size", [0.25, 0.5])
                config["radius_cylinders"].append(radius)
                config["height_cylinders"].append(height)
                config["_cylinder_comments"].append(comment)
            elif object_type == "cone":
                config["num_cones"] += 1
                config["position_cones"].extend(position)
                radius, height = obj_data.get("size", [0.25, 0.5])
                config["radius_cones"].append(radius)
                config["height_cones"].append(height)
                config["_cone_comments"].append(comment)
            elif object_type == "capsule":
                config["num_capsules"] += 1
                config["position_capsules"].extend(position)
                radius, height = obj_data.get("size", [0.2, 0.4])
                config["radius_capsules"].append(radius)
                config["height_capsules"].append(height)
                config["_capsule_comments"].append(comment)

    return config


def get_object_volume(object: RigidObjectCfg) -> float:

    spawn_cfg = object.spawn # Changed from object.cfg to object.spawn

    if isinstance(spawn_cfg, sim_utils.SphereCfg):
        # Volume = (4/3)πr³
        return (4/3) * 3.14159 * (spawn_cfg.radius ** 3)
    elif isinstance(spawn_cfg, sim_utils.CuboidCfg):
        # Volume = length * width * height
        return spawn_cfg.size[0] * spawn_cfg.size[1] * spawn_cfg.size[2]
    elif isinstance(spawn_cfg, sim_utils.CylinderCfg):
        # Volume = πr²h
        return 3.14159 * (spawn_cfg.radius ** 2) * spawn_cfg.height
    elif isinstance(spawn_cfg, sim_utils.ConeCfg):
        # Volume = (1/3)πr²h
        return (1/3) * 3.14159 * (spawn_cfg.radius ** 2) * spawn_cfg.height
    elif isinstance(spawn_cfg, sim_utils.CapsuleCfg):
        # Volume = πr²h + (4/3)πr³ (cylinder + two hemisphere ends)
        cylinder_vol = 3.14159 * (spawn_cfg.radius ** 2) * spawn_cfg.height
        hemispheres_vol = (4/3) * 3.14159 * (spawn_cfg.radius ** 3)
        return cylinder_vol + hemispheres_vol
    else:
        raise ValueError(f"Unsupported object type: {type(spawn_cfg)}")

# ================================================================
# Helper utilities
# ================================================================

def get_object_name_mapping(objects_cfg: Dict) -> Dict[str, str]:
    """Return mapping from numbered scene names (``Object1`` …) to descriptive
    comments extracted from the generated ``objects_config`` JSON.

    The numbering scheme in :func:`generate_object_configs` enumerates objects
    strictly in this order: all spheres, cubes, cylinders, cones, capsules.  We
    therefore iterate through the *_comments lists in the same order and build
    a mapping so that downstream prompt generation (LLM) can reference the
    semantic description without needing the entire object configuration.

    Example return::

        {
            "Object1": "ball for robot to pick up",
            "Object2": "table for the ball to sit on",
        }

    Args:
        objects_cfg: Parsed JSON dict produced by ``generate_objects_config``.

    Returns:
        Dictionary mapping numbered object names to human-friendly comments.
    """

    mapping: Dict[str, str] = {}
    
    # This logic now needs to respect the generated order, not a fixed one.
    # We can rebuild the order from the config itself.
    
    order_specs = []
    if objects_cfg.get("num_spheres", 0) > 0:
        order_specs.append(("sphere", "_sphere_comments", objects_cfg.get("num_spheres", 0)))
    if objects_cfg.get("num_cubes", 0) > 0:
        order_specs.append(("cube", "_cube_comments", objects_cfg.get("num_cubes", 0)))
    if objects_cfg.get("num_cylinders", 0) > 0:
        order_specs.append(("cylinder", "_cylinder_comments", objects_cfg.get("num_cylinders", 0)))
    if objects_cfg.get("num_cones", 0) > 0:
        order_specs.append(("cone", "_cone_comments", objects_cfg.get("num_cones", 0)))
    if objects_cfg.get("num_capsules", 0) > 0:
        order_specs.append(("capsule", "_capsule_comments", objects_cfg.get("num_capsules", 0)))

    # This is still not quite right. The config is already generated.
    # We need to reconstruct the order from the config.
    # A better way is to iterate through the object types in the order they were processed.
    
    # Let's try to determine the order from the config fields. This is tricky.
    # The new way is flawed. We'll assume the new way is used.
    
    # We will rely on the order of comments as they appear in the config.
    # This is fragile. The best solution is to use the `object_order` from the source.
    # But this function only receives the final config.

    # Let's rebuild the mapping based on the filled lists in the config.
    # This is the most robust way to do it without changing the function signature.
    
    object_counter = 1
    
    # The order of these checks now matters. It should match `_process_object`
    for i in range(objects_cfg.get("num_spheres", 0)):
        mapping[f"Object{object_counter}"] = objects_cfg["_sphere_comments"][i]
        object_counter += 1
    for i in range(objects_cfg.get("num_cubes", 0)):
        mapping[f"Object{object_counter}"] = objects_cfg["_cube_comments"][i]
        object_counter += 1
    for i in range(objects_cfg.get("num_cylinders", 0)):
        mapping[f"Object{object_counter}"] = objects_cfg["_cylinder_comments"][i]
        object_counter += 1
    for i in range(objects_cfg.get("num_cones", 0)):
        mapping[f"Object{object_counter}"] = objects_cfg["_cone_comments"][i]
        object_counter += 1
    for i in range(objects_cfg.get("num_capsules", 0)):
        mapping[f"Object{object_counter}"] = objects_cfg["_capsule_comments"][i]
        object_counter += 1
        
    return mapping

# ================================================================
# Main class for generation
# ================================================================
class ObjectGenerator:
    """
    This class is a placeholder for the main generation logic.
    It would typically handle the full generation process, including
    task description parsing, object ordering, and final configuration.
    """
    def __init__(self):
        pass

    def generate_objects(self, task_description: str) -> Dict[str, Any]:
        """
        Generates a dictionary of object configurations based on a task description.
        This is a simplified example. In a real application, this would
        involve parsing the task description, determining object types and
        quantities, and then calling generate_object_configs.
        """
        # In a real scenario, you would parse task_description to determine
        # object types, quantities, and their properties.
        # For example:
        # - Spheres: 2, Radius: 0.5
        # - Cubes: 1, Size: (1.0, 0.5, 0.5)
        # - Cylinders: 1, Radius: 0.3, Height: 1.0
        # - Cones: 1, Radius: 0.4, Height: 0.8
        # - Capsules: 1, Radius: 0.2, Height: 0.6

        # This is a placeholder. In a real application, you would call
        # generate_object_configs with a list of objects.
        # For now, we'll just return a dummy config.

        # Example of how to call generate_object_configs
        # objects_list = [
        #     {"type": "sphere", "comment": "a small ball", "radius": 0.5},
        #     {"type": "cube", "comment": "a large cube", "size": (1.0, 1.0, 1.0)},
        #     {"type": "cylinder", "comment": "a tall cylinder", "size": (0.3, 1.0)},
        #     {"type": "cone", "comment": "a wide cone", "size": (0.4, 0.8)},
        #     {"type": "capsule", "comment": "a long capsule", "size": (0.2, 0.6)}
        # ]
        # return generate_object_configs(objects_list)

        # For now, return a dummy config
        return {
            "num_spheres": 2, "num_cubes": 1, "num_cylinders": 1, "num_cones": 1, "num_capsules": 1,
            "position_spheres": [0.0, 0.0, 0.0], "radius_spheres": [0.5], "_sphere_comments": ["a small ball"],
            "position_cubes": [1.0, 0.0, 0.0], "size_cubes": [(1.0, 1.0, 1.0)], "_cube_comments": ["a large cube"],
            "position_cylinders": [2.0, 0.0, 0.0], "radius_cylinders": [0.3], "height_cylinders": [1.0], "_cylinder_comments": ["a tall cylinder"],
            "position_cones": [3.0, 0.0, 0.0], "radius_cones": [0.4], "height_cones": [0.8], "_cone_comments": ["a wide cone"],
            "position_capsules": [4.0, 0.0, 0.0], "radius_capsules": [0.2], "height_capsules": [0.6], "_capsule_comments": ["a long capsule"]
        }
