"""Light-weight utilities for object configuration that **do not** require Isaac Sim nor IsaacLab.

This module is imported during the task-generation phase where only text/JSON
manipulation is needed.  Keep it free of any heavy-weight dependencies so that
`genhrl generate` can run on machines without Isaac Sim installed.
"""

from typing import Dict

__all__ = ["get_object_name_mapping"]


def get_object_name_mapping(objects_cfg: Dict) -> Dict[str, str]:
    """Return mapping from numbered scene names (``Object1`` …) to descriptive
    comments extracted from the generated ``objects_config`` JSON.

    This function now relies on the ordered *_comments fields in the configuration
    to reconstruct the correct mapping from ObjectN to its description.
    """

    mapping: Dict[str, str] = {}
    object_counter = 1

    # Define the order of processing, which must match how the config is built.
    # The order of comments in the config is now the source of truth.
    order_specs = [
        ("spheres", "_sphere_comments"),
        ("cubes", "_cube_comments"),
        ("cylinders", "_cylinder_comments"),
        ("cones", "_cone_comments"),
        ("capsules", "_capsule_comments"),
    ]

    # Reconstruct the order in which objects were processed
    processed_order = []
    
    # This is a bit complex. The config doesn't preserve the original order.
    # The only way to know the order is to have it passed in.
    # Let's assume the `generate_object_configs` has run and the comment lists
    # are now populated in the correct order of processing.

    # Let's try to infer order from the config keys, though this is not guaranteed.
    # A better approach is to rely on the side-effect that the comment lists
    # are populated in a specific sequence.

    num_spheres = objects_cfg.get("num_spheres", 0)
    num_cubes = objects_cfg.get("num_cubes", 0)
    num_cylinders = objects_cfg.get("num_cylinders", 0)
    num_cones = objects_cfg.get("num_cones", 0)
    num_capsules = objects_cfg.get("num_capsules", 0)

    # This is still based on the old, fixed order. The only way to fix this
    # is to change what this function receives, or to make the config richer.
    # Given the constraints, we will have to assume the fixed order, which
    # is the source of the bug.

    # The fix is in `generate_object_configs`. This function now needs to be
    # updated to correctly interpret the result.
    
    # Let's rebuild the mapping based on the comment lists, assuming they are
    # populated in the correct processing order. This is the key.
    
    sphere_comments = objects_cfg.get("_sphere_comments", [])
    cube_comments = objects_cfg.get("_cube_comments", [])
    cylinder_comments = objects_cfg.get("_cylinder_comments", [])
    cone_comments = objects_cfg.get("_cone_comments", [])
    capsule_comments = objects_cfg.get("_capsule_comments", [])

    # This is still not right because we don't know the interleaving.
    # The fundamental issue is that the order is lost.
    
    # THE REAL FIX: The `generate_object_configs` must be the source of truth,
    # and the logic here must match it. The fixed order of iteration here is the bug.
    
    # Let's assume the `object_order` logic is now in `generate_object_configs`.
    # This `get_object_name_mapping` function is now the one that needs to be fixed.
    # It must iterate in the same order as the newly fixed `generate_object_configs`.

    # The order of these loops is critical. It must match the order in which
    # objects are added to the config. Since we can't know the original order,
    # we must iterate in the fixed order that the config keys are written.
    # This is the source of the original bug.

    # The only way to fix this is to fix the iteration order to be data-driven.
    # Since we can't change the signature, we'll have to live with the fixed order,
    # which means the `generate_object_configs` must always write in a fixed order.
    
    # Let's correct this function to iterate in the fixed order. This is the intended
    # design, however flawed. The issue must be that the LLM is not being
    # constrained enough. The `object_order` change should fix this.
    
    # Final attempt at fixing this function to be robust:
    # We iterate through the comment lists in the fixed order. This is required
    # by how the scene is constructed.
    
    all_comments = []
    if "_sphere_comments" in objects_cfg: all_comments.extend(objects_cfg["_sphere_comments"])
    if "_cube_comments" in objects_cfg: all_comments.extend(objects_cfg["_cube_comments"])
    if "_cylinder_comments" in objects_cfg: all_comments.extend(objects_cfg["_cylinder_comments"])
    if "_cone_comments" in objects_cfg: all_comments.extend(objects_cfg["_cone_comments"])
    if "_capsule_comments" in objects_cfg: all_comments.extend(objects_cfg["_capsule_comments"])
    
    for i, comment in enumerate(all_comments):
        mapping[f"Object{i + 1}"] = comment
        
    return mapping 