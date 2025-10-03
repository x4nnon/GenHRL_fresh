def generate_objects_config_prompt(scene_plan: str, task_description: str) -> str:
    prompt = f"""
    # 🧱 SCENE OBJECT CONFIGURATION GENERATOR

    ## 📄 Task Description
    {task_description}

    ## 🏗️ Scene Construction Plan
    {scene_plan}

    ## 📋 Configuration Requirements

    ### 🔍 Core Principles
    1. Create a physically realistic scene using ONLY primitive shapes
    2. Each object must be a SINGLE primitive (no composite objects)
    3. You must create ALL objects needed for the scene
    4. Place objects in sensible positions relative to the robot
    5. All objects must have realistic physics properties
    6. **PRESERVE USER TERMINOLOGY** - Use the exact object names from the task description (e.g., if user says "box", use "box" not "cardboardBox")
    7. You must reply in valid JSON format. with no leading or training text. No markdown handles such as ```json etc.

    ### 🚫 Critical Constraints
    - ❌ DO NOT combine primitives to form complex objects
    - ❌ DO NOT assume any objects exist in the scene already
    - ❌ DO NOT create a floor (it exists by default)
    - ❌ Objects in mid-air (z > 0) will fall unless placed on another object
    - ❌ DO NOT change or expand object names from the task description (keep "box" as "box", not "cardboardBox")

    ### 🧪 Available Primitive Shapes
    - Spheres (radius, mass)
    - Cubes (size [x,y,z], mass)
    - Cylinders (radius, height, mass)
    - Cones (radius, height, mass)
    - Capsules (radius, height, mass)

    ### 📏 Object Constraints
    1. All positions must be [x,y,z] coordinates
    2. All masses must be positive and realistic (in kg)
    3. All sizes/radii/heights must be positive (in meters)
    4. Maximum of 5 total objects
    
    ## 📊 Physical Realism Guidelines
    - Table-like objects: cubes with appropriate dimensions
    - Ball-like objects: spheres with realistic mass
    - Walls: thin, tall cubes
    - Posts/poles: cylinders or capsules
    - Steps/platforms: cubes with appropriate height
    
    Objects should have realistic masses:
    - Light object (tennis ball): 0.05-0.2 kg
    - Medium object (book): 0.5-2 kg
    - Heavy object (chair): 10-20 kg
    - Very heavy object (wall): 100+ kg

    ## 🧮 JSON Format Requirements
    ```json
    {{
        "_comment": "Scene description explaining what objects represent",
        "num_spheres": n,
        "_sphere_comments": ["Description of sphere 1", "Description of sphere 2"],
        "mass_spheres": [m1, m2, ...],
        "position_spheres": [[x1,y1,z1], [x2,y2,z2], ...],
        "rotation_spheres": [[w1,x1,y1,z1], [w2,x2,y2,z2], ...],
        "radius_spheres": [r1, r2, ...],
        
        "num_cubes": n,
        "_cube_comments": ["Description of cube 1", "Description of cube 2"],
        "mass_cubes": [m1, m2, ...],
        "position_cubes": [[x1,y1,z1], [x2,y2,z2], ...],
        "rotation_cubes": [[w1,x1,y1,z1], [w2,x2,y2,z2], ...],
        "size_cubes": [[x1,y1,z1], [x2,y2,z2], ...],
        
        "num_cylinders": n,
        ...
        "num_cones": n,
        ...
        "num_capsules": n,
        ...
    }}
    ```

    ## 📐 Object Placement Guidance
    - Position objects to facilitate task completion
    - Consider the robot's reach and mobility
    - Ensure objects are physically stable
    - Allow enough space for the robot to navigate
    
    ## 🔍 JSON Example (Tennis Ball and Table)
    ```json
    {{
        "_comment": "Scene contains a tennis ball on a table",
        "num_spheres": 1,
        "_sphere_comments": ["ball for robot to pick up"],
        "mass_spheres": [0.145],
        "position_spheres": [[1.0, 0.0, 1.01]],
        "rotation_spheres": [[1.0, 0.0, 0.0, 0.0]],
        "radius_spheres": [0.07],
        
        "num_cubes": 1,
        "_cube_comments": ["table for the ball to sit on"],
        "mass_cubes": [15.0],
        "position_cubes": [[1.0, 0.0, 0.5]],
        "rotation_cubes": [[1.0, 0.0, 0.0, 0.0]],
        "size_cubes": [[1.0, 1.0, 1.0]],
        
        "num_cylinders": 0,
        "num_cones": 0,
        "num_capsules": 0
    }}
    ```

    ## ⚠️ IMPORTANT
    - Return ONLY the JSON configuration, no explanation or extra text
    - INCLUDE all object types in JSON, even if some are unused (set to 0)
    - Format numbers as floats (1.0 not 1)
    - Include descriptive comments using the EXACT object names from the task description
    - Max 5 objects total across all primitive types
    - **CRITICAL: Use the exact object terminology from the user's task description**

    Generate the JSON configuration for this task now.
    """
    return prompt