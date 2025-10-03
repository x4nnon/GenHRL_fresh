#!/usr/bin/env python3
"""
Test script for image generation functionality.
This creates a simple test hierarchy to verify graphviz integration works.
"""

import json
from pathlib import Path
from visualize_skill_hierarchies import SkillHierarchyVisualizer

def create_test_hierarchy():
    """Create a simple test hierarchy for testing."""
    return {
        "name": "test_task",
        "description": "A simple test task to verify image generation works correctly",
        "children": [
            {
                "name": "high_level_skill_1",
                "description": "First high level skill with multiple sub-skills",
                "children": [
                    {
                        "name": "sub_skill_1_1",
                        "description": "First sub-skill",
                        "object_config_paths": ["/path/to/config1.json"]
                    },
                    {
                        "name": "sub_skill_1_2", 
                        "description": "Second sub-skill",
                        "object_config_paths": ["/path/to/config2.json"]
                    }
                ]
            },
            {
                "name": "high_level_skill_2",
                "description": "Second high level skill",
                "children": [
                    {
                        "name": "sub_skill_2_1",
                        "description": "Another sub-skill",
                        "object_config_paths": ["/path/to/config3.json"]
                    }
                ]
            }
        ]
    }

def main():
    """Test the image generation functionality."""
    print("🧪 Testing image generation functionality...")
    
    # Create test output directory
    test_output_dir = Path(__file__).parent.parent.parent / "writing_images" / "test_hierarchies"
    
    try:
        # Create visualizer with image output
        visualizer = SkillHierarchyVisualizer(
            tasks_base_path="dummy",  # Not used for this test
            save_output=True,
            output_dir=str(test_output_dir),
            output_format="image"
        )
        
        # Create test hierarchy
        test_hierarchy = create_test_hierarchy()
        
        # Generate image
        dot = visualizer.create_graph_diagram(test_hierarchy, "test_task")
        
        if dot:
            visualizer.save_graph_image(dot, "test_task")
            print("✅ Image generation test passed!")
            print(f"📁 Check {test_output_dir} for generated images")
        else:
            print("❌ Failed to create graph diagram")
            
    except Exception as e:
        print(f"❌ Image generation test failed: {e}")
        print("💡 Make sure graphviz is installed:")
        print("   pip install graphviz")
        print("   sudo apt-get install graphviz  # Ubuntu/Debian")
        print("   brew install graphviz          # macOS")

if __name__ == "__main__":
    main()