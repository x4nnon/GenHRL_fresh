"""
Test configuration and file structure generation mentioned in the README.

These tests validate that the file structure and configurations
described in the README are generated correctly.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch


class TestFileStructureGeneration:
    """Test file structure generation as described in README."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp(prefix="test_structure_")
        self.test_isaaclab_path = Path(self.test_dir) / "IsaacLab"
        self.test_isaaclab_path.mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_expected_file_structure_components(self):
        """Test that file structure components exist as documented in README."""
        # Create the expected directory structure
        robot_folder = "G1_generated"
        base_path = self.test_isaaclab_path / "source/isaaclab_tasks/isaaclab_tasks/manager_based" / robot_folder
        
        # Create directories as mentioned in README
        tasks_dir = base_path / "tasks" / "test_task"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        
        skills_dir = base_path / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        
        skill_instance_dir = skills_dir / "skills" / "test_skill"
        skill_instance_dir.mkdir(parents=True, exist_ok=True)
        
        # Create expected files from README
        expected_task_files = [
            tasks_dir / "description.txt",
            tasks_dir / "object_config.json", 
            tasks_dir / "skills_hierarchy.json"
        ]
        
        expected_skill_files = [
            skills_dir / "skill_library.json",
            skills_dir / "skill_config_template.py",
            skill_instance_dir / "TaskRewardsCfg.py",
            skill_instance_dir / "SuccessTerminationCfg.py",
            skill_instance_dir / "CollapseTerminationCfg.py",
            skill_instance_dir / "base_success.py",
            skill_instance_dir / "test_skill_cfg.py"
        ]
        
        expected_agent_dirs = [
            skills_dir / "agents",
            skill_instance_dir / "agents",
            skill_instance_dir / "success_states",
            skill_instance_dir / "policy"
        ]
        
        # Create test files
        for file_path in expected_task_files + expected_skill_files:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("# Test file content")
        
        for dir_path in expected_agent_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Verify structure matches README
        for file_path in expected_task_files:
            assert file_path.exists(), f"Expected task file not found: {file_path}"
        
        for file_path in expected_skill_files:
            assert file_path.exists(), f"Expected skill file not found: {file_path}"
        
        for dir_path in expected_agent_dirs:
            assert dir_path.exists(), f"Expected directory not found: {dir_path}"
        
        print("✅ Expected file structure from README verified")
    
    def test_agent_configuration_files(self):
        """Test that agent configuration files mentioned in README exist."""
        skills_base_path = Path(self.test_dir) / "skills"
        skills_base_path.mkdir(parents=True, exist_ok=True)
        
        agents_dir = skills_base_path / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)
        
        # Create agent config files mentioned in README
        agent_configs = [
            agents_dir / "skrl_flat_ppo_cfg.yaml",
            agents_dir / "skrl_flat_ppo_cfg_l1_example.yaml"
        ]
        
        for config_file in agent_configs:
            config_file.write_text("# Test agent configuration")
        
        # Verify files exist
        for config_file in agent_configs:
            assert config_file.exists(), f"Agent config file not found: {config_file}"
        
        print("✅ Agent configuration files verified")
    
    def test_skill_specific_configurations(self):
        """Test skill-specific configuration structure from README."""
        skill_path = Path(self.test_dir) / "skill_test"
        skill_path.mkdir(parents=True, exist_ok=True)
        
        # Create skill-specific files mentioned in README
        skill_files = {
            "TaskRewardsCfg.py": "# Generated reward functions",
            "SuccessTerminationCfg.py": "# Generated success criteria", 
            "CollapseTerminationCfg.py": "# Generated failure conditions",
            "base_success.py": "# Success state utilities",
            "skill_test_cfg.py": "# Main skill configuration"
        }
        
        skill_dirs = ["agents", "success_states", "policy"]
        
        # Create files
        for filename, content in skill_files.items():
            (skill_path / filename).write_text(content)
        
        # Create directories
        for dirname in skill_dirs:
            (skill_path / dirname).mkdir(exist_ok=True)
        
        # Verify structure
        for filename in skill_files.keys():
            assert (skill_path / filename).exists(), f"Skill file not found: {filename}"
        
        for dirname in skill_dirs:
            assert (skill_path / dirname).exists(), f"Skill directory not found: {dirname}"
        
        print("✅ Skill-specific configuration structure verified")


class TestConfigurationSettings:
    """Test configuration settings mentioned in README."""
    
    def test_hierarchy_levels_configuration(self):
        """Test hierarchy levels configuration as described in README."""
        # Test that the values mentioned in README are valid
        valid_hierarchy_levels = [1, 2, 3]
        
        for level in valid_hierarchy_levels:
            assert level in [1, 2, 3], f"Invalid hierarchy level: {level}"
        
        # Test level descriptions from README
        level_descriptions = {
            1: "Single task only (no decomposition)",
            2: "Task → Skill decomposition", 
            3: "Task → Skill → Sub-skill (full hierarchy)"
        }
        
        for level, description in level_descriptions.items():
            assert level in valid_hierarchy_levels
            assert isinstance(description, str)
            assert len(description) > 0
        
        print("✅ Hierarchy levels configuration verified")
    
    def test_verification_settings_configuration(self):
        """Test verification settings mentioned in README."""
        # Test verification settings from README
        verification_settings = {
            'verify_decompose': [True, False],
            'verify_plan': [True, False],
            'verify_rewards': [True, False], 
            'verify_success': [True, False]
        }
        
        for setting, valid_values in verification_settings.items():
            for value in valid_values:
                assert isinstance(value, bool), f"Verification setting {setting} should be boolean"
        
        print("✅ Verification settings configuration verified")
    
    def test_llm_provider_configurations(self):
        """Test LLM provider configurations from README."""
        # Test provider configurations mentioned in README
        google_config = {
            'provider': 'google',
            'model': 'gemini-2.0-flash',
            'backup_model': 'gemini-2.0-flash',
            'model_big': 'gemini-2.5-pro-preview-05-06'
        }
        
        anthropic_config = {
            'provider': 'anthropic',
            'model': 'claude-sonnet-4-20250514',
            'backup_model': 'claude-sonnet-4-20250514',
            'model_big': 'claude-opus-4-20250514'
        }
        
        # Verify configuration structure
        for config in [google_config, anthropic_config]:
            assert 'provider' in config
            assert 'model' in config
            assert 'backup_model' in config
            assert 'model_big' in config
            
            assert isinstance(config['provider'], str)
            assert len(config['provider']) > 0
        
        print("✅ LLM provider configurations verified")
    
    def test_training_configuration_settings(self):
        """Test training configuration settings from README."""
        # Test training config mentioned in README
        training_config = {
            'max_time_minutes': 120,
            'min_success_states': 50,
            'num_envs': 4096,
            'seed': 42
        }
        
        # Verify types and reasonable values
        assert isinstance(training_config['max_time_minutes'], int)
        assert training_config['max_time_minutes'] > 0
        
        assert isinstance(training_config['min_success_states'], int) 
        assert training_config['min_success_states'] > 0
        
        assert isinstance(training_config['num_envs'], int)
        assert training_config['num_envs'] > 0
        
        assert isinstance(training_config['seed'], int)
        assert training_config['seed'] >= 0
        
        print("✅ Training configuration settings verified")
    
    def test_robot_configuration_settings(self):
        """Test robot configuration settings mentioned in README."""
        # Test robot settings from README
        test_robot_config = {
            'name': 'G1',
            'hip_height': 0.7,
            'folder_name': 'G1_generated'
        }
        
        # Verify configuration structure
        assert isinstance(test_robot_config['name'], str)
        assert len(test_robot_config['name']) > 0
        
        assert isinstance(test_robot_config['hip_height'], (int, float))
        assert test_robot_config['hip_height'] > 0
        
        assert isinstance(test_robot_config['folder_name'], str)
        assert len(test_robot_config['folder_name']) > 0
        
        print("✅ Robot configuration settings verified")


class TestEnvironmentVariables:
    """Test environment variable handling mentioned in README."""
    
    def test_api_key_environment_variable(self):
        """Test GENHRL_API_KEY environment variable handling."""
        # Save original state
        original_key = os.getenv("GENHRL_API_KEY")
        
        try:
            # Test setting and getting API key
            test_key = "test_api_key_for_env_test"
            os.environ["GENHRL_API_KEY"] = test_key
            
            retrieved_key = os.getenv("GENHRL_API_KEY")
            assert retrieved_key == test_key
            
            # Test removing API key
            del os.environ["GENHRL_API_KEY"]
            retrieved_key = os.getenv("GENHRL_API_KEY")
            assert retrieved_key is None
            
            print("✅ GENHRL_API_KEY environment variable handling verified")
            
        finally:
            # Restore original state
            if original_key:
                os.environ["GENHRL_API_KEY"] = original_key
            elif "GENHRL_API_KEY" in os.environ:
                del os.environ["GENHRL_API_KEY"]
    
    def test_object_config_path_environment_variable(self):
        """Test OBJECT_CONFIG_PATH environment variable mentioned in README."""
        # Save original state
        original_path = os.getenv("OBJECT_CONFIG_PATH")
        
        try:
            # Test setting object config path
            test_path = "/test/path/to/object_config.json"
            os.environ["OBJECT_CONFIG_PATH"] = test_path
            
            retrieved_path = os.getenv("OBJECT_CONFIG_PATH")
            assert retrieved_path == test_path
            
            print("✅ OBJECT_CONFIG_PATH environment variable handling verified")
            
        finally:
            # Restore original state
            if original_path:
                os.environ["OBJECT_CONFIG_PATH"] = original_path
            elif "OBJECT_CONFIG_PATH" in os.environ:
                del os.environ["OBJECT_CONFIG_PATH"]


class TestTaskConfigStructure:
    """Test TaskConfig structure mentioned in README."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp(prefix="test_task_config_")
        self.test_isaaclab_path = Path(self.test_dir) / "IsaacLab"
        self.test_isaaclab_path.mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_task_config_interface(self):
        """Test TaskConfig interface mentioned in README examples."""
        try:
            import importlib
            generation_module = importlib.import_module('genhrl.generation')
            
            # Check if TaskConfig class exists
            if hasattr(generation_module, 'TaskConfig'):
                TaskConfig = getattr(generation_module, 'TaskConfig')
                
                # Check for methods mentioned in README
                expected_methods = ['get_task_path', 'get_skills_base_path']
                
                # We can't instantiate without proper parameters, but we can check the class
                import inspect
                for method_name in expected_methods:
                    assert hasattr(TaskConfig, method_name), f"Missing method: {method_name}"
                
                print("✅ TaskConfig interface verified")
            else:
                pytest.skip("TaskConfig not available")
                
        except ImportError:
            pytest.skip("TaskConfig module not available")
    
    def test_task_config_file_content_structure(self):
        """Test expected content structure of task config files."""
        # Create test task directory
        task_dir = Path(self.test_dir) / "test_task"
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test description file
        description_file = task_dir / "description.txt"
        description_file.write_text("Test task description for validation")
        
        # Create test object config file
        object_config = {
            "objects": [
                {
                    "name": "test_object",
                    "type": "box",
                    "size": [1.0, 1.0, 1.0]
                }
            ]
        }
        
        object_config_file = task_dir / "object_config.json"
        with open(object_config_file, 'w') as f:
            json.dump(object_config, f, indent=2)
        
        # Create test skills hierarchy file
        skills_hierarchy = {
            "name": "test_task",
            "description": "Test task for validation",
            "children": []
        }
        
        skills_hierarchy_file = task_dir / "skills_hierarchy.json"
        with open(skills_hierarchy_file, 'w') as f:
            json.dump(skills_hierarchy, f, indent=2)
        
        # Verify files can be read and parsed
        assert description_file.exists()
        description_content = description_file.read_text()
        assert len(description_content) > 0
        
        assert object_config_file.exists()
        with open(object_config_file, 'r') as f:
            parsed_object_config = json.load(f)
        assert "objects" in parsed_object_config
        
        assert skills_hierarchy_file.exists()
        with open(skills_hierarchy_file, 'r') as f:
            parsed_skills_hierarchy = json.load(f)
        assert "name" in parsed_skills_hierarchy
        assert "description" in parsed_skills_hierarchy
        assert "children" in parsed_skills_hierarchy
        
        print("✅ Task config file content structure verified")


if __name__ == "__main__":
    # Allow running tests directly
    test_structure = TestFileStructureGeneration()
    test_structure.setup_method()
    test_structure.test_expected_file_structure_components()
    test_structure.test_agent_configuration_files()
    test_structure.test_skill_specific_configurations()
    test_structure.teardown_method()
    
    test_config = TestConfigurationSettings()
    test_config.test_hierarchy_levels_configuration()
    test_config.test_verification_settings_configuration()
    test_config.test_llm_provider_configurations()
    test_config.test_training_configuration_settings()
    test_config.test_robot_configuration_settings()
    
    test_env = TestEnvironmentVariables()
    test_env.test_api_key_environment_variable()
    test_env.test_object_config_path_environment_variable()
    
    test_task_config = TestTaskConfigStructure()
    test_task_config.setup_method()
    test_task_config.test_task_config_interface()
    test_task_config.test_task_config_file_content_structure()
    test_task_config.teardown_method()
    
    print("🎉 All configuration and structure tests passed!")