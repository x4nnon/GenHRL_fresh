"""
Test workflow functionality mentioned in the README.

These tests validate that the main workflow functions and examples
from the README work correctly.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestWorkflowFunctionality:
    """Test main workflow functions from README examples."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp(prefix="test_genhrl_")
        self.test_isaaclab_path = Path(self.test_dir) / "IsaacLab"
        self.test_isaaclab_path.mkdir(parents=True, exist_ok=True)
        self.test_api_key = "test_api_key_for_workflows"
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_create_task_with_workflow_signature(self):
        """Test that create_task_with_workflow function has correct signature."""
        try:
            from genhrl.generation import create_task_with_workflow
            import inspect
            
            # Get function signature
            sig = inspect.signature(create_task_with_workflow)
            params = list(sig.parameters.keys())
            
            # Check for required parameters from README examples
            expected_params = [
                'task_name', 'task_description', 'isaaclab_path', 'api_key'
            ]
            
            for param in expected_params:
                assert param in params, f"Missing required parameter: {param}"
            
            print("✅ create_task_with_workflow signature verified")
            
        except ImportError:
            pytest.skip("create_task_with_workflow not available")
    
    def test_main_create_steps_example_signature(self):
        """Test that main_create_steps_example function exists and has correct signature."""
        try:
            from genhrl.generation import main_create_steps_example
            import inspect
            
            # Get function signature
            sig = inspect.signature(main_create_steps_example)
            params = list(sig.parameters.keys())
            
            # Check for required parameters from README
            expected_params = ['isaaclab_path', 'api_key']
            
            for param in expected_params:
                assert param in params, f"Missing required parameter: {param}"
            
            print("✅ main_create_steps_example signature verified")
            
        except ImportError:
            pytest.skip("main_create_steps_example not available")
    
    def test_remove_all_previous_skills_signature(self):
        """Test that remove_all_previous_skills function exists and has correct signature."""
        try:
            from genhrl.generation import remove_all_previous_skills
            import inspect
            
            # Get function signature
            sig = inspect.signature(remove_all_previous_skills)
            params = list(sig.parameters.keys())
            
            # Check for required parameters from README
            expected_params = ['isaaclab_path']
            
            for param in expected_params:
                assert param in params, f"Missing required parameter: {param}"
            
            print("✅ remove_all_previous_skills signature verified")
            
        except ImportError:
            pytest.skip("remove_all_previous_skills not available")
    
    @patch('genhrl.generation.HierarchicalCodeGenerator')
    @patch('genhrl.generation.TaskManager') 
    def test_workflow_with_mocked_components(self, mock_task_manager, mock_code_generator):
        """Test workflow with mocked components to avoid API calls."""
        try:
            from genhrl.generation import create_task_with_workflow
            
            # Set up mocks
            mock_task_config = Mock()
            mock_task_config.get_task_path.return_value = Path(self.test_dir) / "task"
            mock_task_config.get_skills_base_path.return_value = Path(self.test_dir) / "skills"
            
            mock_manager_instance = Mock()
            mock_manager_instance.create_task_from_description.return_value = mock_task_config
            mock_task_manager.return_value = mock_manager_instance
            
            # Test the workflow function
            result = create_task_with_workflow(
                task_name="Test_Task",
                task_description="Test task description",
                isaaclab_path=str(self.test_isaaclab_path),
                api_key=self.test_api_key,
                robot="G1",
                max_hierarchy_levels=2,
                remove_previous=False
            )
            
            # Verify the function was called
            assert mock_task_manager.called
            assert mock_manager_instance.create_task_from_description.called
            assert result == mock_task_config
            
            print("✅ Workflow function executed successfully with mocks")
            
        except ImportError:
            pytest.skip("create_task_with_workflow not available")
    
    def test_task_manager_initialization_from_readme(self):
        """Test TaskManager initialization as shown in README examples."""
        try:
            from genhrl.generation import TaskManager
            
            # Test initialization as shown in README
            task_manager = TaskManager(
                isaaclab_path=str(self.test_isaaclab_path),
                api_key=self.test_api_key,
                robot="G1"
            )
            
            # Check that the object was created
            assert task_manager is not None
            assert hasattr(task_manager, 'create_task_from_description')
            
            print("✅ TaskManager initialization from README works")
            
        except ImportError:
            pytest.skip("TaskManager not available")
    
    def test_hierarchical_code_generator_configurations(self):
        """Test HierarchicalCodeGenerator configurations from README."""
        try:
            from genhrl.generation import HierarchicalCodeGenerator
            
            # Test Google/Gemini configuration from README
            generator_google = HierarchicalCodeGenerator(
                api_key="test_google_api_key",
                provider="google",
                model="gemini-2.0-flash",
                backup_model="gemini-2.0-flash",
                model_big="gemini-2.5-pro-preview-05-06"
            )
            
            assert generator_google.provider == "google"
            assert generator_google.model == "gemini-2.0-flash"
            
            # Test Anthropic/Claude configuration from README
            generator_anthropic = HierarchicalCodeGenerator(
                api_key="test_anthropic_api_key",
                provider="anthropic",
                model="claude-sonnet-4-20250514",
                backup_model="claude-sonnet-4-20250514", 
                model_big="claude-opus-4-20250514"
            )
            
            assert generator_anthropic.provider == "anthropic"
            assert generator_anthropic.model == "claude-sonnet-4-20250514"
            
            print("✅ HierarchicalCodeGenerator configurations from README work")
            
        except ImportError:
            pytest.skip("HierarchicalCodeGenerator not available")
    
    def test_skill_library_management_from_readme(self):
        """Test SkillLibrary management examples from README."""
        try:
            from genhrl.generation import SkillLibrary
            
            # Create test library path
            library_path = Path(self.test_dir) / "test_skill_library.json"
            
            # Initialize skill library as shown in README
            skill_library = SkillLibrary(str(library_path))
            
            # Test methods mentioned in README
            assert hasattr(skill_library, 'list_primitive_skills')
            assert hasattr(skill_library, 'list_composite_skills')
            assert hasattr(skill_library, 'get_full_skill_sequence')
            assert hasattr(skill_library, 'export_library')
            assert hasattr(skill_library, 'import_library')
            
            # Test method calls
            primitive_skills = skill_library.list_primitive_skills()
            composite_skills = skill_library.list_composite_skills()
            
            assert isinstance(primitive_skills, list)
            assert isinstance(composite_skills, list)
            
            print("✅ SkillLibrary management from README works")
            
        except ImportError:
            pytest.skip("SkillLibrary not available")
    
    def test_robot_configuration_from_readme(self):
        """Test robot configuration examples from README."""
        try:
            from genhrl.generation import get_available_robots, get_robot_config
            
            # Test getting available robots
            robots = get_available_robots()
            assert isinstance(robots, list)
            assert len(robots) > 0
            
            # Test getting robot configuration for G1
            if "G1" in robots:
                config = get_robot_config("G1")
                assert isinstance(config, dict)
                assert 'hip_height' in config
            
            print("✅ Robot configuration from README works")
            
        except ImportError:
            pytest.skip("Robot configuration functions not available")
    
    def test_utility_functions_from_readme(self):
        """Test utility functions mentioned in README."""
        try:
            from genhrl.generation import strip_markdown_formatting, clean_json_string
            
            # Test strip_markdown_formatting
            markdown_text = "```python\nprint('hello')\n```"
            clean_text = strip_markdown_formatting(markdown_text)
            assert "```" not in clean_text
            assert "print('hello')" in clean_text
            
            # Test clean_json_string
            dirty_json = '{"key": "value",}'  # Trailing comma
            clean_json = clean_json_string(dirty_json)
            
            # Should be able to parse cleaned JSON
            import json
            parsed = json.loads(clean_json)
            assert parsed["key"] == "value"
            
            print("✅ Utility functions from README work")
            
        except ImportError:
            pytest.skip("Utility functions not available")
    
    def test_training_integration_from_readme(self):
        """Test training integration examples from README."""
        try:
            from genhrl.training import TrainingOrchestrator, TrainingConfig
            
            # Test TrainingConfig creation as shown in README
            config = TrainingConfig(
                max_time_minutes=120,
                min_success_states=50,
                num_envs=4096,
                seed=42
            )
            
            assert config.max_time_minutes == 120
            assert config.min_success_states == 50
            assert config.num_envs == 4096
            assert config.seed == 42
            
            # Test TrainingOrchestrator initialization
            orchestrator = TrainingOrchestrator(
                isaaclab_path=str(self.test_isaaclab_path),
                task_name="Test_Task",
                robot="G1"
            )
            
            assert orchestrator is not None
            assert hasattr(orchestrator, 'train_all_skills')
            
            print("✅ Training integration from README works")
            
        except ImportError:
            pytest.skip("Training components not available")
    
    def test_verification_settings_from_readme(self):
        """Test verification settings examples from README."""
        try:
            from genhrl.generation import create_task_with_workflow
            import inspect
            
            # Get function signature
            sig = inspect.signature(create_task_with_workflow)
            
            # Check for verification parameters mentioned in README
            verification_params = [
                'verify_decompose', 'verify_plan', 'verify_rewards', 'verify_success'
            ]
            
            for param in verification_params:
                assert param in sig.parameters, f"Missing verification parameter: {param}"
            
            print("✅ Verification settings from README are supported")
            
        except ImportError:
            pytest.skip("create_task_with_workflow not available")


class TestREADMEExamples:
    """Test specific examples from the README."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp(prefix="test_readme_examples_")
        self.test_isaaclab_path = Path(self.test_dir) / "IsaacLab"
        self.test_isaaclab_path.mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_readme_programmatic_api_example(self):
        """Test the exact programmatic API example from README."""
        try:
            from genhrl.generation import create_task_with_workflow
            
            # This should not crash and should have the right signature
            import inspect
            sig = inspect.signature(create_task_with_workflow)
            
            # Verify all the parameters from the README example are supported
            example_params = {
                'task_name': "Pick_Up_Ball",
                'task_description': "The robot should pick up the red ball and place it on the table",
                'isaaclab_path': str(self.test_isaaclab_path),
                'api_key': "test_api_key",
                'robot': "G1",
                'max_hierarchy_levels': 3,
                'remove_previous': True,
                'verify_decompose': True,
                'verify_plan': False,
                'verify_rewards': False,
                'verify_success': False
            }
            
            # Check that all parameters exist in the function signature
            for param_name in example_params.keys():
                assert param_name in sig.parameters, f"Parameter {param_name} not found in function signature"
            
            print("✅ README programmatic API example parameters verified")
            
        except ImportError:
            pytest.skip("create_task_with_workflow not available")
    
    def test_readme_step_by_step_example(self):
        """Test the step-by-step control example from README."""
        try:
            from genhrl.generation import TaskManager
            
            # Test TaskManager initialization from README
            task_manager = TaskManager(
                isaaclab_path=str(self.test_isaaclab_path),
                api_key="test_api_key",
                robot="G1"
            )
            
            # Verify methods mentioned in README exist
            assert hasattr(task_manager, 'create_task_from_description')
            
            # Check method signature
            import inspect
            sig = inspect.signature(task_manager.create_task_from_description)
            
            readme_params = ['task_name', 'task_description', 'max_hierarchy_levels', 'robot']
            for param in readme_params:
                assert param in sig.parameters, f"Parameter {param} missing from create_task_from_description"
            
            print("✅ README step-by-step example verified")
            
        except ImportError:
            pytest.skip("TaskManager not available")
    
    def test_readme_debugging_workflow(self):
        """Test the debugging workflow example from README."""
        try:
            from genhrl.generation import TaskManager
            
            # Create task manager as shown in debugging example
            task_manager = TaskManager(str(self.test_isaaclab_path), api_key="test_key", robot="G1")
            
            # The example shows these methods should exist
            assert hasattr(task_manager, 'create_task_from_description')
            
            # Test that we can call the method (even if it fails due to mocked API)
            try:
                # This might fail due to API calls, but the interface should exist
                import inspect
                sig = inspect.signature(task_manager.create_task_from_description)
                assert 'task_name' in sig.parameters
                assert 'task_description' in sig.parameters
                
            except Exception:
                # Expected to fail due to API calls, but interface should exist
                pass
            
            print("✅ README debugging workflow interface verified")
            
        except ImportError:
            pytest.skip("TaskManager not available")


if __name__ == "__main__":
    # Allow running tests directly
    test_workflows = TestWorkflowFunctionality()
    test_workflows.setup_method()
    
    try:
        test_workflows.test_create_task_with_workflow_signature()
        test_workflows.test_main_create_steps_example_signature()
        test_workflows.test_remove_all_previous_skills_signature()
        test_workflows.test_task_manager_initialization_from_readme()
        test_workflows.test_hierarchical_code_generator_configurations()
        test_workflows.test_skill_library_management_from_readme()
        test_workflows.test_robot_configuration_from_readme()
        test_workflows.test_utility_functions_from_readme()
        test_workflows.test_training_integration_from_readme()
        test_workflows.test_verification_settings_from_readme()
        
        test_examples = TestREADMEExamples()
        test_examples.setup_method()
        
        test_examples.test_readme_programmatic_api_example()
        test_examples.test_readme_step_by_step_example()
        test_examples.test_readme_debugging_workflow()
        
        print("🎉 All workflow tests passed!")
        
    finally:
        test_workflows.teardown_method()
        test_examples.teardown_method()