"""
Hierarchical Code Generator for the iGen framework.

This module contains the LLM-powered code generation system that automatically
creates task hierarchies, reward functions, and success criteria from natural
language descriptions.
"""

import os
import json
import re
import time
import random
import datetime
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

# Prefer the new unified google-genai SDK (>=1.0).  If it's not installed we
# raise a clear ImportError later during initialisation.
try:
    from google import genai as genai_modern  # type: ignore
    from google.genai import types as genai_types  # type: ignore
except ImportError:  # pragma: no cover – handled at runtime
    genai_modern = None  # type: ignore
    genai_types = None  # type: ignore

# Support OpenAI-compatible client for Gemini (https://ai.google.dev/gemini-api/docs/openai)
try:
    from openai import OpenAI as OpenAIClient  # type: ignore
except ImportError:  # pragma: no cover – handled at runtime
    OpenAIClient = None  # type: ignore


@dataclass
class TaskNode:
    name: str
    description: str
    children: List['TaskNode']
    requires: List[str]
    parent: Optional['TaskNode'] = None


class HierarchicalCodeGenerator:
    """
    LLM-powered code generator for hierarchical RL tasks.
    
    This class handles the entire workflow of converting natural language task
    descriptions into executable hierarchical RL environments, including:
    - Scene planning and object configuration
    - Task decomposition into hierarchical skills
    - Reward function generation with planning and verification
    - Success criteria creation with planning and verification
    - Complete project management and file generation
    """
    
    def __init__(self, 
                 api_key: str,
                 provider: str = "google",
                 model: str = "gemini-2.5-pro",
                 backup_model: str = "gemini-2.5-flash",
                 model_big: Optional[str] = None,
                 verify_decompose: bool = True,
                 verify_plan: bool = False,
                 verify_rewards_enabled: bool = False,
                 verify_success_enabled: bool = False,
                 skills_path: Optional[str] = None):
        """
        Initialize the code generator.
        
        Args:
            api_key: API key for the LLM provider
            provider: LLM provider ("google" or "anthropic")
            model: Primary model name
            backup_model: Backup model name for fallbacks
            model_big: Large model name for complex tasks
            verify_decompose: Enable decomposition verification
            verify_plan: Enable plan verification
            verify_rewards: Enable reward verification
            verify_success: Enable success criteria verification
            skills_path: Optional path to skills directory for skill library
        """
        self.api_key = api_key
        self.provider = provider
        self.model = model
        self.backup_model = backup_model
        self.model_big = model_big or model
        
        # Verification flags
        self.verify_decompose = verify_decompose
        self.verify_plan = verify_plan
        self.verify_rewards_enabled = verify_rewards_enabled
        self.verify_success_enabled = verify_success_enabled
        
        # Initialize skill library if skills_path is provided
        self.skill_library = None
        if skills_path:
            from .skill_library import SkillLibrary
            self.skill_library = SkillLibrary(str(skills_path))
        
        # Prepare OpenAI client placeholder
        self._openai_client = None
        
        # Initialize clients
        if provider == "anthropic":
            if Anthropic is None:
                raise ImportError("anthropic package not installed. Install with: pip install anthropic")
            self.anthropic_client = Anthropic(api_key=api_key)
        elif provider == "google":
            try:
                # Unified SDK (google-genai >=1.0)
                self._genai_client = genai_modern.Client(api_key=api_key) if api_key else genai_modern.Client()

                # Also initialise OpenAI-compatible client if available – prefer this
                self._openai_client = None
                if OpenAIClient is not None:
                    try:
                        self._openai_client = OpenAIClient(
                            api_key=api_key,
                            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                        )
                    except Exception as _e:
                        # Fallback: leave as None if client init fails (e.g. wrong version)
                        self._openai_client = None

                # END openai client init
            except Exception as e:
                raise ImportError(
                    "google-genai package (>=1.0) is required. Install with: pip install google-genai"
                ) from e
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def call_llm(self, prompt: str, 
                 model: Optional[str] = None, 
                 max_tokens: int = 8000, 
                 temperature: float = 0.0,
                 max_retries: int = 10, 
                 initial_delay: float = 30.0, 
                 backoff_factor: float = 2.0) -> str:
        """
        Call the LLM with the given prompt and parameters.
        
        Args:
            prompt: The prompt to send to the model
            model: Model name (uses default if None)
            max_tokens: Maximum tokens in response
            temperature: Temperature setting
            max_retries: Maximum retry attempts
            initial_delay: Initial delay before retries
            backoff_factor: Multiplicative factor for delay
            
        Returns:
            Response text from the model
        """
        if model is None:
            model = self.model
            
        if self.provider == "google":
            return self._call_gemini(prompt, model, max_tokens, temperature, 
                                   max_retries, initial_delay, backoff_factor)
        else:
            return self._call_anthropic(prompt, model, max_tokens, temperature,
                                      max_retries, initial_delay, backoff_factor)
    
    def _call_gemini(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        max_retries: int,
        initial_delay: float,
        backoff_factor: float,
    ) -> str:
        """Call Google Gemini models using *only* the new `google-genai` SDK.

        We create the config objects exactly as shown in the official
        quick-start guide:

            from google import genai
            from google.genai import types

            client = genai.Client()
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents="…",
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=…)
                ),
            )

        A small retry/back-off loop is preserved, and we fall back to the
        *backup* model only once if the first call returns an empty text.
        """

        delay = initial_delay
        last_exception: Optional[Exception] = None

        # ---------------------------------------------
        # If an OpenAI-compatible client is available, use it instead of google-genai.
        # This follows the guidance at https://ai.google.dev/gemini-api/docs/openai
        # ---------------------------------------------

        if self._openai_client is not None:
            # Prepare messages structure – prepend system instruction
            system_msg = {
                "role": "system",
                "content": (
                    "Always provide direct responses without markdown formatting. "
                    "When outputting code or JSON, provide only the raw content without using "
                    "backticks, code blocks, or any other markdown syntax."
                ),
            }

            user_msg = {"role": "user", "content": prompt}

            def _reasoning_effort_for(m: str) -> Optional[str]:
                if "flash-lite" in m or "flash-light" in m or ("flash" in m and "light" in m):
                    return "none"
                if "flash" in m:
                    return "low"
                if m.endswith("pro"):
                    return "high"
                return None

            for current_model in (model, self.backup_model):
                for attempt in range(max_retries + 1):
                    try:
                        # Build request kwargs
                        request_kwargs: Dict[str, Any] = {
                            "model": current_model,
                            "messages": [system_msg, user_msg],
                            "temperature": temperature,
                            "top_p": 0.1,
                        }

                        reff = _reasoning_effort_for(current_model)
                        if reff is not None:
                            request_kwargs["reasoning_effort"] = reff

                        response = self._openai_client.chat.completions.create(**request_kwargs)

                        if (
                            hasattr(response, "choices")
                            and response.choices
                            and hasattr(response.choices[0].message, "content")
                        ):
                            text = response.choices[0].message.content
                            if text:
                                return text.strip()

                        raise ValueError("Empty response from model")

                    except Exception as e:
                        last_exception = e
                        if attempt < max_retries:
                            sleep_time = delay * random.uniform(0.8, 1.2)
                            print(
                                f"Gemini (OpenAI compat) call to {current_model} failed: {e}. "
                                f"Retrying in {sleep_time:.1f}s…",
                                file=sys.stderr,
                                flush=True,
                            )
                            time.sleep(sleep_time)
                            delay *= backoff_factor
                        else:
                            break  # give up this model

                # reset delay before backup
                delay = initial_delay

            # If we reach here attempts exhausted, raise later

        # Compose prompt for google-genai fallback (with system instruction)
        enhanced_prompt = (
            "SYSTEM: Always provide direct responses without markdown formatting. "
            "When outputting code or JSON, provide only the raw content without using "
            "backticks, code blocks, or any other markdown syntax.\n\nUSER QUERY:\n" + prompt
        )

        # ---------------------------------------------
        # Fallback to google-genai SDK path (original implementation)
        # ---------------------------------------------

        # Helper to pick thought budget
        def _thought_budget_for(m: str) -> Optional[int]:
            if "flash-lite" in m or "flash-light" in m or ("flash" in m and "light" in m):
                return 0
            if "flash" in m:
                return 2000
            if m.endswith("pro"):
                return 8000
            return None  # Default – let server decide

        # We try the primary model first, then optionally one fallback
        for current_model in (model, self.backup_model):
            for attempt in range(max_retries + 1):
                try:
                    generation_cfg = genai_types.GenerationConfig(
                        temperature=temperature,
                        top_p=0.1,
                        top_k=40,
                    )

                    tb = _thought_budget_for(current_model)
                    thinking_cfg_candidate = (
                        genai_types.ThinkingConfig(thinking_budget=tb) if tb is not None else None
                    )

                    try:
                        # Build request config including thinking config (preferred path)
                        request_cfg = genai_types.GenerateContentConfig(
                            generation_config=generation_cfg,
                            thinking_config=thinking_cfg_candidate,
                        )
                    except Exception:
                        # Installed SDK too old – rebuild without thinking config
                        request_cfg = genai_types.GenerateContentConfig(
                            generation_config=generation_cfg,
                        )

                    # --- make the model call ---------------------------------------------------
                    response = self._genai_client.models.generate_content(
                        model=current_model,
                        contents=enhanced_prompt,
                        config=request_cfg,
                    )

                    # New SDK: aggregated text is available via .text
                    if hasattr(response, "text") and response.text:
                        return response.text.strip()

                    # Fallback extraction for completeness
                    if (
                        hasattr(response, "candidates")
                        and response.candidates
                        and hasattr(response.candidates[0], "content")
                    ):
                        parts = response.candidates[0].content.parts  # type: ignore
                        text_parts = [p.text for p in parts if hasattr(p, "text")]
                        if text_parts:
                            return " ".join(text_parts).strip()

                    raise ValueError("Empty response from model")

                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        sleep_time = delay * random.uniform(0.8, 1.2)
                        print(
                            f"Gemini call to {current_model} failed: {e}. "
                            f"Retrying in {sleep_time:.1f}s…",
                            file=sys.stderr,
                            flush=True,
                        )
                        time.sleep(sleep_time)
                        delay *= backoff_factor
                    else:
                        break  # Give up on this model, maybe try backup

            # If we reach here after attempts loop -> failed for current_model
            delay = initial_delay  # reset delay before trying backup

        # All attempts exhausted
        raise last_exception or RuntimeError("Gemini API call failed after retries")
    
    def _call_anthropic(self, prompt: str, model: str, max_tokens: int, temperature: float,
                       max_retries: int, initial_delay: float, backoff_factor: float) -> str:
        """Call Anthropic Claude API with backoff."""
        delay = initial_delay
        last_exception = None
        
        system_message = "Always provide direct responses without markdown formatting. When outputting code or JSON, provide only the raw content without using backticks, code blocks, or any other markdown syntax."
        
        for attempt in range(max_retries + 1):
            try:
                response = self.anthropic_client.messages.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_message
                )
                
                # Extract text content
                if hasattr(response, 'content') and response.content:
                    text_parts = []
                    for block in response.content:
                        if hasattr(block, 'text'):
                            text_parts.append(block.text)
                        elif isinstance(block, dict) and 'text' in block:
                            text_parts.append(block['text'])
                    if text_parts:
                        return " ".join(text_parts)
                
                raise ValueError("Could not extract text from response")
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    jitter = random.uniform(0.8, 1.2)
                    sleep_time = delay * jitter
                    print(
                        f"API call failed (attempt {attempt+1}/{max_retries+1}): {e}. "
                        f"Retrying after {sleep_time:.2f}s...",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(sleep_time)
                    delay *= backoff_factor
                else:
                    raise last_exception
        
        raise last_exception or RuntimeError("API call failed")
    
    def strip_code_blocks(self, response_text: str, language: str = "python") -> str:
        """Strip markdown code blocks from response."""
        if response_text is None:
            return ""
        
        block_marker = f"```{language}"
        if block_marker in response_text:
            start_idx = response_text.find(block_marker) + len(block_marker)
            end_idx = response_text.rfind("```")
            
            if start_idx != -1 and end_idx != -1:
                return response_text[start_idx:end_idx].strip()
        
        return response_text.strip()
    
    def extract_json_from_response(self, response_text: str) -> str:
        """Extract and clean JSON from LLM response."""
        if response_text is None:
            raise ValueError("Response text cannot be None")
        
        response_text = response_text.strip()
        
        # Remove code block markers
        code_block_markers = ["```json", "```python", "```", "json", "python"]
        for marker in code_block_markers:
            if response_text.startswith(marker):
                response_text = response_text[len(marker):].strip()
        
        response_text = response_text.replace("```", "")
        
        # Clean common JSON issues
        response_text = re.sub(r',\s*([}\]])', r'\1', response_text)
        
        return response_text.strip()
    
    def verify_code_compliance(self, generated_code: str, original_prompt: str, 
                              context: str = "", response_format: str = "code") -> str:
        """
        Verify that generated code complies with all requirements and avoids common errors.
        
        Args:
            generated_code: The code generated by the LLM
            original_prompt: The prompt that was used to generate the code
            context: Additional context about the task (optional)
            response_format: Expected format of the response ("code" or "json")
            
        Returns:
            The verified/improved code
        """
        verification_prompt = f"""
# CODE VERIFICATION
        
## GENERATED CODE
```python
{generated_code}
```

## ORIGINAL TASK
{context}

## ORIGINAL PROMPT
{original_prompt}

## VERIFICATION INSTRUCTIONS
Please verify that the generated code:
1. Precisely follows all the instructions and design requirements in the original prompt. You must go step-by-step through the prompt and ensure that all requirements are met.
2. Contains no syntax errors or bugs
3. Uses correct data access patterns
4. Avoids common pitfalls and errors
5. Is complete and doesn't have any TODO or placeholder comments
6. Follows best practices

If everything is correct, respond with the same exact code.
If there are issues, please add python comments detailing the error and then and fix them. Provide the corrected code WITHOUT any explanations, markdown formatting, or code block markers.
Respond ONLY with the corrected code.
"""
        
        # Call the LLM with the verification prompt
        verified_code = self.call_llm(verification_prompt)
        
        # Strip markdown formatting if needed
        if response_format == "code":
            verified_code = self.strip_code_blocks(verified_code, "python")
        elif response_format == "json":
            verified_code = self.extract_json_from_response(verified_code)
        
        return verified_code
    
    def plan_task_execution(self, task_name: str, task_description: str, max_hierarchy_levels: int = 3, object_name_mapping: str = "") -> str:
        """Generate a detailed sequence of events required for the robot to complete the task."""
        try:
            from .prompts.plan_task_execution import plan_task_execution_prompt
            prompt = plan_task_execution_prompt(task_name, task_description, max_hierarchy_levels, object_name_mapping)
        except ImportError:
            prompt = f"Create a detailed plan for the robot to execute this task: {task_name}\nDescription: {task_description}"
        
        response = self.call_llm(prompt)
        initial_plan = response if response else ""
        print("\nInitial Task Execution Plan:")
        print(initial_plan)
        
        # If verification is enabled, verify and potentially refine the plan
        if self.verify_plan:
            # We'll create a temporary, minimal object config for verification if none exists yet
            temp_objects_config = """{"objects": []}"""
            try:
                # Try to generate a preliminary object config for verification
                scene_plan = self.plan_scene_construction(task_description)
                if scene_plan:  # Ensure scene_plan is not None
                    temp_response = self.generate_objects_config(task_description)
                    if temp_response:  # Ensure temp_response is not None
                        temp_json_response = self.extract_json_from_response(temp_response)
                        # If valid JSON, use it
                        json.loads(temp_json_response)
                        if temp_json_response.startswith('{') and temp_json_response.endswith('}'):
                            temp_objects_config = temp_json_response
            except Exception as e:
                print(f"Using minimal object config for plan verification: {str(e)}")
            
            refined_plan = self.verify_plan_implementation(task_name, task_description, initial_plan, temp_objects_config)
            return refined_plan
        
        return initial_plan
    
    def verify_plan_implementation(self, task_name: str, task_description: str, 
                                 current_plan: str, objects_config: str) -> str:
        """Verify and potentially refine the task execution plan."""
        try:
            from .prompts.verify_plan import verify_plan_prompt
            prompt = verify_plan_prompt(task_name, task_description, current_plan, objects_config)
            
            response = self.call_llm(prompt)
            response = self.strip_code_blocks(response)
            
            verification_result = response
            print("\nExecution Plan Verification Result:")
            print(verification_result)
            
            # Check if refinement is needed
            if "NEEDS REFINEMENT" in verification_result:
                print("\nPlan refinement needed. Extracting refined execution plan...")
                # Extract the refined execution plan from the response
                start_marker = "## Refined Execution Plan"
                if start_marker in verification_result:
                    refined_plan = verification_result.split(start_marker)[1].strip()
                    print("\nRefined execution plan extracted successfully:")
                    print(refined_plan)
                    return refined_plan
                else:
                    print("No refined execution plan found in verification result.")
                    return current_plan
            else:
                print("\nCurrent execution plan passes verification. Continuing with original plan.")
                return current_plan
                
        except Exception as e:
            print(f"Error in execution plan verification: {str(e)}")
            # Return the original plan if verification fails
            return current_plan
    
    def plan_scene_construction(self, task_description: str) -> str:
        """Generate a scene construction plan from task description."""
        try:
            from .prompts.plan_scene_construction import plan_scene_construction_prompt
            prompt = plan_scene_construction_prompt(task_description)
        except ImportError:
            prompt = f"Plan the scene construction for this task: {task_description}"
        
        response = self.call_llm(prompt, model=self.model_big)
        response = self.strip_code_blocks(response)
        
        scene_plan = response
        print("\nScene Construction Plan:")
        print(scene_plan)
        return scene_plan
    
    def generate_objects_config(self, task_description: str) -> str:
        """Generate object configuration JSON from task description."""
        # First generate the scene construction plan
        print("planning the scene")
        scene_plan = self.plan_scene_construction(task_description)
        
        try:
            from .prompts.generate_objects_config import generate_objects_config_prompt
            prompt = generate_objects_config_prompt(scene_plan, task_description)
        except ImportError:
            prompt = f"Generate object configuration JSON for this task: {task_description}"
        
        response = self.call_llm(prompt, model=self.model_big)
        print(f"\nRaw LLM response for objects config: {response[:500]}...")  # Debug print
        
        # Extract just the JSON content
        json_response = self.extract_json_from_response(response)
        print(f"\nExtracted JSON response: {json_response[:200]}...")  # Debug print
        
        # Check if json_response is empty
        if not json_response or not json_response.strip():
            print("Warning: Empty JSON response. Creating fallback object config.")
            json_response = '{"objects": []}'
        
        # Verify code compliance for JSON format
        context_info = f"Task Description: {task_description}\nScene Plan: {scene_plan}"
        json_response = self.verify_code_compliance(json_response, prompt, 
                                                   context=context_info, 
                                                   response_format="json")
        
        # Additional validation
        # Try to parse it to verify it's valid JSON
        try:
            json.loads(json_response)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Problematic JSON: {json_response}")
            # Fallback to simple config
            json_response = '{"objects": []}'
        
        # If it parsed successfully and has the proper JSON structure, return it
        if json_response.startswith('{') and json_response.endswith('}'):
            return json_response
        else:
            print(f"Warning: Extracted JSON doesn't start and end with braces. Attempting to fix...")
            start_idx = json_response.find('{')
            end_idx = json_response.rfind('}')
            if start_idx != -1 and end_idx != -1:
                return json_response[start_idx:end_idx + 1]
            else:
                raise ValueError("Could not find valid JSON structure in response")
    
    def decompose_task(self, task_name: str, task_description: str, 
                      object_config_path: str, objects_config: str, 
                      skill_library: Optional[Dict] = None, max_hierarchy_levels: int = 3, object_name_mapping: str = "") -> Dict:
        """Decompose task into hierarchical skills."""
        # First, generate a detailed plan for the robot to execute the task
        task_plan = self.plan_task_execution(task_name, task_description, max_hierarchy_levels, object_name_mapping)
        
        if skill_library is None:
            skill_library = {}
        
        try:
            from .prompts.decompose_task import decompose_task_prompt
            prompt = decompose_task_prompt(task_plan, task_name, task_description, 
                                         object_config_path, objects_config, skill_library, max_hierarchy_levels, object_name_mapping)
        except ImportError:
            prompt = f"""
Decompose this task into hierarchical skills with a maximum of {max_hierarchy_levels} levels:
Task: {task_name}
Description: {task_description}
Objects: {objects_config}

Return a JSON hierarchy with 'name', 'description', and 'children' fields.
"""
        
        response = self.call_llm(prompt, model=self.model_big)
        
        # Parse the hierarchical skills structure
        json_response = self.extract_json_from_response(response)

        # Verify the decomposition JSON
        context_info = f"Task: {task_name} - {task_description}\nTask Plan: {task_plan}"
        json_response = self.verify_code_compliance(json_response, prompt, 
                                                  context=context_info, 
                                                  response_format="json")

        print("json_response debug: \n", json_response)
        
        skills_hierarchy = json.loads(json_response)
        
        # Validate the skills hierarchy structure
        if not isinstance(skills_hierarchy, dict):
            raise TypeError(f"skills_hierarchy must be a dictionary, got {type(skills_hierarchy)}")
        
        # Ensure required keys exist based on hierarchy level
        if max_hierarchy_levels == 1:
            # For single level, only name and description are required
            required_keys = ["name", "description"]
            missing_keys = [key for key in required_keys if key not in skills_hierarchy]
            if missing_keys:
                raise ValueError(f"Skills hierarchy missing required keys: {missing_keys}")
            
            # Ensure no children are present for single level
            if "children" in skills_hierarchy and skills_hierarchy["children"]:
                print(f"Warning: Found children in max_hierarchy_levels=1 task, removing them.")
                skills_hierarchy.pop("children", None)
        else:
            # For multi-level hierarchy, all keys including children are required
            required_keys = ["name", "description", "children"]
            missing_keys = [key for key in required_keys if key not in skills_hierarchy]
            if missing_keys:
                raise ValueError(f"Skills hierarchy missing required keys: {missing_keys}")
            
            # Validate children structure
            if not isinstance(skills_hierarchy.get("children", []), list):
                raise TypeError(f"The 'children' field must be a list, got {type(skills_hierarchy.get('children', []))}")
            
            # Ensure all children have the required structure
            for i, child in enumerate(skills_hierarchy.get("children", [])):
                if not isinstance(child, dict):
                    raise TypeError(f"Child at index {i} must be a dictionary, got {type(child)}")
                
                child_missing_keys = [key for key in required_keys if key not in child]
                if child_missing_keys:
                    print(f"Child at index {i} missing keys: {child_missing_keys}. Assuming no children.")
                    child["children"] = []  # Assume no children if key is missing
        
        print("Initial hierarchical skills generated:")
        print(json.dumps(skills_hierarchy, indent=2))
        
        # If verification is enabled, verify and potentially refine the decomposition
        if self.verify_decompose:
            skills_hierarchy = self.verify_decompose_implementation(task_name, task_description, task_plan, 
                                                                  skills_hierarchy, object_config_path, objects_config, max_hierarchy_levels)
            
        print("Final hierarchical skills:")
        print(json.dumps(skills_hierarchy, indent=2))
        return skills_hierarchy
    
    def verify_decompose_implementation(self, task_name: str, task_description: str, 
                                      task_plan: str, skills: Dict, object_config_path: str, 
                                      objects_config: str, max_hierarchy_levels: int = 3) -> Dict:
        """Verify and potentially refine the linear sequence of skills."""
        try:
            from .prompts.verify_decompose import verify_decompose_prompt
            prompt = verify_decompose_prompt(task_name, task_description, task_plan, skills, 
                                           object_config_path, objects_config, max_hierarchy_levels)
            
            response = self.call_llm(prompt, model=self.model_big)
            response = self.strip_code_blocks(response)
            
            verification_result = response
            print("\nDecomposition Verification Result:")
            print(verification_result)
            
            # Check if refinement is needed
            if "NEEDS REFINEMENT" in verification_result:
                print("\nRefinement needed. Extracting refined skills...")
                # Extract the refined skill sequence JSON from the response
                start_marker = "## Refined Linear Skill Structure (if needed):"
                if start_marker in verification_result:
                    json_section = verification_result.split(start_marker)[1].strip()
                    # Extract JSON array
                    json_text = self.extract_json_from_response(json_section)
                    try:
                        refined_skills = json.loads(json_text)
                        print("\nRefined skills extracted successfully:")
                        print(json.dumps(refined_skills, indent=2))
                        return refined_skills
                    except json.JSONDecodeError as e:
                        print(f"Error parsing refined skills JSON: {e}")
                        print(f"Raw JSON section: {json_text}")
                        # Return original skills if we can't parse the refinement
                        return skills
                else:
                    print("No refined skill sequence found in verification result.")
                    return skills
            else:
                print("\nCurrent decomposition passes verification. Continuing with original skills.")
                return skills
                
        except Exception as e:
            print(f"Error in decomposition verification: {str(e)}")
            # Return the original skills if verification fails
            return skills
    
    def generate_task_rewards(self, task_description: str, objects_config: str,
                            skill_name: str, skill_description: str,
                            all_skills: List[str], all_skills_descriptions: List[str],
                            objects_mapping: str = "") -> str:
        """Generate reward function code for a specific skill with planning and verification."""
        try:
            next_skill_name = all_skills[all_skills.index(skill_name) + 1]
            next_skill_description = all_skills_descriptions[all_skills.index(skill_name) + 1]
        except IndexError:
            next_skill_name = "No Next Skill"
            next_skill_description = "No Next Skill Description"
        
        # Planning first:
        try:
            from .prompts.planning_rewards import planning_rewards_prompt
            planning_prompt = planning_rewards_prompt(
                skill_name,
                skill_description,
                task_description,
                objects_config,
                next_skill_name,
                next_skill_description,
                objects_mapping,
            )
            response = self.call_llm(planning_prompt, model=self.model_big)
            response = self.strip_code_blocks(response)
            rewards_plan = response

            print(f"\nRewards Plan for {skill_name}:")
            print(rewards_plan)
        except ImportError:
            rewards_plan = f"Generate rewards for {skill_name}: {skill_description}"

        # Code generation:
        try:
            from .prompts.execute_rewards_function_creation import execute_rewards_function_creation_prompt
            execution_prompt = execute_rewards_function_creation_prompt(
                skill_name,
                skill_description,
                task_description,
                objects_config,
                rewards_plan,
                objects_mapping,
            )
        except ImportError:
            execution_prompt = f"""
Generate reward function code for skill '{skill_name}':
Description: {skill_description}
Task: {task_description}
Objects: {objects_config}
Plan: {rewards_plan}

Return Python code for the reward function.
"""
        
        response = self.call_llm(execution_prompt)
        response = self.strip_code_blocks(response)
        rewards_code = response
        
        # Verify code compliance
        context_info = f"Task: {task_description}\nSkill: {skill_name} - {skill_description}"
        rewards_code = self.verify_code_compliance(rewards_code, execution_prompt, context=context_info)
        
        if self.verify_rewards_enabled:
            # Verify the rewards
            verification_result = self.verify_rewards_implementation(skill_name, skill_description, rewards_code)
            
            # If verification indicates issues, regenerate with the feedback
            if "NEEDS IMPROVEMENT" in verification_result:
                improved_prompt = f"""
                # REWARD FUNCTION IMPROVEMENT

                ## PREVIOUS ATTEMPT
                You previously generated reward functions for skill: {skill_name}
                Description: {skill_description}

                Your code had issues that need to be fixed:
                ```python
                {rewards_code}
                ```
                
                ## VERIFICATION FEEDBACK
                {verification_result}
                
                ## REQUIREMENTS FOR NEW VERSION
                Please regenerate the reward functions, addressing ALL issues identified above.
                **You must respond with only the python code.**
                
                Remember:
                1. Create ONE main reward function (weight 1.0) directly measuring progress toward the primary goal
                2. Create MULTIPLE shaping rewards (weights totaling 0.5) to encourage helpful behaviors
                
                {execution_prompt}  # Use the execution prompt for regeneration
                """
                
                improved_response = self.call_llm(improved_prompt)
                improved_code = self.strip_code_blocks(improved_response)
                
                # Verify the improved code
                improved_code = self.verify_code_compliance(improved_code, improved_prompt, context=context_info)
                rewards_code = improved_code if improved_code else rewards_code
        
        print("\nReward Functions Generated:")
        print(rewards_code)
        
        return rewards_code
    
    def generate_success_criteria(self, task_description: str, objects_config: str,
                                skill_name: str, skill_description: str, skill_rewards: str,
                                all_skills: List[str], all_skills_descriptions: List[str],
                                objects_mapping: str = "") -> str:
        """Generate success criteria code for a specific skill with planning and verification."""
        try:
            next_skill_name = all_skills[all_skills.index(skill_name) + 1]
            next_skill_description = all_skills_descriptions[all_skills.index(skill_name) + 1]
        except IndexError:
            next_skill_name = "No Next Skill"
            next_skill_description = "No Next Skill Description"

        # Planning first:
        try:
            from .prompts.planning_success import planning_success_prompt
            planning_prompt = planning_success_prompt(
                skill_name,
                skill_description,
                task_description,
                objects_config,
                skill_rewards,
                next_skill_name,
                next_skill_description,
                objects_mapping,
            )
            response = self.call_llm(planning_prompt, model=self.model_big)
            response = self.strip_code_blocks(response)
            success_plan = response
        except ImportError:
            success_plan = f"Generate success criteria for {skill_name}: {skill_description}"

        # Code generation:
        try:
            from .prompts.execute_success_function_creation import execute_success_function_creation_prompt
            execution_prompt = execute_success_function_creation_prompt(
                skill_name,
                skill_description,
                task_description,
                objects_config,
                skill_rewards,
                success_plan,
                objects_mapping,
            )
        except ImportError:
            execution_prompt = f"""
Generate success criteria code for skill '{skill_name}':
Description: {skill_description}
Task: {task_description}
Objects: {objects_config}
Rewards: {skill_rewards}
Plan: {success_plan}

Return Python code for the success criteria function.
"""
        
        response = self.call_llm(execution_prompt)
        response = self.strip_code_blocks(response)
        success_code = response
        
        # Verify code compliance
        context_info = f"Task: {task_description}\nSkill: {skill_name} - {skill_description}"
        success_code = self.verify_code_compliance(success_code, execution_prompt, context=context_info)
        
        if self.verify_success_enabled:
            # Verify the success criteria
            verification_result = self.verify_success_criteria_implementation(skill_name, skill_description, success_code)
            
            # If verification indicates issues, regenerate with the feedback
            if "NEEDS IMPROVEMENT" in verification_result:
                improved_prompt = f"""
                # SUCCESS CRITERIA IMPROVEMENT

                ## PREVIOUS ATTEMPT
                You previously generated success criteria for skill: {skill_name}
                Description: {skill_description}

                Your code had issues that need to be fixed:
                ```python
                {success_code}
                ```
                
                ## VERIFICATION FEEDBACK
                {verification_result}
                
                ## REQUIREMENTS FOR NEW VERSION
                Please regenerate the success criteria, addressing ALL issues identified above.
                **You must respond with only the python code.**
                
                Remember:
                1. Success criteria should directly measure task completion
                2. Thresholds should be reasonable and justified
                3. Avoid arbitrary values without clear reasoning
                4. Criteria should use correct data access patterns
                
                {execution_prompt}  # Original prompt with all requirements
                """
                
                improved_response = self.call_llm(improved_prompt)
                improved_response = self.strip_code_blocks(improved_response)
                
                # Verify the improved code
                improved_code = self.verify_code_compliance(improved_response, improved_prompt, context=context_info)
                success_code = improved_code if improved_code else success_code
        
        print("\nSuccess Criteria Generated:")
        print(success_code)
        
        return success_code
    
    def verify_rewards_implementation(self, skill_name: str, skill_description: str, rewards_code: str) -> str:
        """Use a second LLM to verify the reward function and identify potential issues."""
        try:
            from .prompts.verify_rewards import verify_rewards_prompt
            prompt = verify_rewards_prompt(skill_name, skill_description, rewards_code)
        except ImportError:
            prompt = f"""
Verify and improve this reward function:
Skill: {skill_name}
Description: {skill_description}
Code: {rewards_code}

Return improved Python code.
"""
        
        response = self.call_llm(prompt, model=self.model_big)
        response = self.strip_code_blocks(response)
        
        verification_result = response

        print("\nReward Verification Results:")
        print(verification_result)

        return verification_result
    
    def verify_success_criteria_implementation(self, skill_name: str, skill_description: str, success_code: str) -> str:
        """Use multiple specialized LLM calls to verify the success criteria and identify potential issues."""
        try:
            from .prompts.verify_success import verify_success_prompt
            prompt = verify_success_prompt(skill_name, skill_description, success_code)
        except ImportError:
            prompt = f"""
Verify and improve this success criteria:
Skill: {skill_name}
Description: {skill_description}
Code: {success_code}

Return improved Python code.
"""
        
        response = self.call_llm(prompt, model=self.model_big)
        response = self.strip_code_blocks(response)
        verification_result = response

        print("\nSuccess Criteria Verification Results:")
        print(verification_result)

        return verification_result
    
    def generate_collapse_termination(self, task_description: str, objects_config: str, skill_name: str, 
                                     skill_description: str, skill_rewards: str, all_skills: List[str],
                                     objects_mapping: str = "") -> str:
        """Generate termination condition for when the robot deviates too far completion of the skill"""
        try:
            from .prompts.generate_collapse_termination import generate_collapse_termination_prompt
            prompt = generate_collapse_termination_prompt(
                skill_name,
                skill_description,
                skill_rewards,
                objects_config,
                objects_mapping,
            )
        except ImportError:
            prompt = f"""
Generate collapse termination condition for skill '{skill_name}':
Description: {skill_description}
Task: {task_description}
Rewards: {skill_rewards}

Return Python code for the collapse termination function.
"""
        
        response = self.call_llm(prompt)
        response = self.strip_code_blocks(response)
        collapse_code = response
        
        # Verify code compliance
        context_info = f"Task: {task_description}\nSkill: {skill_name} - {skill_description}"
        collapse_code = self.verify_code_compliance(collapse_code, prompt, context=context_info)
                    
        print("\nCollapse Termination Generated:")
        print(collapse_code)
        
        return collapse_code
    
    def verify_rewards(self, skill_name: str, skill_description: str, rewards_code: str) -> str:
        """Verify and improve reward function code."""
        return self.verify_rewards_implementation(skill_name, skill_description, rewards_code)
    
    def verify_success_criteria(self, skill_name: str, skill_description: str, success_code: str) -> str:
        """Verify and improve success criteria code."""
        return self.verify_success_criteria_implementation(skill_name, skill_description, success_code)
    
    def generate_gymnasium_registration(self, task_name: str, skill_name: str, robot: str = "G1", is_primitive: bool = False) -> str:
        """
        Generate gymnasium registration code for a specific skill.
        
        Args:
            task_name: Name of the task containing the skill
            skill_name: Name of the skill to register
            robot: Robot type (default: G1)
            is_primitive: Whether this is a primitive skill (affects config file choice)
            
        Returns:
            Gymnasium registration code as a string
        """
        # Convert skill name to title case for the environment ID
        skill_name_title = ''.join(word.capitalize() for word in skill_name.split('_'))
        skill_name_lower = skill_name.lower()
        
        # Choose the correct config file based on skill type
        config_file = "skrl_flat_ppo_cfg.yaml" if is_primitive else "skrl_ppo_cfg.yaml"
        
        registration_code = f"""
# Skill: {skill_name}
gym.register(
    id="Isaac-RobotFlat{skill_name_title}-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={{
        "env_cfg_entry_point": f"{{__name__}}.skills.{task_name}.skills.{skill_name}.{skill_name_lower}_cfg:RobotFlatEnvCfg",
        "skrl_cfg_entry_point": f"{{__name__}}.skills.{task_name}.skills.{skill_name}.agents:{config_file}",
    }},
)

gym.register(
    id="Isaac-RobotRough{skill_name_title}-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={{
        "env_cfg_entry_point": f"{{__name__}}.skills.{task_name}.skills.{skill_name}.{skill_name_lower}_cfg:RobotRoughEnvCfg",
        "skrl_cfg_entry_point": f"{{__name__}}.skills.{task_name}.skills.{skill_name}.agents:{config_file}",
    }},
)
"""
        return registration_code
    
    def generate_hierarchical_task_registration(self, task_name: str, robot: str = "G1") -> str:
        """
        Generate gymnasium registration code for the hierarchical task itself.
        
        Args:
            task_name: Name of the hierarchical task
            robot: Robot type (default: G1)
            
        Returns:
            Gymnasium registration code for the hierarchical task
        """
        task_name_title = ''.join(word.capitalize() for word in task_name.split('_'))
        task_name_lower = task_name.lower()
        
        registration_code = f"""
# Hierarchical Task: {task_name}
gym.register(
    id="Isaac-RobotHierarchical{task_name_title}-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={{
        "env_cfg_entry_point": f"{{__name__}}.tasks.{task_name}.{task_name_lower}_cfg:RobotHierarchicalEnvCfg",
        "": f"{{__name__}}.tasks.{task_name}.agents:skrl_hierarchical_ppo_cfg.yaml",
    }},
)
"""
        return registration_code
    
    def update_gymnasium_registrations(self, init_file_path: str, task_name: str, 
                                     skill_names: List[str], robot: str = "G1", 
                                     skill_library=None) -> None:
        """
        Update or create gymnasium registrations in the __init__.py file.
        
        Args:
            init_file_path: Path to the __init__.py file where registrations should be added
            task_name: Name of the task
            skill_names: List of skill names to register
            robot: Robot type (default: G1)
            skill_library: Optional skill library to get skill type information
        """
        from pathlib import Path
        
        init_path = Path(init_file_path)
        
        # Read existing content if file exists
        existing_content = ""
        if init_path.exists():
            with open(init_path, 'r') as f:
                existing_content = f.read()
        
        # Create base imports if file doesn't exist or is missing imports
        if not existing_content or 'import gymnasium as gym' not in existing_content:
            base_imports = '''# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Generated environments registration for {}.
""".format(task_name)

import gymnasium as gym

##
# Register Generated Environments
##
'''
            existing_content = base_imports + "\n"
        
        # Generate registration code for all skills
        registrations_to_add = ""
        
        for skill_name in skill_names:
            # Check if this skill is already registered
            skill_name_title = ''.join(word.capitalize() for word in skill_name.split('_'))
            flat_id = f"Isaac-RobotFlat{skill_name_title}-v0"
            rough_id = f"Isaac-RobotRough{skill_name_title}-v0"
            
            if f'id="{flat_id}"' not in existing_content and f'id="{rough_id}"' not in existing_content:
                # Determine if skill is primitive
                is_primitive = False
                if skill_library and hasattr(skill_library, 'get_skill_info'):
                    skill_info = skill_library.get_skill_info(skill_name)
                    if skill_info:
                        is_primitive = skill_info.get('is_primitive', False)
                
                registration_code = self.generate_gymnasium_registration(task_name, skill_name, robot, is_primitive)
                registrations_to_add += registration_code
                print(f"✅ Adding gymnasium registration for {'primitive' if is_primitive else 'composite'} skill: {skill_name}")
            else:
                print(f"⚠️ Skill '{skill_name}' already registered, skipping")
        
        # Add hierarchical task registration
        task_name_title = ''.join(word.capitalize() for word in task_name.split('_'))
        hierarchical_id = f"Isaac-RobotHierarchical{task_name_title}-v0"
        
        if f'id="{hierarchical_id}"' not in existing_content:
            task_registration = self.generate_hierarchical_task_registration(task_name, robot)
            registrations_to_add += task_registration
            print(f"✅ Adding gymnasium registration for hierarchical task: {task_name}")
        
        # Write the updated content
        if registrations_to_add:
            # Ensure the directory exists
            init_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(init_path, 'w') as f:
                f.write(existing_content + registrations_to_add)
            
            print(f"✅ Updated gymnasium registrations in: {init_path}")
        else:
            print(f"ℹ️ No new registrations needed for: {init_path}")
    
    def extract_skill_names_from_hierarchy(self, skills_hierarchy: Dict) -> List[str]:
        """
        Extract all skill names from a hierarchical skills structure.
        
        Args:
            skills_hierarchy: The hierarchical skills dictionary
            
        Returns:
            List of all skill names in the hierarchy
        """
        skill_names = []
        
        def extract_names_recursive(node: Dict):
            if 'name' in node:
                skill_names.append(node['name'])
            
            if 'children' in node and isinstance(node['children'], list):
                for child in node['children']:
                    extract_names_recursive(child)
        
        extract_names_recursive(skills_hierarchy)
        return skill_names
    
    def register_generated_skills(self, task_name: str, skills_hierarchy: Dict, 
                                skills_base_path: str, robot: str = "G1", skill_library=None) -> None:
        """
        Automatically register all skills from a decomposed task hierarchy with gymnasium.
        
        Args:
            task_name: Name of the task
            skills_hierarchy: The hierarchical skills dictionary from decompose_task
            skills_base_path: Base path where skills are generated 
            robot: Robot type (default: G1)
            skill_library: Optional skill library to get skill type information
        """
        from pathlib import Path
        
        # Extract all skill names from the hierarchy
        skill_names = self.extract_skill_names_from_hierarchy(skills_hierarchy)
        
        if not skill_names:
            print("⚠️ No skills found in hierarchy to register")
            return
        
        print(f"🔧 Registering {len(skill_names)} skills with gymnasium: {skill_names}")
        
        # Determine the path to the __init__.py file where registrations should go
        # This should be in the skills package directory
        skills_package_path = Path(skills_base_path)
        init_file_path = skills_package_path / "__init__.py"
        
        # Update the gymnasium registrations
        self.update_gymnasium_registrations(
            str(init_file_path), 
            task_name, 
            skill_names, 
            robot,
            skill_library
        )
        
        print(f"✅ Successfully registered all skills for task '{task_name}' with gymnasium")
    
    def decompose_task_with_registration(self, task_name: str, task_description: str, 
                                       object_config_path: str, objects_config: str, 
                                       skills_base_path: str, robot: str = "G1",
                                       skill_library: Optional[Dict] = None, 
                                       max_hierarchy_levels: int = 3,
                                       auto_register: bool = True, object_name_mapping: str = "") -> Dict:
        """
        Decompose task into hierarchical skills and optionally register with gymnasium.
        
        Args:
            task_name: Name of the task to decompose
            task_description: Description of the task
            object_config_path: Path to object configuration file
            objects_config: Object configuration content
            skills_base_path: Base path where skills will be generated
            robot: Robot type (default: G1)
            skill_library: Optional existing skill library
            max_hierarchy_levels: Maximum hierarchy levels for decomposition
            auto_register: Whether to automatically register with gymnasium (default: True)
            
        Returns:
            Hierarchical skills dictionary
        """
        # Call the original decompose_task method
        skills_hierarchy = self.decompose_task(
            task_name=task_name,
            task_description=task_description,
            object_config_path=object_config_path,
            objects_config=objects_config,
            skill_library=skill_library,
            max_hierarchy_levels=max_hierarchy_levels,
            object_name_mapping=object_name_mapping
        )
        
        # Automatically register with gymnasium if requested
        if auto_register:
            try:
                self.register_generated_skills(
                    task_name=task_name,
                    skills_hierarchy=skills_hierarchy,
                    skills_base_path=skills_base_path,
                    robot=robot
                )
            except Exception as e:
                print(f"⚠️ Warning: Could not auto-register skills with gymnasium: {e}")
                print("You may need to manually call register_generated_skills() later")
        
        return skills_hierarchy