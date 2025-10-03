import importlib
import types
from unittest import mock
import torch


def test_success_state_cache_clears_and_resets():
    """Ensure preload_success_states clears cache when switching skills."""
    events = importlib.import_module("genhrl.generation.mdp.events")

    # Ensure a clean start
    events._preloaded_success_states.clear()
    events._current_skill_name = None

    # Populate dummy cache for skill_one
    dummy_state = {
        "robot": {"dummy_body": {"pos": torch.zeros(3)}}
    }
    events._preloaded_success_states.append(dummy_state)
    events._current_skill_name = "skill_one"

    # Monkey-patch filesystem interactions so preload returns early without error
    with mock.patch("pathlib.Path.exists", return_value=False), \
         mock.patch("pathlib.Path.iterdir", side_effect=FileNotFoundError), \
         mock.patch("pathlib.Path.glob", return_value=[]):
        events.preload_success_states(device=torch.device("cpu"), skill_name="skill_two", max_states_total=10)

    # Cache should have been cleared because skill changed and no files loaded
    assert len(events._preloaded_success_states) == 0
    assert events._current_skill_name in (None, "skill_two")
    # Loading flag should not be stuck
    assert events._loading_in_progress is False