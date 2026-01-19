"""
This module translates high-level intent into a machine-readable goal for the DRL agent.
"""

import re

def translate_intent(intent_string: str) -> dict:
    """
    Parses a simple intent string to extract a goal for the agent.

    Args:
        intent_string: A string like "Create a robust link from A to D".

    Returns:
        A dictionary defining the goal, e.g., {'start_node': 'A', 'end_node': 'D'}.
    """
    # Use regex to find the start and end nodes. This is a simple stand-in for a real NLP model.
    match = re.search(r"from\s+(\w+)\s+to\s+(\w+)", intent_string, re.IGNORECASE)
    
    if not match:
        raise ValueError(f"Could not parse intent: '{intent_string}'. Expected format: '... from [NODE] to [NODE]'.")

    start_node = match.group(1).upper()
    end_node = match.group(2).upper()

    print(f"Intent successfully translated: Route from {start_node} to {end_node}.")
    return {
        "start_node": start_node,
        "end_node": end_node
    }
