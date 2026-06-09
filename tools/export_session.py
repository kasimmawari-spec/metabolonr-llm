import json
import os
from datetime import datetime

def export_session(session_log: list, output_dir: str = "logs") -> dict:
    """
    Saves the full session log to a JSON file.
    Each entry in session_log is a dict describing a tool call:
    {"tool": "qc_filter", "params": {...}, "result_summary": {...}}
    This allows the session to be replayed without the LLM.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    session_data = {
        "timestamp": timestamp,
        "n_steps": len(session_log),
        "steps": session_log
    }

    with open(filepath, "w") as f:
        json.dump(session_data, f, indent=2)

    print(f"Session exported to {filepath} ({len(session_log)} steps logged).")
    return {"filepath": filepath, "n_steps": len(session_log)}
    