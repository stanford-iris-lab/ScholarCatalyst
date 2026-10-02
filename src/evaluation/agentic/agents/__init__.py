"""Agents. Each module exposes run(query, view_dir, traj, tools) -> answer text.

view_dir is the per query corpus view or None; traj is the Trajectory to log every call on; tools holds
"retrieve", "corpus" and "model". The runner parses corpus ids out of the answer, so any answer format works.
"""
