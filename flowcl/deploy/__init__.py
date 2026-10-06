"""Serving flowcl policies to a real robot (Dobot X-Trainer try-out, branch dobot-hw).

``images``, ``wire`` and ``safety`` depend only on numpy, OpenCV and msgpack, so the
robot PC imports them from a plain checkout without torch. ``server`` needs the full
flowcl environment and is never imported from here.
"""
