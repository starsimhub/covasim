"""Quarantine for Covasim v3 tests that exercise removed/replaced v4 APIs.

When a milestone replaces a v3 subsystem, the v3 tests that exercise its removed
API move here so they neither run nor block CI, while staying available as a
reference for what behavior the v4 port must preserve. Deleted wholesale at M10.

Empty in M0: nothing has been removed yet.
"""
