"""Quarantine for Covasim v3 modules during the v4.0 Starsim port.

During the port, v3 modules that the current milestone has not yet replaced are
moved here as a porting reference. Active code in ``covasim/`` NEVER imports from
this package -- it exists purely so the v3 implementation stays readable alongside
the v4 reimplementation. This package is deleted wholesale at M10.

Empty in M0: no migration code has landed yet, so nothing is quarantined.
"""
