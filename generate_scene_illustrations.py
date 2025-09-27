#!/usr/bin/env python3
"""Compatibility shim for the packaged illustration generator module."""

from illustrator.generate_scene_illustrations import *  # noqa: F401,F403

if __name__ == "__main__":
    import asyncio
    from illustrator.generate_scene_illustrations import main

    asyncio.run(main())
