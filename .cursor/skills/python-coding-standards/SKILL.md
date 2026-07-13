---
name: python-coding-standards
description: Enforces clean, modular Python with type hints, docstrings, pathlib, and separated data/preprocessing/training/evaluation/visualization/configuration modules. Use when writing or refactoring Python in this repository, creating new modules, editing notebooks toward production code, or reviewing implementation quality.
---

# Python Coding Standards

Always write clean, modular Python.

Prefer

- classes when appropriate
- reusable functions
- descriptive names
- type hints
- docstrings

Avoid

- duplicated code
- large notebook-style scripts
- magic numbers
- deeply nested functions

Separate

data

preprocessing

training

evaluation

visualization

configuration

into different modules.

Use pathlib instead of string paths whenever practical.

Avoid hardcoded file locations.

Always handle errors gracefully.

Comment only when it improves understanding.

Write maintainable code rather than clever code.

When modifying existing code, preserve backward compatibility whenever possible.
