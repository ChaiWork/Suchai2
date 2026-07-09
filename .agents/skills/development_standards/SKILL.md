---
name: development_standards
description: "Guidelines for coworker relationship, coding standards, documentation, python analysis, and testing."
---

# Development Standards & Guidelines

Follow these standards whenever writing code, testing, or documenting features in this workspace.

## Coworker Relationship
* We are a team. Your success is my success.
* We are informal, but professional.
* We both have valuable, complementary experience.
* It is okay to admit when we don't know something.
* Push back with evidence when appropriate.

## Coding Standards
* Use simple, clean, and maintainable solutions.
* Make the smallest reasonable changes. Ask for permission before rewriting.
* Match the existing code style.
* Stay on task. Create issues for unrelated fixes.
* Do not remove comments unless they are false.
* Use evergreen comments.
* No mock implementations.
* Do not rewrite code to fix a bug without permission.
* Use evergreen naming conventions.

## Documentation
* Store documentation in the documentation directory.
* Use Markdown and create an index named `intro.md` with links.
* Document all commands, sub-commands, and options with examples.

## Analyzing Python Code
* When analyzing Python code, use the `api` module to parse it, UNLESS instructed otherwise.
* Use `api.get_docstring()` to locate a docstring for an item.
* To find type hints, walk the AST using `api.walk_tree()` looking for type parameters with `ast.TypeVar()`, `ast.ParamSpec()`, and `ast.TypeVarTuple()`.

## Getting Help
* Ask for clarification.
* Ask for help when needed.

## Testing
* Tests must cover the implemented functionality.
* Pay attention to logs and test output.
* Test output must be pristine.
* Test for expected errors.
* Practice TDD (Test-Driven Development):
  1. Write a failing test.
  2. Write the minimum code to pass the test.
  3. Refactor.
  4. Repeat.

## Specific Technologies
* Refer to custom documentation at `~/.gemini/docs/python.md` when developing Python.
