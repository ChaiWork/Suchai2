# Project Rules: Pokémon TCG AI Battle Challenge

## Git & File Management
- **Rule**: Never commit or track `.pdf` or `.csv` data files in Git. They exceed file limits and are ignored by `.gitignore`.
- **Rule**: If a commit history is rejected by GitHub due to large files, reset the branch history using `git checkout --orphan` to wipe the bad history from the log.

## Kaggle Environment Constraints
- **Rule**: Always assume files are running under `/kaggle_simulations/agent/` on the remote server.
- **Rule**: Keep imports and file resource paths relative to the root directory, or ensure they fall back to checking `/kaggle_simulations/agent/` dynamically.

## Submission Packaging
- **Rule**: When creating `submission.tar.gz`, do not include PDF files, temporary logs, or large raw datasets. Keep the bundle size strictly below **197.7 MiB**.

## Coworker Relationship
- **Rule**: We are a team. Your success is my success.
- **Rule**: We are informal, but professional.
- **Rule**: We both have valuable, complementary experience.
- **Rule**: It's okay to admit when we don't know something.
- **Rule**: Push back with evidence when appropriate.

## Coding Standards
- **Rule**: Use simple, clean, and maintainable solutions.
- **Rule**: Make the smallest reasonable changes. Ask for permission before rewriting.
- **Rule**: Match the existing code style.
- **Rule**: Stay on task. Create issues for unrelated fixes.
- **Rule**: Don't remove comments unless they are false.
- **Rule**: Use evergreen comments.
- **Rule**: No mock implementations.
- **Rule**: Do not rewrite code to fix a bug without permission.
- **Rule**: Use evergreen naming conventions.

## Documentation
- **Rule**: Store documentation in the documentation directory.
- **Rule**: Use Markdown and create an index named intro.md with links.
- **Rule**: Document all commands, sub-commands, and options with examples.

## Analyzing Python Code
- **Rule**: When analyzing Python code, use the api module to parse it, UNLESS instructed otherwise.
- **Rule**: Use api.get_docstring() to locate a docstring for an item.
- **Rule**: To find type hints, walk the AST using api.walk_tree() looking for type parameters with ast.TypeVar(), ast.ParamSpec(), and ast.TypeVarTuple().

## Getting Help
- **Rule**: Ask for clarification and help when needed.

## Testing
- **Rule**: Tests must cover the implemented functionality.
- **Rule**: Pay attention to logs and test output.
- **Rule**: Test output must be pristine.
- **Rule**: Test for expected errors.
- **Rule**: Practice TDD: write a failing test, write the minimum code to pass, refactor, and repeat.

## Specific Technologies
- **Rule**: Refer to Python documentation at `~/.gemini/docs/python.md` when developing.

