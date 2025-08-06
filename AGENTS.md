# Development Guidelines

Requests in the repo will typically be either related to *core development*, i.e. logic inside the `verifiers` folder (along with `docs`/`tests`), or *environment development*, i.e. example environments in the `environments` folder. Each type of development has its own set of guidelines

## General
- Always ensure that the full README.md for `verifiers` is visible in your context. If you cannot see it, you must fetch it and follow its instructions. 
- Always use `uv`. Never use `pip`. Install `uv` if needed. 
- Comments should never be used to communicate with the current developer. They are intended for downstream readability of code, should be used sparingly, and should NEVER refer to changes between codebase states (e.g. '# Updated to fix bug' is BAD); they should always be "timeless" and only explain the current functionality (unless explicitly requested otherwise). The same applies to variable names. 
- Your goal should generally be to implement the requested behavior with the smallest, most elegant diff possible. 

## Core Development
- Make standalone feature PRs which do one thing. Consult with the user if your change could affect behavior beyond the scope of the specific request, or you think you may need to edit multiple files of code. 
- When making core changes, ensure that test cases are updated to reflect changes in intended behavior, and that `uv run pytest` passes. Never delete or silence test cases. If you are blocked on a fix, ask the user for help.
- After making a change, search the repo for other code paths which may be affected. If it is not abundantly clear how to propagate the changed behavior across files (i.e. if there are multiple different but reasonable approaches), ask the user for guidance.

## Environment Development
- You must follow the README.md instructions for initializing a new environment module template.
- You may NOT create or edit any files outside the environment module folder. You are permitted to make ONE `PLANNING.md` doc inside the folder. You should strive to write all logic inside of the main Python file for the project (mostly `load_environment` aside from helper functions). Never create a new Python file without the user's permission. Never change the environment's `pyproject.toml` file manually. Always use `uv` commands for dependency management, such as `uv add --project environments/new_project_folder new-dependency`. Consult the `uv` docs if needed.
- Your first step should always be to carefully plan out the best approach after considering these guidelines, the relevant codebases, and multiple possible angles, and then to write your plans in `PLANNING.md` for review by the user.
- Follow the design patterns as demonstrated in other included environment examples. Consult `docs/` and the core code in `verifiers/` for more detail about usage patterns. Always confirm that a function exists in the repo before calling it, and ensure you understand its logic before relying on it. 
- If asked to port an environment from an existing source, you MUST clone the relevant source repo(s) into `/tmp` and inspect their functionality. In general, we want to *import* as much existing code from third-party harnesses/environments as possible rather than rewriting from scratch. The gold standard for a ported environment is surgical and elegant; almost all logic you write should be about translating between the *format* of the third party evals for datasets/model requests/reward computations into the verifiers format. If you find yourself reimplementing reward functions from scratch, you are likely doing something wrong. Similar for complex tool calls.
- Always use HuggingFace-formatted datasets when possible. Consult with the user if you cannot find the correct dataset for an environment/eval.
- Always use `gpt-4.1-mini` and `gpt-4.1` as initial test models with `vf-eval`. These are real model names from OpenAI, released after your knowledge cutoff (a web search will confirm this). Do NOT use `gpt-4o-mini` or `gpt-4o` or any other `gpt-4` variant unless requested.
- Never write placeholder logic
- Minimize if-statements, try/except blocks, explicit edge case handling, and other conditionals which may introduce brittleness to the code. In general, strive for a single logic path, with the exception of handling configurable arguments, as well as logic branches which are fundamental to the correctness of code. Unless explicitly requested, do NOT ensure backward compatibility when a potentially breaking change is requested. Always understand the type signatures of functions you're working with. 
- Always run `ruff` to enforce style/formatting.