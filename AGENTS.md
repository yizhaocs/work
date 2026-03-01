## Cursor Cloud specific instructions

This is a small Python CLI project using the OpenAI Agents SDK (`openai-agents`). See `README.md` for setup and run instructions.

### Running the application

- Requires `OPENAI_API_KEY` environment variable. Without it the app will fail at agent initialization.
- Use `EXAMPLES_INTERACTIVE_MODE=auto` to bypass interactive prompts (auto-approve tool calls).
- Entry point: `python3 human_in_the_loop_stream.py`
- The virtual environment is at `.venv/`. Activate with `source .venv/bin/activate`.

### Linting

- `ruff check .` — linting
- `ruff format --check .` — format check

### Gotchas

- The `python3.12-venv` system package must be installed for `python3 -m venv` to work (already handled by the VM snapshot).
- There are no automated tests in this repo; validation is done by running the script end-to-end with auto mode.
