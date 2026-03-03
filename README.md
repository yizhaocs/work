## Human-in-the-Loop Streaming Example

This project mirrors the `human_in_the_loop_stream.py` example from
the OpenAI Agents Python SDK, and also includes an `Agent.as_tool` HITL example.

### Setup

1. Create and activate a virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Set your API key in code:

```
# Edit `human_in_the_loop_stream.py` and replace the placeholder:
os.environ.setdefault("OPENAI_API_KEY", "your_api_key")
```

### Run

```
python3 human_in_the_loop_stream.py
```

### Auto Mode (optional)

To auto-approve prompts in the example:

```
export EXAMPLES_INTERACTIVE_MODE=auto
python3 human_in_the_loop_stream.py
```
