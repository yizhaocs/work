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

### Inference-based HITL routing

The script now performs a routing inference step before the main streamed run.
The model returns:

- `tool`: one of `get_temperature`, `get_weather`, `ask_climate_explainer`
- `score`: confidence score (`0.0` to `1.0`) that climate explanation is needed

When `score >= INFERENCE_APPROVAL_THRESHOLD`, the program enters HITL and asks
whether you want to manually choose the tool.

Useful environment variables:

```
# User question (default is an Oakland climate question)
export EXAMPLE_USER_QUERY="what is the weather in Paris today?"

# Threshold for triggering manual tool selection (default: 0.7)
export INFERENCE_APPROVAL_THRESHOLD=0.6

# Optional auto mode for CI/demo
export EXAMPLES_INTERACTIVE_MODE=auto
python3 human_in_the_loop_stream.py
```
