"""Human-in-the-loop example with streaming.

This example demonstrates the human-in-the-loop (HITL) pattern with streaming.
The agent will pause execution when a tool requiring approval is called,
allowing you to approve or reject the tool call before continuing.

The streaming version provides real-time feedback as the agent processes
the request, then pauses for approval when needed.
"""

import asyncio
import json
import os
import re
from typing import Any

from agents import Agent, Runner, function_tool
from examples.auto_mode import confirm_with_fallback, input_with_fallback
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent
from agents.stream_events import RawResponsesStreamEvent

# Set your API key here if you don't want to use `export OPENAI_API_KEY`.
os.environ.setdefault("OPENAI_API_KEY", "your key")

DEFAULT_USER_QUERY = os.environ.get(
    "EXAMPLE_USER_QUERY", "can you explain the city's climate pattern in Oakland?"
)
DEFAULT_APPROVAL_THRESHOLD = 0.7
AVAILABLE_TOOL_NAMES = ("get_temperature", "get_weather", "ask_climate_explainer")


def _read_approval_threshold() -> float:
    """Read and clamp the inference threshold used to trigger HITL."""
    raw = os.environ.get(
        "INFERENCE_APPROVAL_THRESHOLD", str(DEFAULT_APPROVAL_THRESHOLD)
    )
    try:
        parsed = float(raw)
    except ValueError:
        parsed = DEFAULT_APPROVAL_THRESHOLD
    return max(0.0, min(1.0, parsed))


def _parse_routing_result(raw_output: str) -> dict[str, Any]:
    """Parse route decision from model output with safe fallbacks."""
    fallback = {
        "tool": "get_weather",
        "score": 0.0,
        "reason": "Fallback route because structured parsing failed.",
    }

    json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not json_match:
        return fallback

    try:
        data = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return fallback

    tool = str(data.get("tool", fallback["tool"])).strip()
    if tool not in AVAILABLE_TOOL_NAMES:
        tool = fallback["tool"]

    try:
        score = float(data.get("score", fallback["score"]))
    except (TypeError, ValueError):
        score = float(fallback["score"])
    score = max(0.0, min(1.0, score))

    reason = str(data.get("reason", fallback["reason"])).strip() or fallback["reason"]
    return {"tool": tool, "score": score, "reason": reason}


async def infer_tool_route(user_query: str) -> dict[str, Any]:
    """Infer preferred tool and confidence score for the user query."""
    router_agent = Agent(
        name="Tool Router",
        instructions=(
            "You decide which tool is best for a weather-related request. "
            "Return ONLY JSON with keys: tool, score, reason. "
            "tool must be one of: get_temperature, get_weather, ask_climate_explainer. "
            "score must be a float between 0 and 1 representing confidence "
            "that ask_climate_explainer is needed."
        ),
    )
    routing_result = await Runner.run(
        router_agent,
        (
            "Decide route for this user request:\n"
            f"{user_query}\n"
            "Remember: output strict JSON only."
        ),
    )
    return _parse_routing_result(str(routing_result.final_output))


def choose_tool(default_tool: str) -> str:
    """Ask user to choose one tool, with auto-mode fallback."""
    options = {
        "1": "get_temperature",
        "2": "get_weather",
        "3": "ask_climate_explainer",
    }
    reverse_options = {name: key for key, name in options.items()}
    default_choice = reverse_options.get(default_tool, "2")

    print("\nChoose tool:")
    print("  1) get_temperature")
    print("  2) get_weather")
    print("  3) ask_climate_explainer")
    choice = input_with_fallback(
        f"Select tool [1-3] (default={default_choice}): ", fallback=default_choice
    ).strip()

    if not choice:
        choice = default_choice
    return options.get(choice, options[default_choice])


async def _needs_temperature_approval(_ctx, params, _call_id) -> bool:
    """Check if temperature tool needs approval."""
    return "Oakland" in params.get("city", "")


@function_tool(
    # Dynamic approval: only require approval for Oakland
    needs_approval=_needs_temperature_approval
)
async def get_temperature(city: str) -> str:
    """Get the temperature for a given city.

    Args:
        city: The city to get temperature for.

    Returns:
        Temperature information for the city.
    """
    return f"The temperature in {city} is 20° Celsius"


@function_tool
async def get_weather(city: str) -> str:
    """Get the weather for a given city.

    Args:
        city: The city to get weather for.

    Returns:
        Weather information for the city.
    """
    return f"The weather in {city} is sunny."


async def confirm(question: str) -> bool:
    """Prompt user for yes/no confirmation.

    Args:
        question: The question to ask.

    Returns:
        True if user confirms, False otherwise.
    """
    return confirm_with_fallback(f"{question} (y/n): ", default=True)


async def stream_text_deltas(result) -> None:
    """Print streamed response text deltas as they arrive."""
    printed_any_token = False
    async for event in result.stream_events():
        if isinstance(event, RawResponsesStreamEvent) and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            print(event.data.delta, end="", flush=True)
            printed_any_token = True

    if printed_any_token:
        print()


async def main():
    """Run the human-in-the-loop example."""
    threshold = _read_approval_threshold()

    climate_agent = Agent(
        name="Climate Explainer",
        instructions=(
            "You explain climate context and general weather patterns in plain language. "
            "Do not call tools; provide concise educational explanations."
        ),
    )

    climate_tool = climate_agent.as_tool(
        tool_name="ask_climate_explainer",
        tool_description=(
            "Use this tool when the user asks for climate context, background, "
            "or explanation beyond direct weather/temperature lookup."
        ),
        needs_approval=True,
    )

    user_query = DEFAULT_USER_QUERY
    route = await infer_tool_route(user_query)
    selected_tool_name = route["tool"]

    print("=" * 80)
    print("Inference route decision")
    print("=" * 80)
    print(f"User query: {user_query}")
    print(f"Suggested tool: {route['tool']}")
    print(f"Score: {route['score']:.2f} (threshold={threshold:.2f})")
    print(f"Reason: {route['reason']}")

    if route["score"] >= threshold:
        print("\nHITL triggered by inference score threshold.")
        should_choose = await confirm(
            "Do you want to manually choose which tool to use?"
        )
        if should_choose:
            selected_tool_name = choose_tool(default_tool=route["tool"])
            print(f"Selected by user: {selected_tool_name}")
        else:
            print("Using inferred tool without manual override.")
    else:
        print("\nThreshold not reached, no manual tool selection required.")

    tool_registry = {
        "get_temperature": get_temperature,
        "get_weather": get_weather,
        "ask_climate_explainer": climate_tool,
    }

    main_agent = Agent(
        name="Weather Assistant",
        instructions=(
            "You are a helpful weather assistant. "
            "Use the provided tool to answer the request. "
            "If the tool result is not enough, summarize limitations briefly."
        ),
        tools=[tool_registry[selected_tool_name]],
    )

    # Run the agent with streaming
    result = Runner.run_streamed(
        main_agent,
        user_query,
    )
    await stream_text_deltas(result)

    # Handle interruptions
    while len(result.interruptions) > 0:
        print("\n" + "=" * 80)
        print("Human-in-the-loop: approval required for the following tool calls:")
        print("=" * 80)

        state = result.to_state()

        for interruption in result.interruptions:
            print("\nTool call details:")
            print(f"  Agent: {interruption.agent.name}")
            print(f"  Tool: {interruption.name}")
            print(f"  Arguments: {interruption.arguments}")

            confirmed = await confirm("\nDo you approve this tool call?")

            if confirmed:
                print(f"✓ Approved: {interruption.name}")
                state.approve(interruption)
            else:
                print(f"✗ Rejected: {interruption.name}")
                state.reject(interruption)

        # Resume execution with streaming
        print("\nResuming agent execution...")
        result = Runner.run_streamed(main_agent, state)
        await stream_text_deltas(result)

    print("\n" + "=" * 80)
    print("Final Output:")
    print("=" * 80)
    print(result.final_output)
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
