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

from agents import Agent, Runner, function_tool
from examples.auto_mode import confirm_with_fallback, input_with_fallback
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent
from agents.stream_events import RawResponsesStreamEvent

# Set your API key here if you don't want to use `export OPENAI_API_KEY`.
os.environ.setdefault("OPENAI_API_KEY", "your key")


TOOL_TEMP = "get_temperature_followup"
TOOL_PRECIP = "get_precipitation_followup"
TOOL_OPTION_MAP = {"1": TOOL_TEMP, "2": TOOL_PRECIP}
TOOL_LABELS = {
    TOOL_TEMP: "1) Temperature follow-up",
    TOOL_PRECIP: "2) Precipitation follow-up",
}


@function_tool(needs_approval=True)
async def get_temperature_followup(city: str, climate_inference: str) -> str:
    """Provide temperature-focused follow-up based on climate inference.

    Args:
        city: The city in question.
        climate_inference: Structured climate inference generated upstream.

    Returns:
        Temperature-oriented follow-up advice.
    """
    return (
        f"Temperature follow-up for {city}: keep light layers for daytime swings. "
        f"Context used: {climate_inference}"
    )


@function_tool(needs_approval=True)
async def get_precipitation_followup(city: str, climate_inference: str) -> str:
    """Provide precipitation-focused follow-up based on climate inference.

    Args:
        city: The city in question.
        climate_inference: Structured climate inference generated upstream.

    Returns:
        Precipitation-oriented follow-up advice.
    """
    return (
        f"Precipitation follow-up for {city}: keep a light rain layer available. "
        f"Context used: {climate_inference}"
    )


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


def parse_climate_inference(payload: str) -> dict:
    """Parse climate inference payload into a dictionary."""
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = {}

    if not isinstance(parsed, dict):
        return {}
    return parsed


def recommended_tool_from_inference(climate_inference: dict) -> str:
    """Map climate inference focus to a follow-up tool."""
    focus = str(climate_inference.get("focus", "")).strip().lower()
    if focus == "precipitation":
        return TOOL_PRECIP
    return TOOL_TEMP


def selected_tool_city(climate_inference: dict) -> str:
    """Pick city from inference payload with fallback."""
    city = str(climate_inference.get("city", "")).strip()
    return city or "Oakland"


async def choose_tool_from_human(
    recommended_tool_name: str,
    proposed_tool_name: str,
    climate_inference_payload: str,
) -> str:
    """Ask human to choose one of two follow-up tools."""
    print("\n" + "-" * 80)
    print("Climate inference received:")
    print(climate_inference_payload)
    print("-" * 80)
    print("Model proposed:", TOOL_LABELS.get(proposed_tool_name, proposed_tool_name))
    print("Choose one follow-up tool:")
    print(f"  {TOOL_LABELS[TOOL_TEMP]}")
    print(f"  {TOOL_LABELS[TOOL_PRECIP]}")

    default_option = "1" if recommended_tool_name == TOOL_TEMP else "2"
    while True:
        option = input_with_fallback(
            "Enter 1 or 2 to choose tool: ",
            fallback=default_option,
        ).strip()
        selected_tool = TOOL_OPTION_MAP.get(option)
        if selected_tool:
            print(f"Selected: {TOOL_LABELS[selected_tool]}")
            return selected_tool
        print("Invalid option. Please enter 1 or 2.")


async def main():
    """Run the human-in-the-loop example."""
    climate_agent = Agent(
        name="Climate Explainer",
        instructions=(
            "You explain climate context and classify follow-up focus. "
            "Return strict JSON only with keys: city, focus, inference, reason. "
            "focus must be either 'temperature' or 'precipitation'."
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

    inference_agent = Agent(
        name="Climate Inference Assistant",
        instructions=(
            "Always call ask_climate_explainer exactly once. "
            "Then return only the JSON result from that tool call."
        ),
        tools=[climate_tool],
    )

    user_question = "can you explain the city's climate pattern in Oakland?"
    result = Runner.run_streamed(inference_agent, user_question)
    await stream_text_deltas(result)

    # Phase 1: climate inference + approval.
    while len(result.interruptions) > 0:
        print("\n" + "=" * 80)
        print("Human-in-the-loop: approval required for climate inference tool call:")
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
        result = Runner.run_streamed(inference_agent, state)
        await stream_text_deltas(result)

    climate_inference_payload = str(result.final_output).strip()
    climate_inference = parse_climate_inference(climate_inference_payload)
    city = selected_tool_city(climate_inference)
    recommended_tool_name = recommended_tool_from_inference(climate_inference)

    followup_agent = Agent(
        name="Climate Follow-up Assistant",
        instructions=(
            "Use climate inference to choose exactly one follow-up tool. "
            f"Choose {TOOL_TEMP} when focus is temperature. "
            f"Choose {TOOL_PRECIP} when focus is precipitation. "
            "Call exactly one tool and return only that tool result."
        ),
        tools=[get_temperature_followup, get_precipitation_followup],
    )

    followup_prompt = (
        f"User question: {user_question}\n"
        f"City: {city}\n"
        f"Climate inference JSON: {climate_inference_payload}\n"
        "Pick one follow-up tool and execute it."
    )
    print("\n" + "=" * 80)
    print("Starting follow-up tool selection phase...")
    print("=" * 80)
    result = Runner.run_streamed(followup_agent, followup_prompt)
    await stream_text_deltas(result)

    # Phase 2: explicit human tool selection from two tools.
    while len(result.interruptions) > 0:
        print("\n" + "=" * 80)
        print("Human-in-the-loop: choose one follow-up tool")
        print("=" * 80)

        state = result.to_state()

        for interruption in result.interruptions:
            print("\nTool call details:")
            print(f"  Agent: {interruption.agent.name}")
            print(f"  Tool: {interruption.name}")
            print(f"  Arguments: {interruption.arguments}")

            if interruption.name in {TOOL_TEMP, TOOL_PRECIP}:
                selected_tool_name = await choose_tool_from_human(
                    recommended_tool_name=recommended_tool_name,
                    proposed_tool_name=interruption.name,
                    climate_inference_payload=climate_inference_payload,
                )

                if selected_tool_name == interruption.name:
                    print(f"✓ Approved selected tool: {interruption.name}")
                    state.approve(interruption)
                else:
                    print(
                        f"↺ Rejected {interruption.name}, forcing alternate tool selection"
                    )
                    state.reject(interruption, always_reject=True)
                continue

            confirmed = await confirm("\nDo you approve this tool call?")
            if confirmed:
                print(f"✓ Approved: {interruption.name}")
                state.approve(interruption)
            else:
                print(f"✗ Rejected: {interruption.name}")
                state.reject(interruption)

        print("\nResuming agent execution...")
        result = Runner.run_streamed(followup_agent, state)
        await stream_text_deltas(result)

    print("\n" + "=" * 80)
    print("Final Output:")
    print("=" * 80)
    print(result.final_output)
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
