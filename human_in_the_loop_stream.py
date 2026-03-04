"""Human-in-the-loop example with streaming.

This example demonstrates the human-in-the-loop (HITL) pattern with streaming.
The agent will pause execution when a tool requiring approval is called,
allowing you to approve or reject the tool call before continuing.

The streaming version provides real-time feedback as the agent processes
the request, then pauses for approval when needed.
"""

from agents.stream_events import RawResponsesStreamEvent
import asyncio
import os

from agents import Agent, Runner, function_tool, handoff
from examples.auto_mode import confirm_with_fallback
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

# Set your API key here if you don't want to use `export OPENAI_API_KEY`.
os.environ.setdefault("OPENAI_API_KEY", "your key")


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

    main_agent = Agent(
        name="Weather Assistant",
        instructions=(
            "You are a helpful weather assistant. "
            "Answer questions about weather and temperature using the available tools. "
            "If the user asks for broader climate explanation, call ask_climate_explainer."
        ),
        tools=[climate_tool],
        # tools=[get_temperature, get_weather, climate_tool],
    )

    handoff_router = Agent(
        name="Weather Router",
        instructions=(
            "You route user weather and climate questions to the right specialist. "
            "Always transfer weather and climate requests to the Weather Assistant."
        ),
        handoffs=[
            handoff(
                main_agent,
                tool_name_override="transfer_to_weather_assistant",
                tool_description_override=(
                    "Transfer weather and climate questions to Weather Assistant."
                ),
            )
        ],
    )

    # Run the agent with streaming
    result = Runner.run_streamed(
        handoff_router,
        "can you explain the city's climate pattern in Oakland?",
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
        result = Runner.run_streamed(handoff_router, state)
        await stream_text_deltas(result)

    print("\n" + "=" * 80)
    print("Final Output:")
    print("=" * 80)
    print(result.final_output)
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
