"""
Dental Appointment System — powered by LangGraph + Groq
"""
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessageChunk
from dental_agent.workflows.graph import dental_graph

BANNER = """
╔══════════════════════════════════════════════════════════╗
║         Dental Appointment Management System             ║
║         Powered by LangGraph + Groq (llama-3.3-70b)     ║
╚══════════════════════════════════════════════════════════╝
Examples:
  • Show available slots for an orthodontist
  • Book patient 1000082 with Emily Johnson on 5/10/2026 9:00
  • Cancel appointment for patient 1000082 at 5/10/2026 9:00
  • Reschedule patient 1000082 from 5/10/2026 9:00 to 5/12/2026 10:00
  • What appointments does patient 1000048 have?

Type 'quit' to exit.
"""


def run():
    print(BANNER)
    history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "bye"}:
            print("Goodbye!")
            break

        history.append(HumanMessage(content=user_input))
        print("\nAgent: ", end="", flush=True)

        final_messages = None
        streamed_any_token = False

        try:
            for event_type, data in dental_graph.stream(
                {"messages": history},
                stream_mode=["messages", "values"],
                config={"recursion_limit": 20},
            ):
                if event_type == "messages":
                    chunk, _meta = data
                    # Stream tokens only from the agent (not tool results)
                    if (
                        isinstance(chunk, AIMessageChunk)
                        and chunk.content
                        and not getattr(chunk, "tool_calls", None)
                    ):
                        streamed_any_token = True
                        print(chunk.content, end="", flush=True)

                elif event_type == "values":
                    final_messages = data.get("messages", [])

        except Exception as exc:
            print(f"\nError: {exc}")
            history.pop()  # Remove HumanMessage to avoid consecutive user messages
            continue

        if (
            not streamed_any_token
            and final_messages
            and getattr(final_messages[-1], "content", "")
        ):
            print(final_messages[-1].content, end="", flush=True)

        print()

        if final_messages:
            history = final_messages


if __name__ == "__main__":
    run()