from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import ToolNode
import re
from dental_agent.config.settings import GROQ_API_KEY, MODEL_NAME, TEMPERATURE
from dental_agent.models.state import AppointmentState
from dental_agent.tools.csv_reader import get_available_slots, check_slot_availability
from dental_agent.tools.csv_writer import book_appointment
from dental_agent.utils import sanitize_messages

BOOKING_TOOLS = [get_available_slots, check_slot_availability, book_appointment]

BOOKING_SYSTEM = """You are the Booking Agent for a dental appointment management system.
Your ONLY job is to book NEW appointments for patients.

## Workflow
1. Collect REQUIRED information (ask if missing):
   - patient_id       : numeric patient ID (e.g., 1000082)
   - specialization   : the type of dentist needed
   - doctor_name      : specific doctor (or help user choose from available)
   - date_slot        : desired date/time in M/D/YYYY H:MM format

2. Call check_slot_availability first to confirm the slot is free.
   - If the slot is taken, call get_available_slots to show alternatives.

3. Once confirmed available, call book_appointment with all parameters.

4. Confirm the booking to the user with all details.

## Rules
- NEVER book without first verifying availability via check_slot_availability.
- If a slot is taken, proactively offer alternatives using get_available_slots.
- Be explicit about what was booked: doctor, date, time, patient ID.
- Ask for ONE missing piece of information at a time.

## Date Format
M/D/YYYY H:MM (e.g., 5/10/2026 9:00)
"""

BOOKING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", BOOKING_SYSTEM),
    ("placeholder", "{messages}"),
])

booking_tool_node = ToolNode(tools=BOOKING_TOOLS)


def _latest_user_text(state: AppointmentState) -> str:
    latest_user = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    if not latest_user or not isinstance(latest_user.content, str):
        return ""
    return latest_user.content.strip()


def _parse_direct_booking(text: str) -> tuple[str, str, str] | None:
    pattern = re.compile(
        r"book\s+patient\s+(\d+)\s+with\s+(.+?)\s+on\s+([0-9]{1,2}/[0-9]{1,2}/[0-9]{4}\s+[0-9]{1,2}:[0-9]{2})",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return None
    patient_id, doctor_name, date_slot = match.groups()
    return patient_id.strip(), doctor_name.strip(), date_slot.strip()


def _direct_booking_response(patient_id: str, doctor_name: str, date_slot: str) -> AIMessage:
    availability = check_slot_availability.invoke(
        {"doctor_name": doctor_name, "date_slot": date_slot}
    )
    if not availability.get("found", False):
        alternatives = get_available_slots.invoke(
            {"specialization": "", "doctor_name": doctor_name, "date_filter": ""}
        )
        if alternatives:
            lines = [
                f"I could not find the slot `{date_slot}` for Dr. {doctor_name.title()}.",
                "Here are available slots for this doctor:",
            ]
            for i, slot in enumerate(alternatives[:10], start=1):
                lines.append(f"{i}. {slot['date_slot']}")
            lines.append("Tell me which one you want, and I can book it.")
            return AIMessage(content="\n".join(lines))
        return AIMessage(
            content=(
                f"I could not find slot `{date_slot}` for Dr. {doctor_name.title()}. "
                "Please share another date/time."
            )
        )

    if not availability.get("is_available", False):
        return AIMessage(
            content=(
                f"That slot is already booked for Dr. {doctor_name.title()} at `{date_slot}`. "
                "Share another date/time and I will check it."
            )
        )

    result = book_appointment.invoke(
        {"patient_id": patient_id, "doctor_name": doctor_name, "date_slot": date_slot}
    )
    if result.get("success"):
        return AIMessage(content=result.get("message", "Appointment booked successfully."))
    return AIMessage(content=result.get("message", "I could not complete that booking."))


def booking_agent_node(state: AppointmentState) -> dict:
    user_text = _latest_user_text(state)
    parsed = _parse_direct_booking(user_text)
    if parsed is not None:
        patient_id, doctor_name, date_slot = parsed
        response = _direct_booking_response(patient_id, doctor_name, date_slot)
        return {
            "messages": [response],
            "final_response": response.content,
        }

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
    ).bind_tools(BOOKING_TOOLS)

    chain = BOOKING_PROMPT | llm
    response = chain.invoke({"messages": sanitize_messages(state["messages"])})

    return {
        "messages": [response],
        "final_response": response.content if not response.tool_calls else None,
    }