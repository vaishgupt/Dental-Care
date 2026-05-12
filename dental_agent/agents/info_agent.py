from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import ToolNode
import re
from dental_agent.config.settings import GROQ_API_KEY, MODEL_NAME, TEMPERATURE
from dental_agent.models.state import AppointmentState
from dental_agent.tools.csv_reader import (
    get_available_slots,
    get_patient_appointments,
    check_slot_availability,
    list_doctors_by_specialization,
)
from dental_agent.utils import sanitize_messages
from dental_agent.config.settings import VALID_SPECIALIZATIONS

INFO_TOOLS = [
    get_available_slots,
    get_patient_appointments,
    check_slot_availability,
    list_doctors_by_specialization,
]

INFO_SYSTEM = """You are the Information Agent for a dental appointment system.
Your role is to answer queries about doctor availability, schedules, and appointment status.

## Available Tools
- get_available_slots(specialization, doctor_name, date_filter) — find open slots
- get_patient_appointments(patient_id) — look up a patient's bookings
- check_slot_availability(doctor_name, date_slot) — verify a specific slot
- list_doctors_by_specialization(specialization) — list doctors in a specialty

## Guidelines
1. Use tools to fetch real data. Never invent slot times or doctor names.
2. If the user has not provided enough parameters, ask a focused clarifying question.
3. Present results in a clear, friendly, numbered list.
4. Valid specializations: general_dentist, oral_surgeon, orthodontist, 
   cosmetic_dentist, prosthodontist, pediatric_dentist, emergency_dentist.
5. After answering, ask if the user needs anything else.

## Date Format
All dates follow M/D/YYYY H:MM format (e.g., 5/10/2026 9:00).
"""

INFO_PROMPT = ChatPromptTemplate.from_messages([
    ("system", INFO_SYSTEM),
    ("placeholder", "{messages}"),
])

info_tool_node = ToolNode(tools=INFO_TOOLS)


def _extract_specialization(user_text: str) -> str:
    text = user_text.lower()
    for specialization in VALID_SPECIALIZATIONS:
        if specialization in text or specialization.replace("_", " ") in text:
            return specialization
    return ""


def _format_available_slots(slots: list, specialization: str) -> str:
    if not slots:
        if specialization:
            return f"I could not find open slots for `{specialization}` right now. Want me to check another specialty or a specific date?"
        return "I could not find open slots right now. Want me to check a specific specialty or date?"

    lines = ["Here are the available slots:"]
    for i, slot in enumerate(slots, start=1):
        lines.append(
            f"{i}. {slot['date_slot']} - {slot['doctor_name'].title()} ({slot['specialization']})"
        )
    lines.append("Would you like me to help you book one of these?")
    return "\n".join(lines)


def _format_patient_appointments(appointments: list, patient_id: str) -> str:
    if not appointments:
        return f"I could not find any appointments for patient `{patient_id}`."
    lines = [f"Here are appointments for patient `{patient_id}`:"]
    for i, appt in enumerate(appointments, start=1):
        lines.append(
            f"{i}. {appt['date_slot']} - {appt['doctor_name'].title()} ({appt['specialization']})"
        )
    return "\n".join(lines)


def _extract_doctor_name(user_text: str) -> str:
    patterns = [
        r"dr\.?\s+([a-z]+(?:\s+[a-z]+)+?)(?=\s+(?:on|at)\b|$)",
        r"(?:for|with)\s+([a-z]+(?:\s+[a-z]+)+)\s+(?:on|at)\b",
        r"(?:for|with)\s+([a-z]+(?:\s+[a-z]+)+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def _extract_date(user_text: str) -> str:
    match = re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", user_text)
    return match.group(0) if match else ""


def _extract_patient_id(user_text: str) -> str:
    match = re.search(r"patient\s+(\d+)", user_text, re.IGNORECASE)
    return match.group(1) if match else ""


def _fallback_info_response(state: AppointmentState) -> AIMessage | None:
    latest_user = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    if not latest_user or not isinstance(latest_user.content, str):
        return None

    user_text = latest_user.content.lower()

    if "available slot" in user_text or "availability" in user_text:
        specialization = _extract_specialization(user_text)
        doctor_name = _extract_doctor_name(latest_user.content)
        date_filter = _extract_date(latest_user.content)
        slots = get_available_slots.invoke(
            {
                "specialization": specialization,
                "doctor_name": doctor_name,
                "date_filter": date_filter,
            }
        )
        return AIMessage(content=_format_available_slots(slots, specialization))

    if "what appointments" in user_text or "appointments does patient" in user_text:
        patient_id = _extract_patient_id(latest_user.content)
        if not patient_id:
            return AIMessage(content="Please share the patient ID so I can check appointments.")
        appointments = get_patient_appointments.invoke({"patient_id": patient_id})
        return AIMessage(content=_format_patient_appointments(appointments, patient_id))

    return None


def info_agent_node(state: AppointmentState) -> dict:
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
    ).bind_tools(INFO_TOOLS)

    chain = INFO_PROMPT | llm
    try:
        response = chain.invoke({"messages": sanitize_messages(state["messages"])})
    except Exception:
        # Fall back to deterministic handlers when provider/tool-call errors occur.
        fallback = _fallback_info_response(state)
        if fallback is not None:
            response = fallback
        else:
            raise

    return {
        "messages": [response],
        "final_response": response.content if not response.tool_calls else None,
    }