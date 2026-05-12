# Dental Appointment System

A conversational dental appointment management assistant built as a data-driven prototype using LangGraph, Groq, LangChain, and CSV-backed tooling.

## Overview

This project demonstrates a production-inspired assistant that:
- queries and filters appointment availability using a structured dataset
- books new appointments only after validating time-slot availability
- cancels appointments with confirmation and state updates
- reschedules appointments while preserving existing patient and doctor constraints
- answers natural language questions about patient schedules and doctor availability

The system integrates a conversational LLM workflow with a CSV-based dataset, making it easy to inspect, extend, and migrate to a relational database or API-backed service.

## Data Model and CSV Schema

The application uses `doctor_availability.csv` as the single source of truth for appointment state. Each row represents an appointment slot and contains:

- `date_slot` – timestamp for the appointment slot in `M/D/YYYY H:MM` format
- `specialization` – dental specialty such as `general_dentist`, `orthodontist`, or `pediatric_dentist`
- `doctor_name` – provider name tied to the slot
- `is_available` – boolean-like flag tracking whether the slot is free
- `patient_to_attend` – patient identifier for booked appointments

This dataset structure enables:
- schedule analytics and availability lookups
- doctor-specific filtering and specialization-based search
- appointment lifecycle management through read/write tool abstractions

## Key Technical Highlights

- `langchain-core` and `langchain-groq` for LLM integration
- `langgraph` react-agent workflow for stepwise reasoning and tool orchestration
- `pandas`-based CSV reader/writer utilities for production-style data handling
- environment-driven configuration using `python-dotenv`
- modular tool set with clear separation between I/O, business logic, and conversational flow

## Project Structure

- `main.py` – application entry point and interactive CLI
- `doctor_availability.csv` – appointment availability dataset
- `dental_agent/agent.py` – graph agent definition, prompt template, and tool registration
- `dental_agent/config/settings.py` – environment variables and application constants
- `dental_agent/tools/` – CSV reader and writer utility implementations
- `dental_agent/workflows/graph.py` – reusable LangGraph workflow definition

## Requirements

- Python 3.11+ recommended
- `venv` or a virtual environment for isolation

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root with the following values:

```env
GROQ_API_KEY=your_groq_api_key
MODEL_NAME=llama-3.3-70b-versatile
TEMPERATURE=0
```

- `GROQ_API_KEY` is required for the Groq chat model.
- `MODEL_NAME` can be adjusted if you have access to another supported model.
- `TEMPERATURE` controls response randomness.

## Running the App

From the project root:

```bash
python main.py
```

Example prompts:
- `Show available slots for an orthodontist`
- `Book patient 1000082 with Emily Johnson on 5/10/2026 9:00`
- `Cancel appointment for patient 1000082 at 5/10/2026 9:00`
- `Reschedule patient 1000082 from 5/10/2026 9:00 to 5/12/2026 10:00`
- `What appointments does patient 1000048 have?`

Type `quit`, `exit`, or `bye` to end the session.

## Recruiter-Friendly Calling Card

This repository highlights:
- end-to-end system design combining AI, tooling, and data integration
- clean separation of conversational logic from state management
- a realistic CSV-backed workflow that can be extended to databases, APIs, or analytics dashboards
- thoughtful prompt engineering, slot validation, and retry-safe tool execution

## Notes

- The app uses `doctor_availability.csv` for all appointment state.
- Date/time input should follow `M/D/YYYY H:MM` format.
- The system validates slot availability before booking.
- `doctor_availability.csv` is intentionally easy to inspect and extend for demos, testing, or migration.


