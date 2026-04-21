"""
agents/flight_agent.py
----------------------
CrewAI-based Flight Agent with role-based task execution.
Responsible for finding, ranking, and selecting optimal flights.
"""

from __future__ import annotations

import json
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai_tools import tool
from langchain_openai import ChatOpenAI
from loguru import logger

from ..config.prompts import (
    FLIGHT_AGENT_BACKSTORY,
    FLIGHT_AGENT_GOAL,
    FLIGHT_AGENT_ROLE,
)
from ..config.settings import get_settings
from ..schemas.travel_state import FlightOption, TravelPlannerState
from ..tools.flight_tools import get_flight_price_history, search_flights

settings = get_settings()


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.default_llm_model,
        api_key=settings.openai_api_key,
        temperature=0.1,
    )


# ── CrewAI Agent Definition ───────────────────────────────────────────────────

def create_flight_agent() -> Agent:
    """Create the CrewAI Flight Search Agent."""
    return Agent(
        role=FLIGHT_AGENT_ROLE,
        goal=FLIGHT_AGENT_GOAL,
        backstory=FLIGHT_AGENT_BACKSTORY,
        tools=[search_flights, get_flight_price_history],
        llm=_get_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=settings.max_agent_iterations,
    )


def create_flight_analysis_agent() -> Agent:
    """Secondary agent that analyzes and ranks flight options."""
    return Agent(
        role="Flight Value Analyst",
        goal="Analyze flight options and recommend the best value choice based on price, duration, stops, and traveler preferences.",
        backstory="You are an analytical travel optimizer with expertise in evaluating flight options. "
                  "You consider total journey time (including layovers), airline reliability, "
                  "price-to-comfort ratio, and carbon footprint.",
        tools=[],
        llm=_get_llm(),
        verbose=True,
        allow_delegation=False,
    )


# ── Tasks ─────────────────────────────────────────────────────────────────────

def create_flight_search_task(agent: Agent, preferences: dict) -> Task:
    return Task(
        description=f"""
        Search for available flights based on these travel preferences:
        
        - Origin: {preferences['origin']}
        - Destination: {preferences['destination']}
        - Departure Date: {preferences['departure_date']}
        - Return Date: {preferences['return_date']}
        - Travelers: {preferences['travelers']}
        - Cabin Class: {preferences.get('preferred_cabin', 'economy')}
        - Max Stops: {preferences.get('max_stops', 1)}
        - Budget for flights: ${preferences.get('flight_budget', 'flexible')}
        
        Search for both outbound and return flights.
        Also check price history to advise on booking timing.
        Return ALL options found as a JSON list.
        """,
        agent=agent,
        expected_output="A JSON list of flight options with prices, times, airlines, and stops.",
    )


def create_flight_selection_task(agent: Agent, budget_usd: float) -> Task:
    return Task(
        description=f"""
        Review the flight options from the search results and recommend the TOP 3 choices.
        
        Selection criteria (in order of priority):
        1. Total price must be within budget: ${budget_usd:.0f} USD
        2. Minimize total journey time (including layovers)
        3. Prefer fewer stops for comfort
        4. Consider airline reliability scores
        5. Factor in carbon footprint if all else is equal
        
        For your TOP PICK, provide a clear justification.
        Format output as JSON with 'recommendations' array and 'top_pick' object.
        """,
        agent=agent,
        expected_output="JSON with top 3 flight recommendations and a highlighted best choice.",
    )


# ── Main Execution Function ───────────────────────────────────────────────────

async def run_flight_agent(state: TravelPlannerState) -> TravelPlannerState:
    """
    Execute the Flight Agent crew and update the travel planner state.
    Called as a LangGraph node.
    """
    logger.info("🛫 Flight Agent starting...")
    prefs = state["preferences"]
    budget = state["budget"]

    # Allocate ~40% of total budget to flights as initial target
    flight_budget = budget.effective_budget * 0.40

    pref_dict = {
        "origin": prefs.origin,
        "destination": prefs.destination,
        "departure_date": str(prefs.departure_date),
        "return_date": str(prefs.return_date),
        "travelers": prefs.travelers,
        "preferred_cabin": prefs.preferred_cabin,
        "max_stops": prefs.max_stops,
        "flight_budget": flight_budget,
    }

    # Create crew
    search_agent = create_flight_agent()
    analysis_agent = create_flight_analysis_agent()

    search_task = create_flight_search_task(search_agent, pref_dict)
    selection_task = create_flight_selection_task(analysis_agent, flight_budget)
    selection_task.context = [search_task]  # Feed search results to selection

    crew = Crew(
        agents=[search_agent, analysis_agent],
        tasks=[search_task, selection_task],
        process=Process.sequential,
        verbose=True,
    )

    try:
        result = crew.kickoff()
        raw_output = result.raw

        # Parse flight options from crew output
        flight_options = _parse_crew_output(raw_output, prefs)

        # Select the best option (first/top recommendation)
        selected = flight_options[0] if flight_options else None

        # Update budget
        if selected:
            budget.spent_flights = selected.price_usd

        logger.info(f"✅ Flight Agent found {len(flight_options)} options. Selected: {selected and selected.airline}")

        return {
            **state,
            "flight_options": flight_options,
            "selected_flight": selected,
            "budget": budget,
            "current_step": "hotel_search",
            "completed_steps": ["flight_search"],
        }

    except Exception as e:
        logger.error(f"Flight Agent error: {e}")
        # Fallback: generate mock options directly
        from ..tools.flight_tools import _generate_mock_flights
        mock_flights = _generate_mock_flights(
            prefs.origin, prefs.destination,
            str(prefs.departure_date), prefs.travelers,
        )
        flight_options = [FlightOption(**f) for f in mock_flights[:3]]
        selected = flight_options[0] if flight_options else None

        if selected:
            budget.spent_flights = selected.price_usd

        return {
            **state,
            "flight_options": flight_options,
            "selected_flight": selected,
            "budget": budget,
            "current_step": "hotel_search",
            "completed_steps": ["flight_search"],
            "errors": [f"Flight Agent warning: {str(e)} — using fallback data"],
        }


def _parse_crew_output(raw_output: str, prefs: Any) -> list[FlightOption]:
    """Parse CrewAI output string into FlightOption objects."""
    try:
        # Try to extract JSON from the output
        import re
        json_match = re.search(r'\[.*\]', raw_output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return [FlightOption(**item) for item in data[:5]]
    except Exception:
        pass

    # Fallback: generate mock flights
    from ..tools.flight_tools import _generate_mock_flights
    mock = _generate_mock_flights(
        prefs.origin, prefs.destination,
        str(prefs.departure_date), prefs.travelers,
    )
    return [FlightOption(**f) for f in mock[:3]]
