"""
agents/activity_agent.py
------------------------
CrewAI-based Activity Agent with a Researcher + Curator crew.
Finds and schedules day-by-day experiences tailored to user interests.
"""

from __future__ import annotations

import json
from typing import Any

from crewai import Agent, Crew, Process, Task
from langchain_openai import ChatOpenAI
from loguru import logger

from ..config.settings import get_settings
from ..schemas.travel_state import ActivityOption, TravelPlannerState
from ..tools.activity_tools import get_activity_reviews, search_activities

settings = get_settings()


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.default_llm_model,
        api_key=settings.openai_api_key,
        temperature=0.3,
    )


# ── CrewAI Agents ─────────────────────────────────────────────────────────────

def create_activity_researcher() -> Agent:
    return Agent(
        role="Local Experiences Researcher",
        goal="Research and find the best activities, attractions, and experiences "
             "that match the traveler's interests and budget.",
        backstory="You are an obsessive travel researcher who has spent years "
                  "cataloging the best experiences worldwide. You know the difference "
                  "between tourist traps and authentic gems. You always check reviews "
                  "and verify that activities are genuinely worth the price.",
        tools=[search_activities, get_activity_reviews],
        llm=_get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_activity_curator() -> Agent:
    return Agent(
        role="Itinerary Curator & Day Planner",
        goal="Transform a list of activities into a logical, enjoyable day-by-day "
             "schedule that minimizes travel time, respects energy levels, and "
             "creates a natural narrative flow for each day.",
        backstory="You are a master itinerary designer with a gift for storytelling "
                  "through travel. You understand how to pace a trip — mornings for "
                  "energy-heavy sights, afternoons for culture, evenings for experiences. "
                  "You always cluster activities by geography and create seamless transitions.",
        tools=[],
        llm=_get_llm(),
        verbose=True,
        allow_delegation=False,
    )


# ── Tasks ─────────────────────────────────────────────────────────────────────

def create_activity_research_task(
    agent: Agent,
    destination: str,
    trip_days: int,
    interests: list[str],
    daily_budget: float,
    pace: str,
) -> Task:
    return Task(
        description=f"""
        Research activities and experiences for this trip:
        
        - Destination: {destination}
        - Trip Duration: {trip_days} days
        - Interests: {', '.join(interests) if interests else 'general sightseeing'}
        - Daily Activity Budget: ${daily_budget:.0f} USD
        - Pace: {pace}
        
        Use the search_activities tool to find options.
        For top recommendations, check reviews with get_activity_reviews.
        
        Find at least {trip_days * 2} diverse activities covering different times of day.
        Prioritize: authenticity, value for money, match to stated interests.
        Return results as a JSON array.
        """,
        agent=agent,
        expected_output="JSON array of activity options with day assignments, times, prices, and descriptions.",
    )


def create_schedule_task(
    agent: Agent,
    trip_days: int,
    destination: str,
    pace: str,
) -> Task:
    return Task(
        description=f"""
        Create an optimized day-by-day activity schedule for {trip_days} days in {destination}.
        
        Using the research results, build a schedule that:
        1. Groups activities by geographic proximity (minimize travel time)
        2. Schedules high-energy activities in the morning
        3. Places cultural/relaxing activities in afternoons
        4. Reserves evenings for dining experiences and entertainment
        5. Doesn't overload any single day (respect the '{pace}' pace)
        6. Includes realistic travel time between venues
        7. Highlights 1-2 "must-do" anchors per day
        
        Format each day as:
        Day N: [Theme/Title]
        - Morning (9:00-12:00): [Activity]
        - Afternoon (13:00-17:00): [Activity]  
        - Evening (18:00-22:00): [Activity]
        
        Also output a JSON summary with 'scheduled_activities' array.
        """,
        agent=agent,
        expected_output="Formatted day-by-day schedule with JSON summary of all scheduled activities.",
    )


# ── Main Execution Function ───────────────────────────────────────────────────

async def run_activity_agent(state: TravelPlannerState) -> TravelPlannerState:
    """
    Execute the Activity Agent crew and update state.
    Called as a LangGraph node.
    """
    logger.info("🎯 Activity Agent starting...")
    prefs = state["preferences"]
    budget = state["budget"]

    trip_days = prefs.trip_days
    remaining = budget.remaining
    # Activities ~20-25% of total budget
    activity_budget = min(remaining * 0.50, budget.effective_budget * 0.25)
    daily_activity_budget = activity_budget / trip_days if trip_days > 0 else activity_budget

    researcher = create_activity_researcher()
    curator = create_activity_curator()

    research_task = create_activity_research_task(
        researcher,
        destination=prefs.destination,
        trip_days=trip_days,
        interests=prefs.interests,
        daily_budget=daily_activity_budget,
        pace=prefs.activity_pace,
    )

    schedule_task = create_schedule_task(
        curator,
        trip_days=trip_days,
        destination=prefs.destination,
        pace=prefs.activity_pace,
    )
    schedule_task.context = [research_task]

    crew = Crew(
        agents=[researcher, curator],
        tasks=[research_task, schedule_task],
        process=Process.sequential,
        verbose=True,
    )

    activity_options = []
    selected_activities = []

    try:
        result = crew.kickoff()
        raw_output = result.raw

        activity_options, selected_activities = _parse_activity_output(
            raw_output, prefs, daily_activity_budget
        )

    except Exception as e:
        logger.error(f"Activity Agent error: {e}")
        activity_options, selected_activities = _fallback_activities(
            prefs, daily_activity_budget
        )
        state["errors"] = state.get("errors", []) + [f"Activity Agent warning: {str(e)}"]

    # Update budget
    total_activity_cost = sum(a.price_usd for a in selected_activities)
    budget.spent_activities = total_activity_cost

    logger.info(
        f"✅ Activity Agent complete. {len(selected_activities)} activities scheduled. "
        f"Total cost: ${total_activity_cost:.2f}"
    )

    return {
        **state,
        "activity_options": activity_options,
        "selected_activities": selected_activities,
        "budget": budget,
        "current_step": "budget_review",
        "completed_steps": ["activity_search"],
        "requires_human_approval": settings.enable_human_in_loop,
    }


def _parse_activity_output(
    raw_output: str,
    prefs: Any,
    daily_budget: float,
) -> tuple[list[ActivityOption], list[ActivityOption]]:
    """Parse CrewAI activity output."""
    try:
        import re
        json_match = re.search(r'"scheduled_activities"\s*:\s*(\[.*?\])', raw_output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            activities = [ActivityOption(**item) for item in data]
            return activities, activities
    except Exception:
        pass

    return _fallback_activities(prefs, daily_budget)


def _fallback_activities(prefs: Any, daily_budget: float) -> tuple[list[ActivityOption], list[ActivityOption]]:
    """Generate activities directly without crew output parsing."""
    from ..tools.activity_tools import _generate_activities

    raw_activities = _generate_activities(
        destination=prefs.destination,
        trip_days=prefs.trip_days,
        interests=prefs.interests or ["food", "culture", "history"],
        budget_per_day=daily_budget,
        pace=prefs.activity_pace,
    )
    activities = [ActivityOption(**a) for a in raw_activities]
    return activities, activities
