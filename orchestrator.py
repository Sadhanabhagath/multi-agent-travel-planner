"""
agents/orchestrator.py
----------------------
LangGraph-based master orchestrator.
Defines the stateful workflow graph connecting all agents,
human-in-the-loop checkpoints, and itinerary synthesis.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from ..config.prompts import ITINERARY_SYNTHESIS_PROMPT, ORCHESTRATOR_SYSTEM_PROMPT
from ..config.settings import get_settings
from ..schemas.travel_state import BudgetBreakdown, TravelPlannerState
from .flight_agent import run_flight_agent
from .hotel_agent import run_hotel_agent
from .activity_agent import run_activity_agent
from .budget_agent import run_budget_agent

settings = get_settings()
console = Console()


# ── Node Functions ────────────────────────────────────────────────────────────

async def initialize_state(state: TravelPlannerState) -> TravelPlannerState:
    """
    NODE: Initialize the workflow state.
    Validates preferences and sets up budget tracking.
    """
    logger.info("🚀 Initializing travel planner state...")
    prefs = state["preferences"]

    budget = BudgetBreakdown(
        total_budget_usd=prefs.budget_usd,
        buffer_percent=settings.budget_buffer_percent,
    )

    console.print(Panel(
        f"[bold cyan]✈️  Planning your trip to {prefs.destination}[/bold cyan]\n"
        f"[dim]{prefs.departure_date} → {prefs.return_date} | "
        f"{prefs.travelers} traveler(s) | Budget: ${prefs.budget_usd:,.0f} USD[/dim]",
        title="Multi-Agent Travel Planner",
        border_style="blue",
    ))

    return {
        **state,
        "budget": budget,
        "flight_options": [],
        "hotel_options": [],
        "activity_options": [],
        "selected_activities": [],
        "selected_flight": None,
        "selected_hotel": None,
        "errors": [],
        "completed_steps": [],
        "current_step": "flight_search",
        "human_feedback": None,
        "requires_human_approval": False,
        "final_itinerary": None,
        "itinerary_markdown": None,
    }


async def human_approval_node(state: TravelPlannerState) -> TravelPlannerState:
    """
    NODE: Human-in-the-loop checkpoint.
    In CLI mode: prompts user. In API mode: returns state for external approval.
    """
    logger.info("⏸️  Waiting for human approval...")
    budget = state["budget"]
    flight = state.get("selected_flight")
    hotel = state.get("selected_hotel")
    activities = state.get("selected_activities", [])

    summary = _build_approval_summary(flight, hotel, activities, budget)
    console.print(summary)

    if settings.app_env == "development":
        # Interactive CLI approval
        try:
            user_input = input("\n✅ Approve this plan? (yes/no/modify): ").strip().lower()
            feedback = user_input if user_input in ["no", "modify"] else "approved"
        except (EOFError, KeyboardInterrupt):
            feedback = "approved"
    else:
        # In production/API mode, approval comes via state injection
        feedback = state.get("human_feedback", "approved")

    return {
        **state,
        "human_feedback": feedback,
        "requires_human_approval": False,
        "current_step": "synthesis" if feedback == "approved" else "re_plan",
        "completed_steps": ["human_approval"],
    }


async def synthesize_itinerary(state: TravelPlannerState) -> TravelPlannerState:
    """
    NODE: Generate the final polished itinerary using LLM synthesis.
    """
    logger.info("📝 Synthesizing final itinerary...")

    prefs = state["preferences"]
    budget = state["budget"]
    flight = state.get("selected_flight")
    hotel = state.get("selected_hotel")
    activities = state.get("selected_activities", [])

    # Build context for LLM synthesis
    context = _build_synthesis_context(prefs, flight, hotel, activities, budget)

    try:
        llm = ChatOpenAI(
            model=settings.orchestrator_llm_model,
            api_key=settings.openai_api_key,
            temperature=0.4,
        )

        messages = [
            {"role": "system", "content": ITINERARY_SYNTHESIS_PROMPT},
            {"role": "user", "content": context},
        ]

        response = await llm.ainvoke(messages)
        itinerary_md = response.content

    except Exception as e:
        logger.warning(f"LLM synthesis error: {e}. Using template fallback.")
        itinerary_md = _generate_template_itinerary(prefs, flight, hotel, activities, budget)

    # Build structured itinerary dict
    final_itinerary = {
        "destination": prefs.destination,
        "origin": prefs.origin,
        "dates": {
            "departure": str(prefs.departure_date),
            "return": str(prefs.return_date),
            "duration_days": prefs.trip_days,
        },
        "travelers": prefs.travelers,
        "flight": flight.model_dump() if flight else None,
        "hotel": hotel.model_dump() if hotel else None,
        "activities": [a.model_dump() for a in activities],
        "budget": {
            "total_usd": budget.total_budget_usd,
            "spent_usd": budget.total_spent,
            "remaining_usd": budget.remaining,
            "breakdown": {
                "flights": budget.spent_flights,
                "hotels": budget.spent_hotels,
                "activities": budget.spent_activities,
            },
            "utilization_percent": budget.utilization_percent,
        },
        "session_id": state.get("session_id", ""),
    }

    console.print("\n")
    console.print(Panel(Markdown(itinerary_md[:2000] + "..." if len(itinerary_md) > 2000 else itinerary_md),
                        title="🌍 Your Travel Itinerary", border_style="green"))

    return {
        **state,
        "final_itinerary": final_itinerary,
        "itinerary_markdown": itinerary_md,
        "current_step": "complete",
        "completed_steps": ["synthesis"],
    }


async def handle_errors(state: TravelPlannerState) -> TravelPlannerState:
    """NODE: Error handling and graceful degradation."""
    errors = state.get("errors", [])
    logger.warning(f"⚠️  Handling {len(errors)} error(s): {errors}")

    console.print(Panel(
        f"[yellow]⚠️  Some data used mock/fallback values:[/yellow]\n" +
        "\n".join(f"  • {e}" for e in errors),
        title="Notice",
        border_style="yellow",
    ))

    return {**state, "current_step": "synthesis"}


# ── Routing Functions ─────────────────────────────────────────────────────────

def route_after_init(state: TravelPlannerState) -> Literal["flight_search"]:
    return "flight_search"


def route_after_budget(state: TravelPlannerState) -> Literal["human_approval", "synthesis", "error_handler"]:
    errors = state.get("errors", [])
    if errors and len(errors) > 2:
        return "error_handler"
    if state.get("requires_human_approval"):
        return "human_approval"
    return "synthesis"


def route_after_approval(state: TravelPlannerState) -> Literal["synthesis", "flight_search"]:
    feedback = state.get("human_feedback", "approved")
    if feedback in ["no", "modify"]:
        return "flight_search"  # Re-plan from scratch
    return "synthesis"


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_travel_planner_graph() -> StateGraph:
    """
    Construct the LangGraph workflow.
    
    Flow:
    START → initialize → flights → hotels → activities → budget → 
    [human_approval?] → synthesis → END
    """
    graph = StateGraph(TravelPlannerState)

    # Add nodes
    graph.add_node("initialize", initialize_state)
    graph.add_node("flight_search", run_flight_agent)
    graph.add_node("hotel_search", run_hotel_agent)
    graph.add_node("activity_search", run_activity_agent)
    graph.add_node("budget_review", run_budget_agent)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("synthesis", synthesize_itinerary)
    graph.add_node("error_handler", handle_errors)

    # Add edges
    graph.add_edge(START, "initialize")
    graph.add_conditional_edges("initialize", route_after_init)
    graph.add_edge("flight_search", "hotel_search")
    graph.add_edge("hotel_search", "activity_search")
    graph.add_edge("activity_search", "budget_review")
    graph.add_conditional_edges("budget_review", route_after_budget)
    graph.add_conditional_edges("human_approval", route_after_approval)
    graph.add_edge("error_handler", "synthesis")
    graph.add_edge("synthesis", END)

    return graph


def create_compiled_graph():
    """Compile the graph with memory checkpointing for human-in-the-loop."""
    graph = build_travel_planner_graph()
    memory = MemorySaver()
    return graph.compile(
        checkpointer=memory,
        interrupt_before=["human_approval"] if settings.enable_human_in_loop else [],
    )


# ── Main Entry Point ──────────────────────────────────────────────────────────

class TravelPlannerOrchestrator:
    """High-level interface to the travel planner system."""

    def __init__(self):
        self.graph = create_compiled_graph()
        self.sessions: dict[str, Any] = {}

    async def plan_trip(self, preferences: Any) -> dict[str, Any]:
        """
        Run the full travel planning workflow.
        
        Args:
            preferences: NormalizedPreferences object
            
        Returns:
            Final itinerary dict
        """
        session_id = str(uuid.uuid4())[:8]
        config = {"configurable": {"thread_id": session_id}}

        initial_state = TravelPlannerState(
            preferences=preferences,
            session_id=session_id,
            flight_options=[],
            hotel_options=[],
            activity_options=[],
            selected_activities=[],
            selected_flight=None,
            selected_hotel=None,
            budget=BudgetBreakdown(total_budget_usd=preferences.budget_usd),
            current_step="initialize",
            completed_steps=[],
            errors=[],
            human_feedback=None,
            requires_human_approval=False,
            messages=[],
            final_itinerary=None,
            itinerary_markdown=None,
        )

        logger.info(f"Starting travel planning session: {session_id}")

        final_state = await self.graph.ainvoke(initial_state, config=config)

        if final_state.get("final_itinerary"):
            logger.info(f"✅ Trip planning complete! Session: {session_id}")
        else:
            logger.error("Trip planning failed to produce an itinerary")

        return final_state

    async def provide_feedback(self, session_id: str, feedback: str) -> dict[str, Any]:
        """
        Resume a paused workflow after human feedback.
        Used in API/async contexts.
        """
        config = {"configurable": {"thread_id": session_id}}
        state_update = {"human_feedback": feedback}

        final_state = await self.graph.ainvoke(state_update, config=config)
        return final_state


# ── Helper Functions ──────────────────────────────────────────────────────────

def _build_approval_summary(flight: Any, hotel: Any, activities: list, budget: Any) -> Panel:
    lines = ["[bold]📋 Trip Summary for Approval[/bold]\n"]

    if flight:
        lines.append(f"✈️  Flight: {flight.airline} {flight.origin}→{flight.destination} "
                     f"| ${flight.price_usd:.0f} | {flight.stops} stop(s)")
    if hotel:
        lines.append(f"🏨 Hotel: {hotel.name} ({hotel.stars}★) | "
                     f"${hotel.price_per_night_usd:.0f}/night | Total: ${hotel.total_price_usd:.0f}")

    lines.append(f"🎯 Activities: {len(activities)} planned | "
                 f"Total: ${sum(a.price_usd for a in activities):.0f}")
    lines.append(f"\n💰 Budget: ${budget.total_spent:.0f} / ${budget.total_budget_usd:.0f} "
                 f"({budget.utilization_percent:.0f}% utilized)")

    if budget.is_over_budget:
        lines.append("[red]⚠️  OVER BUDGET![/red]")
    else:
        lines.append(f"[green]✅ ${budget.remaining:.0f} remaining[/green]")

    return Panel("\n".join(lines), title="Approval Required", border_style="yellow")


def _build_synthesis_context(prefs: Any, flight: Any, hotel: Any, activities: list, budget: Any) -> str:
    return f"""
Create a detailed travel itinerary for:

**Destination:** {prefs.destination}
**Origin:** {prefs.origin}  
**Dates:** {prefs.departure_date} to {prefs.return_date} ({prefs.trip_days} days)
**Travelers:** {prefs.travelers}
**Interests:** {', '.join(prefs.interests) if prefs.interests else 'general travel'}

**Selected Flight:**
{flight.model_dump_json(indent=2) if flight else 'No flight selected'}

**Selected Hotel:**
{hotel.model_dump_json(indent=2) if hotel else 'No hotel selected'}

**Scheduled Activities ({len(activities)} total):**
{chr(10).join(f"Day {a.day_number} ({a.time_of_day}): {a.name} — ${a.price_usd:.0f}" for a in activities[:20])}

**Budget Summary:**
Total Budget: ${budget.total_budget_usd:.0f}
Total Spent: ${budget.total_spent:.0f} ({budget.utilization_percent:.0f}%)
Remaining: ${budget.remaining:.0f}

Create an engaging, complete itinerary with all the details a traveler needs.
"""


def _generate_template_itinerary(prefs: Any, flight: Any, hotel: Any, activities: list, budget: Any) -> str:
    """Fallback template itinerary when LLM is unavailable."""
    lines = [
        f"# ✈️ {prefs.destination} Itinerary",
        f"\n**{prefs.departure_date} — {prefs.return_date}** | "
        f"{prefs.travelers} traveler(s) | ${budget.total_budget_usd:,.0f} budget\n",
        "## ✈️ Flights",
    ]

    if flight:
        lines.append(f"- **Outbound:** {flight.airline} | {flight.origin} → {flight.destination}")
        lines.append(f"  - Departure: {flight.departure_time} | Duration: {flight.duration_minutes}min")
        lines.append(f"  - Cost: ${flight.price_usd:.2f} | Stops: {flight.stops}")

    lines.append("\n## 🏨 Hotel")
    if hotel:
        lines.append(f"- **{hotel.name}** ({hotel.stars}★) — {hotel.location}")
        lines.append(f"  - Check-in: {hotel.check_in} | Check-out: {hotel.check_out}")
        lines.append(f"  - ${hotel.price_per_night_usd:.0f}/night | Total: ${hotel.total_price_usd:.2f}")
        lines.append(f"  - Amenities: {', '.join(hotel.amenities[:5])}")

    lines.append("\n## 📅 Daily Schedule")
    current_day = 0
    for act in sorted(activities, key=lambda x: (x.day_number, x.time_of_day)):
        if act.day_number != current_day:
            current_day = act.day_number
            lines.append(f"\n### Day {current_day}")
        lines.append(f"- **{act.time_of_day.title()}:** {act.name} (${act.price_usd:.0f}, {act.duration_hours}h)")

    lines.extend([
        "\n## 💰 Budget Summary",
        f"| Category | Amount |",
        f"|----------|--------|",
        f"| Flights | ${budget.spent_flights:.2f} |",
        f"| Hotels | ${budget.spent_hotels:.2f} |",
        f"| Activities | ${budget.spent_activities:.2f} |",
        f"| **Total** | **${budget.total_spent:.2f}** |",
        f"| Remaining | ${budget.remaining:.2f} |",
    ])

    return "\n".join(lines)
