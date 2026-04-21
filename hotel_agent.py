"""
agents/hotel_agent.py
---------------------
AutoGen-based Hotel Agent using conversational multi-agent dialogue.
An AssistantAgent and UserProxyAgent negotiate hotel preferences
through back-and-forth conversation.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from ..config.prompts import HOTEL_AGENT_SYSTEM_PROMPT, HOTEL_REFINEMENT_PROMPT
from ..config.settings import get_settings
from ..schemas.travel_state import HotelOption, TravelPlannerState
from ..tools.hotel_tools import get_hotel_details, search_hotels

settings = get_settings()


def _get_autogen_config() -> dict:
    """Build AutoGen LLM configuration."""
    if settings.openai_api_key:
        return {
            "config_list": [{
                "model": settings.default_llm_model,
                "api_key": settings.openai_api_key,
            }],
            "temperature": 0.2,
        }
    elif settings.anthropic_api_key:
        return {
            "config_list": [{
                "model": "claude-3-5-haiku-20241022",
                "api_key": settings.anthropic_api_key,
                "api_type": "anthropic",
            }],
            "temperature": 0.2,
        }
    else:
        raise ValueError("No LLM API key configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env")


# ── AutoGen Agent Setup ───────────────────────────────────────────────────────

def create_hotel_crew(preferences: dict, remaining_budget: float) -> tuple:
    """
    Create an AutoGen hotel agent crew.
    Returns (assistant_agent, user_proxy_agent)
    """
    try:
        from autogen import AssistantAgent, UserProxyAgent

        llm_config = _get_autogen_config()

        # The main hotel search agent
        hotel_assistant = AssistantAgent(
            name="HotelSearchAgent",
            system_message=f"""
{HOTEL_AGENT_SYSTEM_PROMPT}

Current trip context:
- Destination: {preferences['destination']}
- Check-in: {preferences['check_in']}
- Check-out: {preferences['check_out']}
- Travelers: {preferences['travelers']}
- Budget remaining for hotel: ${remaining_budget:.0f} USD
- Min stars required: {preferences.get('min_stars', 3.0)}
- Required amenities: {preferences.get('amenities', [])}

Use the search_hotels function to find options, then present them clearly.
After presenting options, ask if the traveler wants to refine their search.
When satisfied, output the final selection as JSON with key 'selected_hotel'.
""",
            llm_config=llm_config,
        )

        # A proxy that executes function calls and routes messages
        user_proxy = UserProxyAgent(
            name="TravelPlannerProxy",
            human_input_mode="NEVER",  # Fully automated; LangGraph handles human input
            max_consecutive_auto_reply=settings.max_agent_iterations,
            code_execution_config=False,
            function_map={
                "search_hotels": lambda **kwargs: search_hotels(**kwargs),
                "get_hotel_details": lambda hotel_id: get_hotel_details(hotel_id),
            },
            system_message="""You are a travel planning assistant proxy.
            Execute hotel search functions when requested.
            After finding hotels, provide brief feedback and ask for refinements.
            Stop the conversation when you have a satisfactory hotel selection.
            Termination phrase: HOTEL_SELECTED
            """,
        )

        return hotel_assistant, user_proxy

    except ImportError:
        logger.warning("AutoGen not installed. Using simplified hotel search.")
        return None, None


# ── Simplified fallback (no AutoGen installed) ────────────────────────────────

def _simple_hotel_search(preferences: dict, remaining_budget: float) -> list[HotelOption]:
    """Direct hotel search without AutoGen conversation loop."""
    from datetime import datetime

    check_in = preferences["check_in"]
    check_out = preferences["check_out"]

    try:
        nights = (
            datetime.strptime(check_out, "%Y-%m-%d") -
            datetime.strptime(check_in, "%Y-%m-%d")
        ).days
    except ValueError:
        nights = 3

    max_per_night = (remaining_budget / nights) if nights > 0 else remaining_budget

    result_json = search_hotels(
        destination=preferences["destination"],
        check_in=check_in,
        check_out=check_out,
        travelers=preferences.get("travelers", 1),
        min_stars=preferences.get("min_stars", 3.0),
        max_budget_per_night=max_per_night,
    )

    data = json.loads(result_json)
    hotels = data.get("hotels", [])
    return [HotelOption(**h) for h in hotels[:3]]


# ── Main Execution Function ───────────────────────────────────────────────────

async def run_hotel_agent(state: TravelPlannerState) -> TravelPlannerState:
    """
    Execute the AutoGen Hotel Agent and update state.
    Called as a LangGraph node.
    """
    logger.info("🏨 Hotel Agent starting...")
    prefs = state["preferences"]
    budget = state["budget"]

    # Remaining budget after flights
    remaining_for_hotel = budget.remaining
    # Hotels typically ~40% of total budget
    hotel_budget = min(remaining_for_hotel * 0.65, budget.effective_budget * 0.40)

    pref_dict = {
        "destination": prefs.destination,
        "check_in": str(prefs.departure_date),
        "check_out": str(prefs.return_date),
        "travelers": prefs.travelers,
        "min_stars": prefs.min_hotel_stars,
        "amenities": prefs.hotel_amenities,
        "neighborhoods": prefs.preferred_neighborhoods,
    }

    hotel_options = []
    selected_hotel = None

    try:
        assistant, proxy = create_hotel_crew(pref_dict, hotel_budget)

        if assistant and proxy:
            # Run AutoGen conversation
            initial_message = f"""
            Please search for hotels in {prefs.destination} for {prefs.travelers} traveler(s).
            Check-in: {prefs.departure_date}, Check-out: {prefs.return_date}
            Budget for accommodation: ${hotel_budget:.0f} USD total
            Preferences: {prefs.min_hotel_stars}+ stars, amenities: {prefs.hotel_amenities}
            
            Start by searching for available options.
            """

            proxy.initiate_chat(
                assistant,
                message=initial_message,
                max_turns=6,
            )

            # Extract hotel selection from conversation
            conversation_history = proxy.chat_messages.get(assistant, [])
            hotel_options, selected_hotel = _extract_hotels_from_conversation(
                conversation_history, pref_dict, hotel_budget
            )
        else:
            # Fallback without AutoGen
            hotel_options = _simple_hotel_search(pref_dict, hotel_budget)
            selected_hotel = hotel_options[0] if hotel_options else None

    except Exception as e:
        logger.error(f"Hotel Agent error: {e}")
        hotel_options = _simple_hotel_search(pref_dict, hotel_budget)
        selected_hotel = hotel_options[0] if hotel_options else None
        state["errors"] = state.get("errors", []) + [f"Hotel Agent warning: {str(e)}"]

    # Update budget
    if selected_hotel:
        budget.spent_hotels = selected_hotel.total_price_usd

    logger.info(f"✅ Hotel Agent complete. Selected: {selected_hotel and selected_hotel.name}")

    return {
        **state,
        "hotel_options": hotel_options,
        "selected_hotel": selected_hotel,
        "budget": budget,
        "current_step": "activity_search",
        "completed_steps": ["hotel_search"],
    }


def _extract_hotels_from_conversation(
    messages: list[dict],
    pref_dict: dict,
    hotel_budget: float,
) -> tuple[list[HotelOption], HotelOption | None]:
    """Parse AutoGen conversation to extract hotel recommendations."""
    # Look for JSON in the last few messages
    for msg in reversed(messages):
        content = msg.get("content", "")
        try:
            import re
            json_match = re.search(r'"selected_hotel"\s*:\s*(\{[^}]+\})', content, re.DOTALL)
            if json_match:
                hotel_data = json.loads(json_match.group(1))
                selected = HotelOption(**hotel_data)
                return [selected], selected
        except Exception:
            pass

    # Fallback to direct search
    hotels = _simple_hotel_search(pref_dict, hotel_budget)
    return hotels, hotels[0] if hotels else None
