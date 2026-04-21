"""
schemas/travel_state.py
-----------------------
LangGraph state schema — the shared "memory" passed between all agents.
Every agent reads from and writes back to this TypedDict.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
import operator


# ── Sub-schemas ──────────────────────────────────────────────────────────────

class FlightOption(BaseModel):
    """A single flight option returned by the Flight Agent."""
    id: str
    airline: str
    origin: str
    destination: str
    departure_time: str
    arrival_time: str
    duration_minutes: int
    stops: int
    price_usd: float
    cabin_class: str = "economy"
    booking_url: str = ""
    carbon_kg: float | None = None


class HotelOption(BaseModel):
    """A single hotel option returned by the Hotel Agent."""
    id: str
    name: str
    stars: float
    location: str
    price_per_night_usd: float
    total_price_usd: float
    amenities: list[str] = Field(default_factory=list)
    rating: float | None = None
    review_count: int = 0
    booking_url: str = ""
    check_in: str = ""
    check_out: str = ""


class ActivityOption(BaseModel):
    """A single activity/experience returned by the Activity Agent."""
    id: str
    name: str
    category: str          # museum | tour | food | adventure | culture | ...
    description: str
    duration_hours: float
    price_usd: float
    location: str
    day_number: int        # Which day of trip (1-based)
    time_of_day: Literal["morning", "afternoon", "evening", "flexible"] = "flexible"
    booking_url: str = ""
    rating: float | None = None


class BudgetBreakdown(BaseModel):
    """Real-time budget tracking shared across all agents."""
    total_budget_usd: float
    spent_flights: float = 0.0
    spent_hotels: float = 0.0
    spent_activities: float = 0.0
    spent_misc: float = 0.0
    buffer_percent: float = 10.0

    @property
    def total_spent(self) -> float:
        return self.spent_flights + self.spent_hotels + self.spent_activities + self.spent_misc

    @property
    def remaining(self) -> float:
        return self.total_budget_usd - self.total_spent

    @property
    def effective_budget(self) -> float:
        """Budget minus buffer reserve."""
        return self.total_budget_usd * (1 - self.buffer_percent / 100)

    @property
    def is_over_budget(self) -> bool:
        return self.total_spent > self.effective_budget

    @property
    def utilization_percent(self) -> float:
        return (self.total_spent / self.total_budget_usd) * 100 if self.total_budget_usd > 0 else 0


class UserPreferences(BaseModel):
    """Captured user preferences driving all agent decisions."""
    destination: str
    origin: str
    departure_date: str          # YYYY-MM-DD
    return_date: str             # YYYY-MM-DD
    travelers: int = 1
    budget_usd: float
    currency: str = "USD"

    # Flight preferences
    preferred_cabin: Literal["economy", "premium_economy", "business", "first"] = "economy"
    max_stops: int = 1
    preferred_airlines: list[str] = Field(default_factory=list)

    # Hotel preferences
    min_hotel_stars: float = 3.0
    preferred_neighborhoods: list[str] = Field(default_factory=list)
    hotel_amenities: list[str] = Field(default_factory=list)

    # Activity preferences
    interests: list[str] = Field(default_factory=list)   # ["food", "history", "adventure"]
    activity_pace: Literal["relaxed", "moderate", "packed"] = "moderate"
    excluded_activities: list[str] = Field(default_factory=list)

    # General
    dietary_restrictions: list[str] = Field(default_factory=list)
    accessibility_needs: str = ""


# ── Main LangGraph State ──────────────────────────────────────────────────────

class TravelPlannerState(TypedDict):
    """
    The central state object passed through the LangGraph workflow.
    Each node (agent) reads and updates this state.
    """

    # ── Core inputs ──
    preferences: UserPreferences
    session_id: str

    # ── Agent outputs ──
    flight_options: list[FlightOption]
    selected_flight: FlightOption | None

    hotel_options: list[HotelOption]
    selected_hotel: HotelOption | None

    activity_options: list[ActivityOption]
    selected_activities: list[ActivityOption]

    # ── Budget state ──
    budget: BudgetBreakdown

    # ── Workflow control ──
    current_step: str
    completed_steps: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    human_feedback: str | None
    requires_human_approval: bool

    # ── Messages (for AutoGen hotel agent) ──
    messages: Annotated[list[Any], add_messages]

    # ── Final output ──
    final_itinerary: dict[str, Any] | None
    itinerary_markdown: str | None
