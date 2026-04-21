"""
tests/test_agents.py
--------------------
Comprehensive tests for all agents using pytest-asyncio and mocking.
"""

from __future__ import annotations

import json
import pytest
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.schemas.travel_state import (
    ActivityOption, BudgetBreakdown, FlightOption, HotelOption, TravelPlannerState
)
from src.schemas.preferences import NormalizedPreferences


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_preferences() -> NormalizedPreferences:
    today = date.today()
    return NormalizedPreferences(
        destination="Paris, France",
        origin="New York, USA",
        departure_date=today + timedelta(days=30),
        return_date=today + timedelta(days=37),
        travelers=2,
        budget_usd=3000.0,
        interests=["food", "culture"],
        activity_pace="moderate",
        min_hotel_stars=3.0,
        max_stops=1,
    )


@pytest.fixture
def sample_budget() -> BudgetBreakdown:
    return BudgetBreakdown(total_budget_usd=3000.0, buffer_percent=10.0)


@pytest.fixture
def sample_flight() -> FlightOption:
    return FlightOption(
        id="FL001",
        airline="Delta",
        origin="JFK",
        destination="CDG",
        departure_time="2025-06-01T10:00:00",
        arrival_time="2025-06-01T22:00:00",
        duration_minutes=420,
        stops=0,
        price_usd=850.0,
        cabin_class="economy",
    )


@pytest.fixture
def sample_hotel() -> HotelOption:
    return HotelOption(
        id="HTL001",
        name="Hotel Le Marais",
        stars=4.0,
        location="Le Marais",
        price_per_night_usd=180.0,
        total_price_usd=1260.0,
        amenities=["Free WiFi", "Breakfast Included"],
        rating=8.7,
        check_in="2025-06-01",
        check_out="2025-06-08",
    )


@pytest.fixture
def sample_state(sample_preferences, sample_budget) -> TravelPlannerState:
    return TravelPlannerState(
        preferences=sample_preferences,
        session_id="test-001",
        flight_options=[],
        hotel_options=[],
        activity_options=[],
        selected_activities=[],
        selected_flight=None,
        selected_hotel=None,
        budget=sample_budget,
        current_step="initialize",
        completed_steps=[],
        errors=[],
        human_feedback=None,
        requires_human_approval=False,
        messages=[],
        final_itinerary=None,
        itinerary_markdown=None,
    )


# ── Schema Tests ──────────────────────────────────────────────────────────────

class TestBudgetBreakdown:
    def test_total_spent(self):
        budget = BudgetBreakdown(
            total_budget_usd=1000.0,
            spent_flights=400.0,
            spent_hotels=300.0,
            spent_activities=150.0,
        )
        assert budget.total_spent == 850.0

    def test_remaining(self):
        budget = BudgetBreakdown(total_budget_usd=1000.0, spent_flights=400.0)
        assert budget.remaining == 600.0

    def test_effective_budget(self):
        budget = BudgetBreakdown(total_budget_usd=1000.0, buffer_percent=10.0)
        assert budget.effective_budget == 900.0

    def test_is_over_budget(self):
        budget = BudgetBreakdown(total_budget_usd=1000.0, spent_flights=950.0, buffer_percent=10.0)
        assert budget.is_over_budget is True

    def test_utilization_percent(self):
        budget = BudgetBreakdown(total_budget_usd=1000.0, spent_flights=500.0)
        assert budget.utilization_percent == 50.0


class TestPreferences:
    def test_trip_days(self, sample_preferences):
        assert sample_preferences.trip_days == 7

    def test_budget_per_day(self, sample_preferences):
        assert sample_preferences.budget_per_day == pytest.approx(3000.0 / 7, rel=0.01)

    def test_invalid_dates(self):
        today = date.today()
        with pytest.raises(ValueError, match="Return date must be after"):
            NormalizedPreferences(
                destination="Paris",
                origin="New York",
                departure_date=today + timedelta(days=10),
                return_date=today + timedelta(days=5),
                budget_usd=1000.0,
            )


# ── Tool Tests ────────────────────────────────────────────────────────────────

class TestFlightTools:
    def test_search_flights_returns_json(self):
        from src.tools.flight_tools import search_flights
        result = search_flights.run({
            "origin": "New York",
            "destination": "Paris",
            "departure_date": "2025-06-01",
            "return_date": "2025-06-08",
            "travelers": 1,
        })
        data = json.loads(result)
        assert "flights" in data
        assert len(data["flights"]) > 0
        assert data["data_source"] == "mock"

    def test_flights_sorted_by_price(self):
        from src.tools.flight_tools import search_flights
        result = search_flights.run({
            "origin": "JFK",
            "destination": "CDG",
            "departure_date": "2025-07-01",
            "return_date": "2025-07-08",
            "travelers": 2,
        })
        data = json.loads(result)
        prices = [f["price_usd"] for f in data["flights"]]
        assert prices == sorted(prices), "Flights should be sorted by price"

    def test_price_history_tool(self):
        from src.tools.flight_tools import get_flight_price_history
        result = get_flight_price_history.run({
            "origin": "NYC",
            "destination": "PAR",
            "month": "2025-06",
        })
        data = json.loads(result)
        assert "average_price_usd" in data
        assert "best_booking_window" in data


class TestHotelTools:
    def test_search_hotels_returns_options(self):
        from src.tools.hotel_tools import search_hotels
        result = search_hotels(
            destination="Paris",
            check_in="2025-06-01",
            check_out="2025-06-08",
            travelers=2,
        )
        data = json.loads(result)
        assert "hotels" in data
        assert len(data["hotels"]) > 0

    def test_hotel_filter_by_stars(self):
        from src.tools.hotel_tools import search_hotels
        result = search_hotels(
            destination="Tokyo",
            check_in="2025-07-01",
            check_out="2025-07-05",
            min_stars=4.0,
        )
        data = json.loads(result)
        for hotel in data["hotels"]:
            assert hotel["stars"] >= 4.0

    def test_hotel_filter_by_price(self):
        from src.tools.hotel_tools import search_hotels
        result = search_hotels(
            destination="London",
            check_in="2025-08-01",
            check_out="2025-08-04",
            max_budget_per_night=100.0,
        )
        data = json.loads(result)
        for hotel in data["hotels"]:
            assert hotel["price_per_night_usd"] <= 100.0


class TestActivityTools:
    def test_search_activities_returns_days(self):
        from src.tools.activity_tools import search_activities
        result = search_activities.run({
            "destination": "Tokyo",
            "trip_days": 5,
            "interests": "food,culture",
            "daily_budget_usd": 100.0,
            "pace": "moderate",
        })
        data = json.loads(result)
        assert "activities_by_day" in data
        assert len(data["activities_by_day"]) == 5

    def test_activities_respect_pace(self):
        from src.tools.activity_tools import search_activities
        relaxed = search_activities.run({
            "destination": "Bali",
            "trip_days": 3,
            "interests": "relaxation",
            "daily_budget_usd": 80.0,
            "pace": "relaxed",
        })
        packed = search_activities.run({
            "destination": "Bali",
            "trip_days": 3,
            "interests": "adventure",
            "daily_budget_usd": 80.0,
            "pace": "packed",
        })
        r_data = json.loads(relaxed)
        p_data = json.loads(packed)
        assert p_data["summary"]["total_activities"] > r_data["summary"]["total_activities"]


# ── Agent Tests ───────────────────────────────────────────────────────────────

class TestBudgetAgent:
    def test_check_budget_alerts_over_budget(self):
        from src.agents.budget_agent import check_budget_alerts
        budget = BudgetBreakdown(
            total_budget_usd=1000.0,
            spent_flights=1100.0,
            buffer_percent=10.0,
        )
        alerts = check_budget_alerts(budget)
        assert any(a.severity == "critical" for a in alerts)

    def test_check_budget_alerts_clean(self):
        from src.agents.budget_agent import check_budget_alerts
        budget = BudgetBreakdown(
            total_budget_usd=2000.0,
            spent_flights=400.0,
            spent_hotels=600.0,
            spent_activities=200.0,
        )
        alerts = check_budget_alerts(budget)
        critical_alerts = [a for a in alerts if a.severity == "critical"]
        assert len(critical_alerts) == 0

    def test_budget_grade(self):
        from src.agents.budget_agent import calculate_budget_grade
        assert calculate_budget_grade(70.0, True) == "A"
        assert calculate_budget_grade(82.0, True) == "B"
        assert calculate_budget_grade(96.0, True) == "C"
        assert calculate_budget_grade(110.0, False) == "D"
        assert calculate_budget_grade(125.0, False) == "F"


@pytest.mark.asyncio
class TestFlightAgentAsync:
    async def test_run_flight_agent_mock(self, sample_state):
        """Flight agent should populate flight_options with mock data."""
        from src.agents.flight_agent import run_flight_agent

        # Patch CrewAI to avoid real LLM calls
        with patch("src.agents.flight_agent.Crew") as MockCrew:
            mock_crew = MagicMock()
            mock_crew.kickoff.return_value = MagicMock(raw="[]")
            MockCrew.return_value = mock_crew

            result = await run_flight_agent(sample_state)

        assert len(result["flight_options"]) > 0
        assert result["selected_flight"] is not None
        assert result["budget"].spent_flights > 0
        assert "flight_search" in result["completed_steps"]


@pytest.mark.asyncio
class TestHotelAgentAsync:
    async def test_run_hotel_agent_fallback(self, sample_state, sample_flight):
        """Hotel agent falls back to direct search when AutoGen unavailable."""
        from src.agents.hotel_agent import run_hotel_agent

        state = {
            **sample_state,
            "selected_flight": sample_flight,
            "budget": BudgetBreakdown(
                total_budget_usd=3000.0,
                spent_flights=sample_flight.price_usd,
            ),
        }

        result = await run_hotel_agent(state)

        assert len(result["hotel_options"]) > 0
        assert result["selected_hotel"] is not None
        assert "hotel_search" in result["completed_steps"]


@pytest.mark.asyncio
class TestActivityAgentAsync:
    async def test_run_activity_agent_fallback(self, sample_state, sample_flight, sample_hotel):
        from src.agents.activity_agent import run_activity_agent

        state = {
            **sample_state,
            "selected_flight": sample_flight,
            "selected_hotel": sample_hotel,
            "budget": BudgetBreakdown(
                total_budget_usd=3000.0,
                spent_flights=sample_flight.price_usd,
                spent_hotels=sample_hotel.total_price_usd,
            ),
        }

        with patch("src.agents.activity_agent.Crew") as MockCrew:
            mock_crew = MagicMock()
            mock_crew.kickoff.return_value = MagicMock(raw="")
            MockCrew.return_value = mock_crew

            result = await run_activity_agent(state)

        assert len(result["selected_activities"]) > 0
        assert result["budget"].spent_activities > 0
        assert "activity_search" in result["completed_steps"]
