"""
schemas/preferences.py
----------------------
PydanticAI-powered preference validation with intelligent defaults.
"""

from __future__ import annotations

from datetime import date, timedelta
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel


class TripPreferenceInput(BaseModel):
    """Raw input from user — loosely typed, will be normalized."""

    raw_input: str = ""

    # These can be provided directly or parsed from raw_input
    destination: str = ""
    origin: str = ""
    departure_date: str = ""
    return_date: str = ""
    travelers: int = 1
    budget_usd: float = 0.0
    interests: list[str] = Field(default_factory=list)
    preferred_cabin: str = "economy"
    min_hotel_stars: float = 3.0


class NormalizedPreferences(BaseModel):
    """Fully validated and normalized trip preferences."""

    destination: str = Field(..., min_length=2, description="Trip destination city/country")
    origin: str = Field(..., min_length=2, description="Departure city")
    departure_date: date
    return_date: date
    travelers: int = Field(default=1, ge=1, le=20)
    budget_usd: float = Field(..., gt=0, description="Total budget in USD")
    currency: str = "USD"
    preferred_cabin: str = "economy"
    min_hotel_stars: float = Field(default=3.0, ge=1.0, le=5.0)
    interests: list[str] = Field(default_factory=list)
    activity_pace: str = "moderate"
    dietary_restrictions: list[str] = Field(default_factory=list)
    max_stops: int = Field(default=1, ge=0, le=3)

    @field_validator("departure_date", "return_date", mode="before")
    @classmethod
    def parse_date(cls, v):
        if isinstance(v, date):
            return v
        if isinstance(v, str):
            from datetime import datetime
            for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y"]:
                try:
                    return datetime.strptime(v, fmt).date()
                except ValueError:
                    continue
        raise ValueError(f"Cannot parse date: {v}")

    @model_validator(mode="after")
    def validate_dates(self) -> "NormalizedPreferences":
        if self.return_date <= self.departure_date:
            raise ValueError("Return date must be after departure date")
        trip_days = (self.return_date - self.departure_date).days
        if trip_days > 90:
            raise ValueError("Trip cannot exceed 90 days")
        return self

    @property
    def trip_days(self) -> int:
        return (self.return_date - self.departure_date).days

    @property
    def budget_per_day(self) -> float:
        return self.budget_usd / self.trip_days if self.trip_days > 0 else 0


# PydanticAI agent to parse natural language preferences
_preference_parser_agent = Agent(
    model=OpenAIModel("gpt-4o-mini"),
    result_type=NormalizedPreferences,
    system_prompt="""You are a travel preference parser. 
    Extract structured travel preferences from natural language input.
    Always infer reasonable defaults:
    - If no origin given, use "New York, USA"
    - If no dates given, set departure to 30 days from today
    - If no budget given, use 2000 USD per person
    - Convert relative dates ("next month", "this summer") to YYYY-MM-DD format
    - Normalize city names to "City, Country" format
    Today's date: use the current date as reference.
    """,
)


async def parse_preferences_from_text(raw_text: str) -> NormalizedPreferences:
    """Use PydanticAI to parse natural language into structured preferences."""
    result = await _preference_parser_agent.run(raw_text)
    return result.data
