"""
tools/activity_tools.py
-----------------------
Activity and experience search tools.
Used by the CrewAI Activity Agent.
"""

from __future__ import annotations

import json
import random
from typing import Any

from crewai.tools import tool
from loguru import logger

from ..config.settings import get_settings

settings = get_settings()


ACTIVITY_TEMPLATES = {
    "food": [
        ("Food Tour: Local Markets & Street Food", 3.0, 45, "morning"),
        ("Cooking Class: Traditional Cuisine", 4.0, 80, "afternoon"),
        ("Fine Dining Tasting Menu", 3.0, 150, "evening"),
        ("Wine & Cheese Tasting", 2.0, 55, "afternoon"),
        ("Night Food Market Tour", 2.5, 35, "evening"),
    ],
    "history": [
        ("Old Town Walking Tour", 2.5, 25, "morning"),
        ("National Museum Visit", 3.0, 20, "morning"),
        ("Ancient Ruins Guided Tour", 4.0, 45, "morning"),
        ("Historical Quarter Self-Guided Walk", 2.0, 0, "afternoon"),
        ("War Memorial & Museum", 2.5, 15, "afternoon"),
    ],
    "adventure": [
        ("Day Hike to Scenic Viewpoint", 6.0, 30, "morning"),
        ("Kayaking / Paddleboarding", 3.0, 65, "morning"),
        ("Rock Climbing Intro Class", 4.0, 85, "morning"),
        ("Cycling Tour of Countryside", 5.0, 55, "morning"),
        ("Ziplining Adventure", 3.0, 90, "afternoon"),
    ],
    "culture": [
        ("Contemporary Art Museum", 2.5, 18, "morning"),
        ("Traditional Dance / Music Show", 2.0, 50, "evening"),
        ("Local Craft Workshop", 3.0, 65, "afternoon"),
        ("Religious Site & Temple Visit", 1.5, 10, "morning"),
        ("Street Art Walking Tour", 2.0, 20, "afternoon"),
    ],
    "relaxation": [
        ("Spa & Wellness Day", 4.0, 120, "afternoon"),
        ("Sunset River Cruise", 2.0, 40, "evening"),
        ("Botanical Gardens Stroll", 2.0, 12, "morning"),
        ("Beach Day", 6.0, 15, "morning"),
        ("Rooftop Bar Sundowner", 2.0, 35, "evening"),
    ],
    "nature": [
        ("National Park Day Tour", 8.0, 55, "morning"),
        ("Wildlife Watching Safari", 4.0, 85, "morning"),
        ("Waterfall Trek", 5.0, 25, "morning"),
        ("Snorkeling / Diving Trip", 4.0, 70, "morning"),
        ("Sunrise Mountain Trek", 5.0, 40, "morning"),
    ],
}


def _generate_activities(
    destination: str,
    trip_days: int,
    interests: list[str],
    budget_per_day: float,
    pace: str = "moderate",
) -> list[dict[str, Any]]:
    """Generate a day-by-day activity schedule."""
    random.seed(hash(f"{destination}{trip_days}{''.join(interests)}"))

    if not interests:
        interests = ["food", "culture", "history"]

    # Activities per day based on pace
    pace_map = {"relaxed": 2, "moderate": 3, "packed": 4}
    activities_per_day = pace_map.get(pace, 3)

    all_activities = []
    activity_id = 1

    for day in range(1, trip_days + 1):
        day_interests = interests[day % len(interests):] + interests[:day % len(interests)]
        day_budget = budget_per_day * 0.3  # ~30% of daily budget on activities

        times_used = set()
        for slot in range(activities_per_day):
            interest = day_interests[slot % len(day_interests)]
            templates = ACTIVITY_TEMPLATES.get(interest, ACTIVITY_TEMPLATES["culture"])
            template = random.choice(templates)

            name, duration, base_price, preferred_time = template

            # Avoid time conflicts
            time_of_day = preferred_time
            if time_of_day in times_used:
                available = {"morning", "afternoon", "evening"} - times_used
                time_of_day = list(available)[0] if available else "flexible"

            times_used.add(time_of_day)
            price = round(base_price * random.uniform(0.85, 1.15), 2)

            all_activities.append({
                "id": f"ACT{activity_id:03d}",
                "name": f"{name} in {destination.split(',')[0]}",
                "category": interest,
                "description": f"An immersive {interest} experience in {destination.split(',')[0]}. "
                               f"Suitable for all levels, this {duration:.0f}-hour activity "
                               f"is a highlight of any visit.",
                "duration_hours": duration,
                "price_usd": price,
                "location": destination.split(",")[0],
                "day_number": day,
                "time_of_day": time_of_day,
                "booking_url": f"https://example.com/activity/ACT{activity_id:03d}",
                "rating": round(random.uniform(4.0, 5.0), 1),
            })

            activity_id += 1

    return all_activities


@tool("Search Activities and Experiences")
def search_activities(
    destination: str,
    trip_days: int,
    interests: str,
    daily_budget_usd: float,
    pace: str = "moderate",
) -> str:
    """
    Search and curate activities and experiences for each day of the trip.

    Args:
        destination: City/region for activities
        trip_days: Total number of trip days
        interests: Comma-separated interests (food,history,adventure,culture,relaxation,nature)
        daily_budget_usd: Available budget per day in USD
        pace: Trip pace - relaxed/moderate/packed

    Returns:
        JSON string with day-by-day activity schedule
    """
    logger.info(f"Searching activities in {destination} for {trip_days} days")

    interest_list = [i.strip().lower() for i in interests.split(",")]
    valid_interests = [i for i in interest_list if i in ACTIVITY_TEMPLATES]
    if not valid_interests:
        valid_interests = ["food", "culture", "history"]

    activities = _generate_activities(
        destination=destination,
        trip_days=trip_days,
        interests=valid_interests,
        budget_per_day=daily_budget_usd,
        pace=pace,
    )

    # Group by day
    by_day = {}
    for act in activities:
        day = act["day_number"]
        by_day.setdefault(day, []).append(act)

    # Sort each day by time
    time_order = {"morning": 0, "afternoon": 1, "evening": 2, "flexible": 3}
    for day_acts in by_day.values():
        day_acts.sort(key=lambda x: time_order.get(x["time_of_day"], 3))

    total_activity_cost = sum(a["price_usd"] for a in activities)

    return json.dumps({
        "activities_by_day": by_day,
        "all_activities": activities,
        "total_cost_usd": round(total_activity_cost, 2),
        "summary": {
            "total_activities": len(activities),
            "trip_days": trip_days,
            "pace": pace,
            "interests_covered": valid_interests,
        },
        "data_source": "mock",
    }, indent=2)


@tool("Get Activity Reviews")
def get_activity_reviews(activity_id: str) -> str:
    """Fetch reviews and ratings for a specific activity."""
    random.seed(hash(activity_id))
    reviews = [
        {"rating": round(random.uniform(3.5, 5.0), 1),
         "comment": random.choice([
             "Absolutely fantastic experience! Highly recommend.",
             "Great guide, very knowledgeable and friendly.",
             "Worth every penny. Would do it again.",
             "Good experience overall, a bit rushed though.",
             "Exceeded expectations. A true highlight of our trip.",
         ]),
         "date": "2024-11", "source": "TripAdvisor"}
        for _ in range(5)
    ]
    return json.dumps({"activity_id": activity_id, "reviews": reviews}, indent=2)
