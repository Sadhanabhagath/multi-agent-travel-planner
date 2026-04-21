"""
tools/hotel_tools.py
--------------------
Hotel search tools with mock data fallback.
Used by the AutoGen Hotel Agent.
"""

from __future__ import annotations

import json
import random
from typing import Any

from loguru import logger

from ..config.settings import get_settings

settings = get_settings()


HOTEL_AMENITIES_POOL = [
    "Free WiFi", "Pool", "Gym", "Spa", "Restaurant", "Bar", "Room Service",
    "Concierge", "Business Center", "Pet Friendly", "Parking", "Airport Shuttle",
    "Breakfast Included", "Air Conditioning", "Balcony", "Sea View", "City View",
]

NEIGHBORHOODS = {
    "paris": ["Le Marais", "Saint-Germain", "Montmartre", "Champs-Élysées", "Opera"],
    "tokyo": ["Shinjuku", "Shibuya", "Asakusa", "Ginza", "Roppongi"],
    "new york": ["Midtown", "Upper West Side", "Brooklyn", "Lower East Side", "SoHo"],
    "london": ["Covent Garden", "Soho", "South Bank", "Shoreditch", "Notting Hill"],
    "default": ["City Center", "Downtown", "Old Town", "Business District", "Waterfront"],
}


def _generate_mock_hotels(
    destination: str,
    check_in: str,
    check_out: str,
    travelers: int,
    nights: int,
    min_stars: float = 3.0,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """Generate realistic mock hotel options."""
    random.seed(hash(f"{destination}{check_in}{travelers}"))

    dest_lower = destination.lower()
    neighborhoods = NEIGHBORHOODS.get(dest_lower, NEIGHBORHOODS["default"])

    hotels = []
    for i in range(max_results):
        stars = round(random.uniform(max(min_stars, 2.5), 5.0) * 2) / 2  # 0.5 increments
        price_per_night = random.uniform(60, 600) * (stars / 3.0)
        total_price = round(price_per_night * nights * (1 + 0.15), 2)  # +15% taxes

        num_amenities = random.randint(4, 10)
        amenities = random.sample(HOTEL_AMENITIES_POOL, num_amenities)

        hotels.append({
            "id": f"HTL{i+1:03d}",
            "name": f"Hotel {'★' * int(stars)} {random.choice(['Grand', 'Royal', 'Central', 'Palace', 'Urban', 'Boutique'])} {destination.split(',')[0]}",
            "stars": stars,
            "location": random.choice(neighborhoods),
            "price_per_night_usd": round(price_per_night, 2),
            "total_price_usd": total_price,
            "amenities": amenities,
            "rating": round(random.uniform(7.0, 9.8), 1),
            "review_count": random.randint(200, 5000),
            "booking_url": f"https://example.com/hotel/HTL{i+1:03d}",
            "check_in": check_in,
            "check_out": check_out,
        })

    return sorted(hotels, key=lambda x: x["price_per_night_usd"])


def search_hotels(
    destination: str,
    check_in: str,
    check_out: str,
    travelers: int = 1,
    min_stars: float = 3.0,
    amenities: list[str] | None = None,
    max_budget_per_night: float | None = None,
) -> str:
    """
    Search for hotels at the destination.

    Args:
        destination: City/region for hotel search
        check_in: Check-in date YYYY-MM-DD
        check_out: Check-out date YYYY-MM-DD
        travelers: Number of guests
        min_stars: Minimum star rating (1.0-5.0)
        amenities: Required amenities list
        max_budget_per_night: Maximum price per night in USD

    Returns:
        JSON string with hotel options
    """
    from datetime import datetime
    try:
        nights = (
            datetime.strptime(check_out, "%Y-%m-%d") -
            datetime.strptime(check_in, "%Y-%m-%d")
        ).days
    except ValueError:
        nights = 3

    logger.info(f"Searching hotels in {destination} for {nights} nights")

    hotels = _generate_mock_hotels(
        destination, check_in, check_out, travelers, nights,
        min_stars=min_stars,
        max_results=settings.max_hotel_results,
    )

    # Apply amenity filter
    if amenities:
        filtered = []
        for hotel in hotels:
            hotel_amenities_lower = [a.lower() for a in hotel["amenities"]]
            if all(req.lower() in hotel_amenities_lower for req in amenities):
                filtered.append(hotel)
        hotels = filtered if filtered else hotels  # Fallback to unfiltered

    # Apply price filter
    if max_budget_per_night:
        hotels = [h for h in hotels if h["price_per_night_usd"] <= max_budget_per_night]

    return json.dumps({
        "hotels": hotels[:settings.max_hotel_results],
        "search_params": {
            "destination": destination,
            "check_in": check_in,
            "check_out": check_out,
            "travelers": travelers,
            "nights": nights,
        },
        "data_source": "mock",
    }, indent=2)


def get_hotel_details(hotel_id: str) -> str:
    """Get detailed information about a specific hotel."""
    return json.dumps({
        "hotel_id": hotel_id,
        "detailed_description": "A wonderful property with excellent service.",
        "cancellation_policy": "Free cancellation up to 48 hours before check-in.",
        "nearby_attractions": ["City Museum (0.3km)", "Central Park (0.5km)", "Main Train Station (0.8km)"],
        "transportation": {
            "airport_distance_km": 25,
            "metro_station": "Central Metro (200m walk)",
            "taxi_to_airport_usd": 35,
        },
        "data_source": "mock",
    }, indent=2)
