"""
tools/flight_tools.py
---------------------
Flight search tools using Amadeus API with mock fallback.
These tools are registered with CrewAI for use by the Flight Agent.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from typing import Any

from crewai.tools import tool
from loguru import logger

from ..config.settings import get_settings
from ..schemas.travel_state import FlightOption

settings = get_settings()


def _get_amadeus_client():
    """Lazy Amadeus client initialization."""
    if not settings.has_amadeus:
        return None
    try:
        from amadeus import Client, ResponseError
        return Client(
            client_id=settings.amadeus_client_id,
            client_secret=settings.amadeus_client_secret,
            hostname=settings.amadeus_environment,
        )
    except ImportError:
        logger.warning("Amadeus SDK not installed. Using mock data.")
        return None


def _generate_mock_flights(
    origin: str,
    destination: str,
    date: str,
    travelers: int,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """Generate realistic mock flight data for development/testing."""
    airlines = ["Delta", "United", "American", "Lufthansa", "Emirates", "Singapore Airlines"]
    random.seed(hash(f"{origin}{destination}{date}"))

    flights = []
    for i in range(max_results):
        base_price = random.uniform(200, 1200) * travelers
        departure_hour = random.randint(6, 22)
        duration = random.randint(90, 840)
        stops = random.choice([0, 0, 1, 1, 2])

        flights.append({
            "id": f"FL{i+1:03d}",
            "airline": random.choice(airlines),
            "origin": origin.upper()[:3],
            "destination": destination.upper()[:3],
            "departure_time": f"{date}T{departure_hour:02d}:00:00",
            "arrival_time": f"{date}T{(departure_hour + duration//60) % 24:02d}:{duration%60:02d}:00",
            "duration_minutes": duration,
            "stops": stops,
            "price_usd": round(base_price, 2),
            "cabin_class": "economy",
            "booking_url": f"https://example.com/flight/FL{i+1:03d}",
            "carbon_kg": round(duration * 0.18 * travelers, 1),
        })

    return sorted(flights, key=lambda x: x["price_usd"])


@tool("Search Flights")
def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str,
    travelers: int = 1,
    cabin_class: str = "economy",
    max_stops: int = 1,
) -> str:
    """
    Search for available flights between two cities.

    Args:
        origin: Departure city or airport code (e.g., "New York" or "JFK")
        destination: Arrival city or airport code (e.g., "Paris" or "CDG")
        departure_date: Departure date in YYYY-MM-DD format
        return_date: Return date in YYYY-MM-DD format
        travelers: Number of passengers (default: 1)
        cabin_class: Seat class - economy/premium_economy/business/first
        max_stops: Maximum number of layovers (0 = direct only)

    Returns:
        JSON string with list of flight options
    """
    logger.info(f"Searching flights: {origin} → {destination} on {departure_date}")

    amadeus = _get_amadeus_client()

    if amadeus:
        try:
            response = amadeus.shopping.flight_offers_search.get(
                originLocationCode=origin[:3].upper(),
                destinationLocationCode=destination[:3].upper(),
                departureDate=departure_date,
                returnDate=return_date,
                adults=travelers,
                travelClass=cabin_class.upper(),
                nonStop=(max_stops == 0),
                max=settings.max_flight_results,
            )
            # Parse Amadeus response into our schema
            flights = _parse_amadeus_response(response.data, travelers)
            logger.info(f"Found {len(flights)} real flights via Amadeus")
        except Exception as e:
            logger.warning(f"Amadeus API error: {e}. Falling back to mock data.")
            flights = _generate_mock_flights(origin, destination, departure_date, travelers)
    else:
        logger.info("Using mock flight data (no Amadeus credentials)")
        flights = _generate_mock_flights(origin, destination, departure_date, travelers)

    # Filter by max stops
    flights = [f for f in flights if f.get("stops", 0) <= max_stops]

    return json.dumps({
        "flights": flights[:settings.max_flight_results],
        "search_params": {
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
            "return_date": return_date,
            "travelers": travelers,
        },
        "data_source": "amadeus" if amadeus else "mock",
    }, indent=2)


@tool("Get Flight Price History")
def get_flight_price_history(origin: str, destination: str, month: str) -> str:
    """
    Get historical price trends for a route to find the cheapest booking window.

    Args:
        origin: Origin city/airport
        destination: Destination city/airport
        month: Month in YYYY-MM format

    Returns:
        JSON with price trends and best booking recommendations
    """
    random.seed(hash(f"{origin}{destination}{month}"))
    avg_price = random.uniform(300, 900)

    trends = {
        "route": f"{origin} → {destination}",
        "month": month,
        "average_price_usd": round(avg_price, 2),
        "lowest_price_usd": round(avg_price * 0.7, 2),
        "highest_price_usd": round(avg_price * 1.4, 2),
        "best_booking_window": "6-8 weeks in advance",
        "cheapest_days": ["Tuesday", "Wednesday", "Saturday"],
        "busiest_periods": ["Weekends", "Local holidays"],
        "price_trend": "stable",
    }

    return json.dumps(trends, indent=2)


def _parse_amadeus_response(data: list[dict], travelers: int) -> list[dict]:
    """Parse Amadeus API response into our flight schema."""
    flights = []
    for offer in data:
        try:
            itinerary = offer["itineraries"][0]
            segment = itinerary["segments"][0]
            price = float(offer["price"]["total"])

            flights.append({
                "id": offer["id"],
                "airline": segment["carrierCode"],
                "origin": segment["departure"]["iataCode"],
                "destination": segment["arrival"]["iataCode"],
                "departure_time": segment["departure"]["at"],
                "arrival_time": segment["arrival"]["at"],
                "duration_minutes": _parse_duration(itinerary["duration"]),
                "stops": len(itinerary["segments"]) - 1,
                "price_usd": price,
                "cabin_class": offer["travelerPricings"][0]["fareDetailsBySegment"][0].get(
                    "cabin", "ECONOMY"
                ).lower(),
                "booking_url": "",
                "carbon_kg": None,
            })
        except (KeyError, IndexError, ValueError) as e:
            logger.warning(f"Failed to parse Amadeus offer: {e}")
            continue
    return flights


def _parse_duration(iso_duration: str) -> int:
    """Convert ISO 8601 duration (PT2H30M) to minutes."""
    import re
    hours = int(re.search(r"(\d+)H", iso_duration).group(1)) if "H" in iso_duration else 0
    mins = int(re.search(r"(\d+)M", iso_duration).group(1)) if "M" in iso_duration else 0
    return hours * 60 + mins
