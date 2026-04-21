"""
examples/budget_asia_trip.py
-----------------------------
Example: Budget backpacker trip across Southeast Asia, $1,500 for 10 days.
Run: python examples/budget_asia_trip.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.schemas.preferences import NormalizedPreferences
from src.agents.orchestrator import TravelPlannerOrchestrator
from datetime import date, timedelta
from rich.console import Console

console = Console()


async def main():
    console.print("[bold yellow]Example: Budget Asia Adventure[/bold yellow]")

    today = date.today()
    preferences = NormalizedPreferences(
        destination="Bangkok, Thailand",
        origin="Sydney, Australia",
        departure_date=today + timedelta(days=45),
        return_date=today + timedelta(days=55),
        travelers=1,
        budget_usd=1500.0,
        currency="USD",
        preferred_cabin="economy",
        min_hotel_stars=2.5,
        interests=["food", "adventure", "culture", "nature"],
        activity_pace="packed",
        hotel_amenities=["Free WiFi"],
        max_stops=1,
        dietary_restrictions=["vegetarian"],
    )

    orchestrator = TravelPlannerOrchestrator()
    result = await orchestrator.plan_trip(preferences)

    if result.get("itinerary_markdown"):
        with open("bangkok_budget_itinerary.md", "w") as f:
            f.write(result["itinerary_markdown"])
        console.print("\n[green]✅ Saved to bangkok_budget_itinerary.md[/green]")

    budget = result.get("budget")
    if budget:
        console.print(f"\n[bold]Per-day cost: ${budget.total_spent / 10:.0f} USD[/bold]")


if __name__ == "__main__":
    asyncio.run(main())
