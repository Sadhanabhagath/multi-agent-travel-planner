"""
examples/paris_weekend.py
--------------------------
Example: A romantic Paris weekend for 2, $2,000 budget.
Run: python examples/paris_weekend.py
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
    console.print("[bold cyan]Example: Paris Weekend for 2[/bold cyan]")

    today = date.today()
    preferences = NormalizedPreferences(
        destination="Paris, France",
        origin="London, UK",
        departure_date=today + timedelta(days=21),
        return_date=today + timedelta(days=24),
        travelers=2,
        budget_usd=2000.0,
        currency="USD",
        preferred_cabin="economy",
        min_hotel_stars=4.0,
        interests=["food", "culture", "history"],
        activity_pace="moderate",
        hotel_amenities=["Free WiFi", "Breakfast Included"],
        max_stops=0,  # Direct flights only
    )

    orchestrator = TravelPlannerOrchestrator()
    result = await orchestrator.plan_trip(preferences)

    if result.get("itinerary_markdown"):
        # Save to file
        with open("paris_weekend_itinerary.md", "w") as f:
            f.write(result["itinerary_markdown"])
        console.print("\n[green]✅ Itinerary saved to paris_weekend_itinerary.md[/green]")

    budget = result.get("budget")
    if budget:
        console.print(f"\n[bold]Budget Summary:[/bold]")
        console.print(f"  Total: ${budget.total_budget_usd:.0f}")
        console.print(f"  Spent: ${budget.total_spent:.0f} ({budget.utilization_percent:.0f}%)")
        console.print(f"  Remaining: ${budget.remaining:.0f}")


if __name__ == "__main__":
    asyncio.run(main())
