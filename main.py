"""
main.py
-------
CLI entry point for the Multi-Agent Travel Planner.
Supports both interactive mode and direct argument passing.

Usage:
    python main.py
    python main.py --destination "Tokyo, Japan" --origin "New York, USA" \
                   --budget 3500 --days 7 --travelers 2 --interests "food,culture,history"
"""

from __future__ import annotations

import asyncio
import sys
from datetime import date, timedelta
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from loguru import logger

app = typer.Typer(
    name="travel-planner",
    help="✈️  Multi-Agent AI Travel Planner — powered by LangGraph, CrewAI, AutoGen & PydanticAI",
    add_completion=False,
)
console = Console()


def _setup_logging(log_level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}",
    )
    logger.add("logs/travel_planner.log", rotation="10 MB", retention="7 days", level="DEBUG")


def _collect_preferences_interactive() -> dict:
    """Interactive CLI wizard to collect trip preferences."""
    console.print(Panel(
        "[bold cyan]✈️  Welcome to the Multi-Agent Travel Planner[/bold cyan]\n"
        "[dim]Powered by LangGraph • CrewAI • AutoGen • PydanticAI[/dim]",
        border_style="blue",
    ))
    console.print()

    destination = Prompt.ask("🌍 [bold]Destination[/bold]", default="Paris, France")
    origin = Prompt.ask("🏠 [bold]Departing from[/bold]", default="New York, USA")

    today = date.today()
    default_depart = (today + timedelta(days=30)).strftime("%Y-%m-%d")
    default_return = (today + timedelta(days=37)).strftime("%Y-%m-%d")

    departure_date = Prompt.ask("📅 [bold]Departure date[/bold] (YYYY-MM-DD)", default=default_depart)
    return_date = Prompt.ask("📅 [bold]Return date[/bold]   (YYYY-MM-DD)", default=default_return)
    travelers = IntPrompt.ask("👥 [bold]Number of travelers[/bold]", default=1)
    budget = FloatPrompt.ask("💰 [bold]Total budget (USD)[/bold]", default=2500.0)

    console.print("\n[dim]Interests (choose from: food, history, adventure, culture, relaxation, nature)[/dim]")
    interests_raw = Prompt.ask("🎯 [bold]Your interests[/bold] (comma-separated)", default="food,culture,history")
    interests = [i.strip().lower() for i in interests_raw.split(",")]

    cabin = Prompt.ask(
        "✈️  [bold]Cabin class[/bold]",
        choices=["economy", "premium_economy", "business", "first"],
        default="economy",
    )
    min_stars = FloatPrompt.ask("⭐ [bold]Minimum hotel stars[/bold]", default=3.0)
    pace = Prompt.ask(
        "🏃 [bold]Trip pace[/bold]",
        choices=["relaxed", "moderate", "packed"],
        default="moderate",
    )

    console.print()

    return {
        "destination": destination,
        "origin": origin,
        "departure_date": departure_date,
        "return_date": return_date,
        "travelers": travelers,
        "budget_usd": budget,
        "interests": interests,
        "preferred_cabin": cabin,
        "min_hotel_stars": min_stars,
        "activity_pace": pace,
        "hotel_amenities": [],
        "preferred_neighborhoods": [],
        "dietary_restrictions": [],
        "accessibility_needs": "",
        "max_stops": 1,
        "preferred_airlines": [],
        "excluded_activities": [],
        "currency": "USD",
    }


async def _run_planner(pref_dict: dict) -> None:
    """Core async function to run the travel planner."""
    from src.schemas.travel_state import TravelPlannerState, BudgetBreakdown
    from src.agents.orchestrator import TravelPlannerOrchestrator

    # Build preferences object
    try:
        from src.schemas.preferences import NormalizedPreferences
        preferences = NormalizedPreferences(**pref_dict)
    except Exception as e:
        console.print(f"[red]❌ Invalid preferences: {e}[/red]")
        raise typer.Exit(1)

    # Run the orchestrator
    orchestrator = TravelPlannerOrchestrator()

    with console.status("[bold green]🤖 Agents are planning your trip...[/bold green]", spinner="dots"):
        result = await orchestrator.plan_trip(preferences)

    if result.get("final_itinerary"):
        console.print("\n[bold green]✅ Trip planning complete![/bold green]\n")

        # Save markdown itinerary to file
        itinerary_md = result.get("itinerary_markdown", "")
        if itinerary_md:
            output_file = f"itinerary_{preferences.destination.split(',')[0].replace(' ', '_').lower()}.md"
            with open(output_file, "w") as f:
                f.write(itinerary_md)
            console.print(f"[dim]📄 Itinerary saved to: {output_file}[/dim]")

        # Print errors/warnings if any
        errors = result.get("errors", [])
        if errors:
            console.print(f"\n[yellow]⚠️  Notes ({len(errors)}):[/yellow]")
            for err in errors:
                console.print(f"  [dim]• {err}[/dim]")
    else:
        console.print("[red]❌ Planning failed — no itinerary was generated.[/red]")
        raise typer.Exit(1)


@app.command()
def main(
    destination: Optional[str] = typer.Option(None, "--destination", "-d", help="Trip destination"),
    origin: Optional[str] = typer.Option(None, "--origin", "-o", help="Departure city"),
    budget: Optional[float] = typer.Option(None, "--budget", "-b", help="Total budget in USD"),
    days: Optional[int] = typer.Option(None, "--days", help="Number of trip days"),
    travelers: int = typer.Option(1, "--travelers", "-t", help="Number of travelers"),
    interests: Optional[str] = typer.Option(None, "--interests", "-i", help="Comma-separated interests"),
    cabin: str = typer.Option("economy", "--cabin", help="Cabin class"),
    pace: str = typer.Option("moderate", "--pace", help="Trip pace: relaxed/moderate/packed"),
    min_stars: float = typer.Option(3.0, "--stars", help="Minimum hotel stars"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
    natural_language: Optional[str] = typer.Option(
        None, "--nl", help='Natural language input e.g. "Paris for 5 days, $2000 budget, love food"'
    ),
) -> None:
    """
    ✈️  Multi-Agent AI Travel Planner

    Plan complete, budget-aware trips using specialized AI agents.
    """
    _setup_logging(log_level)

    if natural_language:
        # Parse from natural language using PydanticAI
        async def run_nl():
            console.print(f"[dim]Parsing: \"{natural_language}\"...[/dim]")
            try:
                from src.schemas.preferences import parse_preferences_from_text
                prefs = await parse_preferences_from_text(natural_language)
                await _run_planner(prefs.model_dump())
            except Exception as e:
                console.print(f"[red]NL parsing failed: {e}. Switching to interactive mode.[/red]")
                pref_dict = _collect_preferences_interactive()
                await _run_planner(pref_dict)

        asyncio.run(run_nl())
        return

    if destination and budget and days:
        # Build from CLI args
        from datetime import date, timedelta
        today = date.today()
        depart = (today + timedelta(days=14)).strftime("%Y-%m-%d")
        ret = (today + timedelta(days=14 + days)).strftime("%Y-%m-%d")

        pref_dict = {
            "destination": destination,
            "origin": origin or "New York, USA",
            "departure_date": depart,
            "return_date": ret,
            "travelers": travelers,
            "budget_usd": budget,
            "interests": [i.strip() for i in (interests or "food,culture").split(",")],
            "preferred_cabin": cabin,
            "min_hotel_stars": min_stars,
            "activity_pace": pace,
            "hotel_amenities": [],
            "preferred_neighborhoods": [],
            "dietary_restrictions": [],
            "accessibility_needs": "",
            "max_stops": 1,
            "preferred_airlines": [],
            "excluded_activities": [],
            "currency": "USD",
        }
    else:
        # Interactive wizard
        pref_dict = _collect_preferences_interactive()

    asyncio.run(_run_planner(pref_dict))


if __name__ == "__main__":
    app()
