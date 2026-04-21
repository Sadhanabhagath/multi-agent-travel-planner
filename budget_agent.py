"""
agents/budget_agent.py
----------------------
PydanticAI-powered Budget Agent with strict type validation.
Tracks spending across all agents and enforces budget constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from loguru import logger

from ..config.prompts import BUDGET_AGENT_SYSTEM_PROMPT
from ..config.settings import get_settings
from ..schemas.travel_state import BudgetBreakdown, TravelPlannerState

settings = get_settings()


# ── Pydantic Output Schemas ───────────────────────────────────────────────────

class BudgetAlert(BaseModel):
    """Alert generated when budget thresholds are exceeded."""
    severity: str = Field(description="low | medium | high | critical")
    category: str = Field(description="flights | hotels | activities | total")
    message: str
    recommended_action: str
    potential_savings_usd: float = 0.0


class BudgetAnalysis(BaseModel):
    """Full budget analysis output from PydanticAI agent."""
    total_budget_usd: float
    total_spent_usd: float
    remaining_usd: float
    utilization_percent: float

    breakdown: dict[str, float] = Field(
        description="Spending by category: flights, hotels, activities, misc"
    )

    is_within_budget: bool
    alerts: list[BudgetAlert] = Field(default_factory=list)

    optimization_suggestions: list[str] = Field(
        description="Specific suggestions to reduce costs if over budget"
    )

    per_person_cost: float
    per_day_cost: float

    budget_grade: str = Field(
        description="A-F grade: A=excellent value, F=significantly over budget"
    )


class OptimizationResult(BaseModel):
    """Result of budget optimization attempt."""
    original_total: float
    optimized_total: float
    savings_usd: float
    changes_made: list[str]
    trade_offs: list[str]


# ── PydanticAI Agent Dependencies ─────────────────────────────────────────────

@dataclass
class BudgetDependencies:
    """Injected dependencies for the budget agent."""
    budget: BudgetBreakdown
    travelers: int
    trip_days: int
    destination: str


# ── PydanticAI Agent ──────────────────────────────────────────────────────────

def _create_budget_agent() -> Agent:
    """Create the PydanticAI budget analysis agent."""
    model = OpenAIModel(
        settings.default_llm_model,
        api_key=settings.openai_api_key,
    ) if settings.openai_api_key else None

    agent: Agent[BudgetDependencies, BudgetAnalysis] = Agent(
        model=model,
        deps_type=BudgetDependencies,
        result_type=BudgetAnalysis,
        system_prompt=BUDGET_AGENT_SYSTEM_PROMPT,
    )

    @agent.system_prompt
    def dynamic_system_prompt(ctx: RunContext[BudgetDependencies]) -> str:
        budget = ctx.deps.budget
        return f"""
        Current budget state:
        - Total budget: ${budget.total_budget_usd:.2f} USD
        - Spent on flights: ${budget.spent_flights:.2f}
        - Spent on hotels: ${budget.spent_hotels:.2f}
        - Spent on activities: ${budget.spent_activities:.2f}
        - Total spent: ${budget.total_spent:.2f}
        - Remaining: ${budget.remaining:.2f}
        - Utilization: {budget.utilization_percent:.1f}%
        - Buffer reserve: {budget.buffer_percent}%
        
        Travelers: {ctx.deps.travelers}
        Trip days: {ctx.deps.trip_days}
        Destination: {ctx.deps.destination}
        
        Analyze this budget, identify any issues, and provide optimization suggestions.
        """

    return agent


# ── Budget Calculation Utilities ──────────────────────────────────────────────

def calculate_budget_grade(utilization: float, is_within: bool) -> str:
    """Assign a letter grade based on budget utilization."""
    if not is_within:
        return "F" if utilization > 120 else "D"
    if utilization <= 75:
        return "A"
    elif utilization <= 85:
        return "B"
    elif utilization <= 95:
        return "C"
    elif utilization <= 100:
        return "D"
    return "F"


def check_budget_alerts(budget: BudgetBreakdown) -> list[BudgetAlert]:
    """Generate budget alerts based on current spending."""
    alerts = []

    # Overall budget alerts
    if budget.utilization_percent > 100:
        alerts.append(BudgetAlert(
            severity="critical",
            category="total",
            message=f"Over budget by ${budget.total_spent - budget.total_budget_usd:.2f}!",
            recommended_action="Review and reduce hotel or activity costs.",
            potential_savings_usd=budget.total_spent - budget.effective_budget,
        ))
    elif budget.utilization_percent > 90:
        alerts.append(BudgetAlert(
            severity="high",
            category="total",
            message=f"Budget {budget.utilization_percent:.0f}% utilized — very little buffer left.",
            recommended_action="Avoid adding more activities or upgrades.",
            potential_savings_usd=0,
        ))

    # Category-specific alerts
    if budget.total_budget_usd > 0:
        flight_pct = (budget.spent_flights / budget.total_budget_usd) * 100
        hotel_pct = (budget.spent_hotels / budget.total_budget_usd) * 100
        activity_pct = (budget.spent_activities / budget.total_budget_usd) * 100

        if flight_pct > 55:
            alerts.append(BudgetAlert(
                severity="medium",
                category="flights",
                message=f"Flights using {flight_pct:.0f}% of total budget.",
                recommended_action="Consider alternative dates or nearby airports.",
                potential_savings_usd=budget.spent_flights * 0.2,
            ))

        if hotel_pct > 50:
            alerts.append(BudgetAlert(
                severity="medium",
                category="hotels",
                message=f"Accommodation using {hotel_pct:.0f}% of total budget.",
                recommended_action="Look for 3-star options or apartments.",
                potential_savings_usd=budget.spent_hotels * 0.25,
            ))

    return alerts


# ── Main Execution Function ───────────────────────────────────────────────────

async def run_budget_agent(state: TravelPlannerState) -> TravelPlannerState:
    """
    Execute the Budget Agent and update state.
    Called as a LangGraph node.
    """
    logger.info("💰 Budget Agent analyzing spending...")
    prefs = state["preferences"]
    budget = state["budget"]

    # Generate alerts using rule-based logic (always works)
    alerts = check_budget_alerts(budget)

    # Attempt full PydanticAI analysis
    try:
        budget_agent = _create_budget_agent()
        deps = BudgetDependencies(
            budget=budget,
            travelers=prefs.travelers,
            trip_days=prefs.trip_days,
            destination=prefs.destination,
        )

        analysis_prompt = f"""
        Analyze the current travel budget for a {prefs.trip_days}-day trip to {prefs.destination}
        for {prefs.travelers} traveler(s).
        
        Budget breakdown:
        - Total: ${budget.total_budget_usd:.2f}
        - Flights: ${budget.spent_flights:.2f}
        - Hotels: ${budget.spent_hotels:.2f}
        - Activities: ${budget.spent_activities:.2f}
        - Total spent: ${budget.total_spent:.2f}
        - Remaining: ${budget.remaining:.2f}
        
        Provide a thorough analysis with specific optimization suggestions.
        """

        result = await budget_agent.run(analysis_prompt, deps=deps)
        analysis = result.data

        logger.info(
            f"✅ Budget Analysis: {analysis.utilization_percent:.1f}% utilized, "
            f"Grade: {analysis.budget_grade}, "
            f"Alerts: {len(analysis.alerts)}"
        )

        # Log budget summary
        _log_budget_summary(budget, analysis)

        return {
            **state,
            "budget": budget,
            "current_step": "human_approval" if settings.enable_human_in_loop else "synthesis",
            "completed_steps": ["budget_review"],
            "requires_human_approval": settings.enable_human_in_loop,
        }

    except Exception as e:
        logger.warning(f"PydanticAI budget agent error: {e}. Using rule-based analysis.")

        # Rule-based fallback
        grade = calculate_budget_grade(budget.utilization_percent, not budget.is_over_budget)
        logger.info(
            f"✅ Budget (rule-based): {budget.utilization_percent:.1f}% utilized, "
            f"Grade: {grade}, Alerts: {len(alerts)}"
        )

        if alerts:
            logger.warning(f"⚠️  Budget alerts: {[a.message for a in alerts]}")

        return {
            **state,
            "budget": budget,
            "current_step": "human_approval" if settings.enable_human_in_loop else "synthesis",
            "completed_steps": ["budget_review"],
            "requires_human_approval": settings.enable_human_in_loop,
        }


def _log_budget_summary(budget: BudgetBreakdown, analysis: BudgetAnalysis) -> None:
    """Pretty-print budget summary to console."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="💰 Budget Summary", show_header=True, header_style="bold blue")
    table.add_column("Category", style="cyan")
    table.add_column("Amount (USD)", justify="right")
    table.add_column("% of Budget", justify="right")

    for category, amount in analysis.breakdown.items():
        pct = (amount / analysis.total_budget_usd * 100) if analysis.total_budget_usd > 0 else 0
        table.add_row(category.title(), f"${amount:.2f}", f"{pct:.1f}%")

    table.add_row("─────────", "──────────", "────────", style="dim")
    table.add_row(
        "TOTAL", f"${analysis.total_spent_usd:.2f}",
        f"{analysis.utilization_percent:.1f}%",
        style="bold"
    )
    table.add_row(
        "Remaining", f"${analysis.remaining_usd:.2f}", "",
        style="green" if analysis.is_within_budget else "red"
    )

    console.print(table)
    console.print(f"\nBudget Grade: [bold]{analysis.budget_grade}[/bold]")
