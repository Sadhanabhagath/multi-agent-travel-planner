# Architecture Deep Dive

## Overview

The Multi-Agent Travel Planner is built around a **directed acyclic graph** (DAG) of specialized agents, each powered by a different AI framework best suited to its task.

## Framework Responsibilities

### LangGraph — The Nervous System

LangGraph acts as the master orchestrator. It manages:

- **State** — A shared `TravelPlannerState` TypedDict passed between every node
- **Control flow** — Conditional edges route the workflow based on budget state, errors, and user feedback  
- **Checkpointing** — `MemorySaver` persists state so the workflow can pause/resume for human approval
- **Human-in-the-loop** — `interrupt_before=["human_approval"]` pauses the graph at the approval node

```python
graph = StateGraph(TravelPlannerState)
graph.add_node("flight_search", run_flight_agent)
graph.add_node("human_approval", human_approval_node)
graph.compile(checkpointer=MemorySaver(), interrupt_before=["human_approval"])
```

### CrewAI — Role-Based Agent Teams

CrewAI powers the **Flight Agent** and **Activity Agent** because both tasks benefit from multiple agents with distinct roles collaborating sequentially:

- **Flight Crew**: A `FlightSearchAgent` finds raw options; a `FlightValueAnalyst` ranks them
- **Activity Crew**: An `ActivityResearcher` finds options; an `ItineraryCurator` builds the day plan

Each agent has a `role`, `goal`, `backstory`, and a set of `tools`. The `Process.sequential` mode ensures the analyst sees the searcher's output.

### AutoGen — Conversational Refinement

The **Hotel Agent** uses AutoGen because hotel selection benefits from iterative dialogue. An `AssistantAgent` searches and presents options; a `UserProxyAgent` provides feedback (automated in pipeline mode, or proxied from the user). They exchange messages until the best hotel is identified.

This mirrors how a real travel agent conversation works — you don't just get a list, you negotiate.

### PydanticAI — Type-Safe Validation Layer

PydanticAI wraps the **Budget Agent** and **Preference Parser**:

- **Budget Agent**: Uses `deps_type=BudgetDependencies` for dependency injection, ensuring budget state is always validated before analysis
- **Preference Parser**: Converts natural language input into a strictly typed `NormalizedPreferences` object
- **All schemas**: Every data object crossing agent boundaries is a Pydantic `BaseModel`

## Data Flow

```
User Input (text or structured)
    │
    ▼
┌─────────────────────────┐
│  Preference Parser       │  ← PydanticAI
│  NormalizedPreferences   │
└──────────┬──────────────┘
           │
    ┌──────▼──────┐
    │  Initialize  │  ← LangGraph node
    │  State Setup │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │   Flights    │  ← CrewAI (2-agent crew)
    │   Search     │    Tools: Amadeus / mock
    └──────┬──────┘
           │  Updates: flight_options, selected_flight, budget.spent_flights
    ┌──────▼──────┐
    │   Hotels     │  ← AutoGen (conversation loop)
    │   Search     │    Tools: Booking / mock
    └──────┬──────┘
           │  Updates: hotel_options, selected_hotel, budget.spent_hotels
    ┌──────▼──────┐
    │  Activities  │  ← CrewAI (2-agent crew)
    │   Search     │    Tools: Google Places / mock
    └──────┬──────┘
           │  Updates: activity_options, selected_activities, budget.spent_activities
    ┌──────▼──────┐
    │   Budget     │  ← PydanticAI + rule-based
    │   Review     │    Validates all spending, generates alerts
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │   Human      │  ← LangGraph interrupt
    │   Approval   │    User can approve, reject, or modify
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Synthesis   │  ← GPT-4o / Claude synthesis
    │  Itinerary   │    Produces markdown itinerary
    └─────────────┘
```

## State Management

The `TravelPlannerState` TypedDict is the single source of truth:

```python
class TravelPlannerState(TypedDict):
    preferences: UserPreferences      # Immutable input
    budget: BudgetBreakdown           # Mutated by each agent
    flight_options: list[FlightOption]
    selected_flight: FlightOption | None
    hotel_options: list[HotelOption]
    selected_hotel: HotelOption | None
    activity_options: list[ActivityOption]
    selected_activities: list[ActivityOption]
    completed_steps: Annotated[list[str], operator.add]  # Append-only
    errors: Annotated[list[str], operator.add]           # Append-only
    messages: Annotated[list[Any], add_messages]         # For AutoGen
    final_itinerary: dict | None
```

`Annotated` fields with `operator.add` are merged (not overwritten) across graph nodes, enabling safe parallel state updates.

## Error Handling & Resilience

Every agent follows a **try → fallback** pattern:

1. Attempt real API call / LLM crew execution  
2. On failure: fall back to mock data, log the error to `state["errors"]`
3. Workflow continues with flagged data rather than crashing
4. `error_handler` node surfaces warnings to user in final output

## Extending the System

To add a new agent (e.g., a Visa Requirements Agent):

1. Create `src/agents/visa_agent.py` with a `run_visa_agent(state)` async function
2. Add tools in `src/tools/visa_tools.py`
3. Register as a LangGraph node: `graph.add_node("visa_check", run_visa_agent)`
4. Add edges to connect it in the workflow
5. Extend `TravelPlannerState` with relevant fields
