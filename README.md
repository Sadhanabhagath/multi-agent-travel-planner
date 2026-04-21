# ✈️ Multi-Agent Travel Planner

> A production-grade agentic AI system where specialized agents collaborate in real-time to build complete, budget-aware travel itineraries — powered by **LangGraph**, **CrewAI**, **AutoGen**, and **PydanticAI**.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green)](https://langchain-ai.github.io/langgraph/)
[![CrewAI](https://img.shields.io/badge/CrewAI-0.80+-orange)](https://crewai.com)
[![AutoGen](https://img.shields.io/badge/AutoGen-0.4+-purple)](https://microsoft.github.io/autogen/)
[![PydanticAI](https://img.shields.io/badge/PydanticAI-0.0.14+-red)](https://ai.pydantic.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🗺️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR LAYER                        │
│              (LangGraph State Machine)                       │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  Flight  │  │  Hotel   │  │Activity  │  │ Budget   │  │
│  │  Agent   │  │  Agent   │  │  Agent   │  │  Agent   │  │
│  │(CrewAI)  │  │(AutoGen) │  │(CrewAI)  │  │(Pydantic)│  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│        │            │              │              │         │
│        └────────────┴──────────────┴──────────────┘        │
│                           │                                  │
│                  ┌────────▼────────┐                        │
│                  │  Itinerary      │                        │
│                  │  Synthesizer    │                        │
│                  │  (PydanticAI)   │                        │
│                  └─────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Key Features

| Feature | Framework | Description |
|---------|-----------|-------------|
| **Stateful Workflow** | LangGraph | Graph-based orchestration with human-in-the-loop checkpoints |
| **Multi-Agent Crews** | CrewAI | Flight & Activity agents with distinct roles and goals |
| **Conversational Refinement** | AutoGen | Hotel agent negotiates preferences via back-and-forth dialogue |
| **Type-Safe Validation** | PydanticAI | Budget tracking and itinerary schema validation |
| **Real-time Search** | Tools | Amadeus, Booking.com, Google Places API integrations |
| **Budget Awareness** | Cross-agent | Agents share budget state and optimize collectively |
| **Human-in-the-loop** | LangGraph | Pause for user confirmation at key decision points |

---

## 📁 Project Structure

```
multi-agent-travel-planner/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── flight_agent.py        # CrewAI flight search & booking agent
│   │   ├── hotel_agent.py         # AutoGen conversational hotel agent
│   │   ├── activity_agent.py      # CrewAI activities & experiences agent
│   │   ├── budget_agent.py        # PydanticAI budget tracking agent
│   │   └── orchestrator.py        # LangGraph master orchestrator
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── flight_tools.py        # Amadeus API tools
│   │   ├── hotel_tools.py         # Booking.com / Hotels API tools
│   │   ├── activity_tools.py      # Google Places / Viator tools
│   │   └── currency_tools.py      # Exchange rate tools
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── travel_state.py        # LangGraph state schema
│   │   ├── itinerary.py           # Pydantic itinerary models
│   │   └── preferences.py         # User preference models
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py            # Environment & API config
│   │   └── prompts.py             # Agent system prompts
│   └── utils/
│       ├── __init__.py
│       ├── budget_calculator.py
│       └── itinerary_formatter.py
├── tests/
│   ├── test_agents.py
│   ├── test_tools.py
│   └── test_schemas.py
├── docs/
│   ├── architecture.md
│   ├── setup.md
│   └── api_reference.md
├── examples/
│   ├── paris_weekend.py
│   └── budget_asia_trip.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── pyproject.toml
├── requirements.txt
├── .env.example
├── docker-compose.yml
├── Dockerfile
└── main.py
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11+
- API Keys: OpenAI (or Anthropic), Amadeus, optionally Google Places

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/multi-agent-travel-planner.git
cd multi-agent-travel-planner

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run

```bash
# Interactive CLI
python main.py

# With specific trip
python main.py --destination "Tokyo" --budget 3000 --days 7 --travelers 2

# Run example
python examples/paris_weekend.py
```

### 4. Docker

```bash
docker-compose up --build
```

---

## 🧠 Framework Deep Dive

### LangGraph — Stateful Orchestration
Controls the master workflow as a directed graph. Manages state transitions between agents, handles retries, and provides human-in-the-loop checkpoints where users can approve or modify plans.

### CrewAI — Role-Based Agent Teams  
The Flight Agent and Activity Agent are defined as CrewAI "crews" with specific roles, backstories, and goals. They can delegate sub-tasks and collaborate through a manager agent.

### AutoGen — Conversational Hotel Search
The Hotel Agent uses AutoGen's multi-agent conversation pattern: a `UserProxyAgent` and `AssistantAgent` iterate through options, refining preferences until the best match is found.

### PydanticAI — Type-Safe Budget & Validation
All data flowing between agents is validated by Pydantic models. The Budget Agent uses PydanticAI's dependency injection to enforce spending constraints throughout the pipeline.

---

## 📊 Agent Workflow

```
User Input → [Validate Preferences] → [Parallel Search]
                                           ├── Flight Agent (CrewAI)
                                           ├── Hotel Agent (AutoGen)  
                                           └── Activity Agent (CrewAI)
                                      ↓
                               [Budget Check] (PydanticAI)
                                      ↓
                          [Human Approval Checkpoint]
                                      ↓
                            [Itinerary Synthesis]
                                      ↓
                              Final Itinerary Output
```

---

## 🔧 Configuration

Key settings in `.env`:

```env
# LLM Provider
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...   # Alternative

# Travel APIs
AMADEUS_CLIENT_ID=...
AMADEUS_CLIENT_SECRET=...
GOOGLE_PLACES_API_KEY=...

# Agent Settings
MAX_SEARCH_RESULTS=5
BUDGET_BUFFER_PERCENT=10
ENABLE_HUMAN_IN_LOOP=true
```

---

## 🤝 Contributing

1. Fork the repo
2. Create your branch: `git checkout -b feature/amazing-agent`
3. Commit: `git commit -m 'Add amazing agent'`
4. Push: `git push origin feature/amazing-agent`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
