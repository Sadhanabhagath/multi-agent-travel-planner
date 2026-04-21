# Setup Guide

## Prerequisites

- Python 3.11 or 3.12
- `git`
- An OpenAI API key (required) or Anthropic API key (alternative)
- Optional: Amadeus API key (for real flight data)

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/multi-agent-travel-planner.git
cd multi-agent-travel-planner
```

---

## Step 2: Create a Virtual Environment

```bash
python -m venv venv

# Activate
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate.bat       # Windows CMD
venv\Scripts\Activate.ps1       # Windows PowerShell
```

---

## Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step 4: Configure API Keys

```bash
cp .env.example .env
```

Open `.env` and add your keys:

```env
# Required (at least one)
OPENAI_API_KEY=sk-your-openai-key

# Optional (enables real flight search)
AMADEUS_CLIENT_ID=your-id
AMADEUS_CLIENT_SECRET=your-secret
AMADEUS_ENVIRONMENT=test   # Use 'test' for free tier
```

### Getting API Keys

| Service | URL | Free Tier |
|---------|-----|-----------|
| OpenAI | https://platform.openai.com/api-keys | $5 credit |
| Anthropic | https://console.anthropic.com | Limited free |
| Amadeus | https://developers.amadeus.com | Yes (test env) |
| Google Places | https://console.cloud.google.com | $200/month credit |

> **Note:** Without API keys, the system runs entirely on realistic mock data. All features work — you just won't get real flight/hotel prices.

---

## Step 5: Run the Planner

### Interactive Mode
```bash
python main.py
```

### Direct Arguments
```bash
python main.py \
  --destination "Tokyo, Japan" \
  --origin "London, UK" \
  --budget 3500 \
  --days 7 \
  --travelers 2 \
  --interests "food,culture,nature"
```

### Natural Language Mode
```bash
python main.py --nl "I want to spend a week in Bali for $2000, I love beaches and local food"
```

### Run an Example
```bash
python examples/paris_weekend.py
python examples/budget_asia_trip.py
```

---

## Step 6: Run Tests

```bash
pytest tests/ -v
```

---

## Docker Setup

```bash
# Build and run
docker-compose up --build

# Custom trip via Docker
docker run --env-file .env multi-agent-travel-planner \
  --destination "Barcelona" --budget 2000 --days 5
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'crewai'`**  
→ Re-run `pip install -r requirements.txt` with your venv activated.

**`ValueError: No LLM API key configured`**  
→ Check your `.env` file has `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` set.

**AutoGen agents not conversing**  
→ AutoGen requires `pyautogen>=0.4`. Check with `pip show pyautogen`.

**Slow execution**  
→ Switch to `DEFAULT_LLM_MODEL=gpt-4o-mini` in `.env` for faster, cheaper runs.
