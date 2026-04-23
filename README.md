# Prompt Engineering Assistant

A multi-agent web application that improves vague development prompts into
structured, high-quality prompts using the **AgentScope** framework.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   FastAPI Server                     │
│  POST /generate  ←──  PromptRequest (prompt, mode)   │
│                  ──→  PromptResponse                 │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│              Orchestrator Agent (ReActAgent)          │
│  Equipped with a Toolkit containing 5 tool functions │
│  Decides which tools to call and in what order        │
└──┬──────┬──────┬──────┬──────┬───────────────────────┘
   │      │      │      │      │
   ▼      ▼      ▼      ▼      ▼
┌─────┐┌─────┐┌─────┐┌─────┐┌────────────┐
│Refine││Crit-││Final││Score││  Detect    │
│Agent ││ique ││izer ││Agent││ Prompt Type│
│      ││Agent││Agent││     ││ (keyword)  │
└─────┘└─────┘└─────┘└─────┘└────────────┘
```

### Agents

| Agent       | Role                                                        |
|-------------|-------------------------------------------------------------|
| **Orchestrator** | ReActAgent with tools — drives the entire workflow      |
| **Refiner**      | Rewrites vague prompts into clear, structured ones     |
| **Critic**       | Reviews prompts for missing details and weaknesses     |
| **Finalizer**    | Merges refinement + critique into the best final prompt|
| **Scorer**       | Rates the final prompt on a 1–10 scale                 |

### Tool-based Orchestration

Each specialist agent is wrapped as a **tool function** (see `tools.py`).
These functions are registered in an AgentScope `Toolkit` and attached to
the Orchestrator agent.  The orchestrator uses **ReAct-style reasoning** to
decide which tool to call next — it is NOT hardcoded chaining.

---

## How AgentScope Runtime Is Used

1. **`agentscope.init()`** is called once at server startup (via FastAPI
   lifespan) to initialise logging and the runtime environment.

2. **Models** are created using `agentscope.model.OpenAIChatModel` — all
   LLM interactions flow through AgentScope's model layer.

3. **Formatters** (`agentscope.formatter.OpenAIChatFormatter`) convert
   messages into the format required by the OpenAI API.

4. **Agents** are `agentscope.agent.ReActAgent` instances with per-agent
   system prompts, memory (`InMemoryMemory`), and model references.

5. **Tools** are registered via `agentscope.tool.Toolkit` and return
   `agentscope.tool.ToolResponse` objects.

6. **Messages** use `agentscope.message.Msg` throughout.

There are **zero** direct OpenAI/Gemini API calls anywhere in the code.

---

## Project Structure

```
prompt_engineering_assistant/
├── .env                 # API keys and configuration
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── main.py              # FastAPI app entry-point + embedded HTML UI
├── config.py            # AgentScope runtime init + model factory
├── agents.py            # Specialist agent definitions
├── tools.py             # Tool wrappers around each agent
├── orchestrator.py      # Orchestrator agent + pipeline runner
└── schemas.py           # Pydantic request/response models
```

---

## Setup

### 1. Install dependencies

```bash
cd prompt-engineering-assistant
pip install -r requirements.txt
```

### 2. Configure API key

Edit `.env` and set your OpenAI API key:

```
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL_NAME=gpt-4o
```

### 3. Run the server

```bash
uvicorn main:app --reload
```

The server starts at `http://localhost:8000`.

---

## Usage

### Web UI

Open `http://localhost:8000` in your browser.  You'll see an interactive
form where you can:

- Type your vague development prompt
- Select **Basic** or **Advanced** mode
- Click **Generate** to run the multi-agent pipeline
- View the refined prompt, critique, final prompt, score, and detected type

### REST API

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "build a login page", "mode": "advanced"}'
```

**Response:**

```json
{
  "original_prompt": "build a login page",
  "refined_prompt": "...",
  "critique": "...",
  "final_prompt": "...",
  "score": "Score: 8/10 ...",
  "prompt_type": "frontend",
  "mode": "advanced"
}
```

---

## Modes

| Mode       | Pipeline                                        |
|------------|-------------------------------------------------|
| **Basic**    | Detect → Refine → Score                       |
| **Advanced** | Detect → Refine → Critique → Finalize → Score |

---

## Bonus Features

- **Prompt mode selector**: Basic vs Advanced via the `mode` field
- **Prompt type detection**: Automatically classifies prompts as frontend,
  backend, ML/AI, DevOps, data-engineering, mobile, or general
- **Clean embedded UI**: Dark-themed responsive HTML served at `/`
