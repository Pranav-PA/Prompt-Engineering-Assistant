# -*- coding: utf-8 -*-
"""FastAPI application entry-point for the Prompt Engineering Assistant.

Start with::

    uvicorn main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from config import init_agentscope
from orchestrator import run_pipeline
from schemas import PromptRequest, PromptResponse


# ---------------------------------------------------------------------------
# Lifespan — initialise AgentScope runtime once on startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialise AgentScope runtime before the app starts serving."""
    init_agentscope()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Prompt Engineering Assistant",
    description=(
        "A multi-agent web application that improves vague development "
        "prompts into structured, high-quality prompts using AgentScope."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a minimal interactive HTML UI.

    Returns:
        `HTMLResponse`:
            A self-contained HTML page with a form and result display.
    """
    return HTMLResponse(content=_HTML_PAGE)


@app.get("/health")
async def health():
    """Health-check endpoint.

    Returns:
        `dict`:
            ``{"status": "ok"}``.
    """
    return {"status": "ok"}


@app.post("/generate", response_model=PromptResponse)
async def generate(request: PromptRequest):
    """Run the multi-agent prompt improvement pipeline.

    Args:
        request (`PromptRequest`):
            The request body containing the raw prompt and mode.

    Returns:
        `PromptResponse`:
            The full pipeline output including refined prompt, critique,
            final prompt, score, and detected prompt type.
    """
    try:
        result = run_pipeline(
            user_prompt=request.prompt,
            mode=request.mode,
        )
        return PromptResponse(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Embedded HTML UI
# ---------------------------------------------------------------------------

_HTML_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Prompt Engineering Assistant</title>
<style>
  :root {
    --bg: #0f172a; --surface: #1e293b; --border: #334155;
    --text: #e2e8f0; --muted: #94a3b8; --accent: #38bdf8;
    --accent-hover: #7dd3fc; --green: #4ade80; --red: #f87171;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--text);
    min-height: 100vh; display: flex; flex-direction: column;
    align-items: center; padding: 2rem 1rem;
  }
  h1 { font-size: 1.8rem; margin-bottom: .25rem; color: var(--accent); }
  .subtitle { color: var(--muted); margin-bottom: 1.5rem; font-size: .95rem; }
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.5rem; width: 100%;
    max-width: 760px; margin-bottom: 1.25rem;
  }
  label { font-weight: 600; display: block; margin-bottom: .4rem; }
  textarea {
    width: 100%; min-height: 120px; background: var(--bg);
    border: 1px solid var(--border); border-radius: 8px;
    color: var(--text); padding: .75rem; font-size: .95rem;
    resize: vertical;
  }
  textarea:focus { outline: none; border-color: var(--accent); }
  .row { display: flex; gap: .75rem; align-items: center; margin-top: 1rem; }
  select {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 8px; color: var(--text); padding: .5rem .75rem;
    font-size: .9rem;
  }
  button {
    background: var(--accent); color: #0f172a; border: none;
    border-radius: 8px; padding: .6rem 1.6rem; font-size: .95rem;
    font-weight: 600; cursor: pointer; transition: background .2s;
  }
  button:hover { background: var(--accent-hover); }
  button:disabled { opacity: .5; cursor: not-allowed; }
  .spinner {
    display: none; width: 20px; height: 20px;
    border: 3px solid var(--border); border-top-color: var(--accent);
    border-radius: 50%; animation: spin .7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .result-section { margin-top: .75rem; }
  .result-section h3 {
    font-size: .85rem; text-transform: uppercase; letter-spacing: .05em;
    color: var(--accent); margin-bottom: .35rem;
  }
  .result-section pre {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 8px; padding: .75rem; white-space: pre-wrap;
    word-wrap: break-word; font-size: .88rem; line-height: 1.5;
  }
  .badge {
    display: inline-block; padding: .15rem .6rem; border-radius: 999px;
    font-size: .78rem; font-weight: 600; background: var(--accent);
    color: #0f172a; margin-right: .4rem;
  }
  #results { display: none; }
  #error { color: var(--red); margin-top: .5rem; display: none; }
</style>
</head>
<body>

<h1>Prompt Engineering Assistant</h1>
<p class="subtitle">Multi-agent pipeline &mdash; powered by AgentScope</p>

<div class="card">
  <label for="prompt">Your development prompt</label>
  <textarea id="prompt" placeholder="e.g. build a login page with React"></textarea>
  <div class="row">
    <select id="mode">
      <option value="advanced">Advanced (full pipeline)</option>
      <option value="basic">Basic (single-pass)</option>
    </select>
    <button id="btn" onclick="generate()">Generate</button>
    <div class="spinner" id="spin"></div>
  </div>
  <p id="error"></p>
</div>

<div class="card" id="results">
  <div class="result-section">
    <h3>Detected Prompt Type</h3>
    <p><span class="badge" id="r-type"></span><span class="badge" id="r-mode"></span></p>
  </div>
  <div class="result-section">
    <h3>Refined Prompt</h3>
    <pre id="r-refined"></pre>
  </div>
  <div class="result-section">
    <h3>Critique</h3>
    <pre id="r-critique"></pre>
  </div>
  <div class="result-section">
    <h3>Final Prompt</h3>
    <pre id="r-final"></pre>
  </div>
  <div class="result-section">
    <h3>Score</h3>
    <pre id="r-score"></pre>
  </div>
</div>

<script>
async function generate() {
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) return;
  const mode = document.getElementById('mode').value;
  const btn = document.getElementById('btn');
  const spin = document.getElementById('spin');
  const err = document.getElementById('error');
  const results = document.getElementById('results');

  btn.disabled = true; spin.style.display = 'inline-block';
  err.style.display = 'none'; results.style.display = 'none';

  try {
    const res = await fetch('/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt, mode}),
    });
    if (!res.ok) {
      const d = await res.json().catch(() => ({}));
      throw new Error(d.detail || res.statusText);
    }
    const data = await res.json();
    document.getElementById('r-type').textContent = data.prompt_type;
    document.getElementById('r-mode').textContent = data.mode;
    document.getElementById('r-refined').textContent = data.refined_prompt;
    document.getElementById('r-critique').textContent = data.critique;
    document.getElementById('r-final').textContent = data.final_prompt;
    document.getElementById('r-score').textContent = data.score;
    results.style.display = 'block';
  } catch(e) {
    err.textContent = 'Error: ' + e.message;
    err.style.display = 'block';
  } finally {
    btn.disabled = false; spin.style.display = 'none';
  }
}
</script>
</body>
</html>
"""

