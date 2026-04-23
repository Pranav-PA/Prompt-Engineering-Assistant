# -*- coding: utf-8 -*-
"""Tool wrappers around each specialist agent.

Every agent is exposed as a plain Python function that returns a
``ToolResponse``.  These functions are registered in the
orchestrator's ``Toolkit`` so that the orchestrator agent can
invoke them dynamically via tool calls.
"""

import asyncio

from agentscope.message import Msg
from agentscope.tool import ToolResponse

from agents import (
    create_refiner_agent,
    create_critic_agent,
    create_finalizer_agent,
    create_scorer_agent,
)


def _run_async(coro):
    """Run an async coroutine from sync context.

    If an event-loop is already running (e.g. inside FastAPI with
    ``uvicorn``), we spin up a background thread with its own loop.
    Otherwise we use ``asyncio.run`` directly.

    Args:
        coro: The coroutine to execute.

    Returns:
        The result of the coroutine.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def refine_prompt(prompt: str) -> ToolResponse:
    """Refine a vague development prompt into a clear, structured one.

    The Refiner Agent rewrites the given prompt to be more specific,
    well-organised, and actionable.

    Args:
        prompt (str): The raw or vague prompt to refine.

    Returns:
        ToolResponse: The refined prompt text.
    """
    agent = create_refiner_agent()
    msg = Msg(name="user", content=prompt, role="user")
    result = _run_async(agent(msg))
    text = result.get_text_content() or ""
    return ToolResponse(
        content=[{"type": "text", "text": text}],
    )


def critique_prompt(prompt: str) -> ToolResponse:
    """Critique a development prompt and identify weaknesses.

    The Critic Agent reviews the prompt for clarity, specificity,
    completeness, and actionability, returning structured feedback.

    Args:
        prompt (str): The prompt to critique.

    Returns:
        ToolResponse: Critique feedback as a bullet-point list.
    """
    agent = create_critic_agent()
    msg = Msg(name="user", content=prompt, role="user")
    result = _run_async(agent(msg))
    text = result.get_text_content() or ""
    return ToolResponse(
        content=[{"type": "text", "text": text}],
    )


def finalize_prompt(
    original_prompt: str,
    refined_prompt: str,
    critique: str,
) -> ToolResponse:
    """Produce the final polished prompt by merging refinement and critique.

    The Finalizer Agent takes the original prompt, the refined version,
    and the critic's feedback to produce the best possible final prompt.

    Args:
        original_prompt (str): The original user prompt.
        refined_prompt (str): The refined version from the Refiner Agent.
        critique (str): The critique from the Critic Agent.

    Returns:
        ToolResponse: The final, polished prompt text.
    """
    agent = create_finalizer_agent()
    combined = (
        f"## Original Prompt\n{original_prompt}\n\n"
        f"## Refined Prompt\n{refined_prompt}\n\n"
        f"## Critic Feedback\n{critique}"
    )
    msg = Msg(name="user", content=combined, role="user")
    result = _run_async(agent(msg))
    text = result.get_text_content() or ""
    return ToolResponse(
        content=[{"type": "text", "text": text}],
    )


def score_prompt(prompt: str) -> ToolResponse:
    """Score a development prompt on a 1-10 quality scale.

    The Scorer Agent evaluates clarity, specificity, completeness,
    actionability, and structure.

    Args:
        prompt (str): The prompt to evaluate.

    Returns:
        ToolResponse: The score breakdown and justification.
    """
    agent = create_scorer_agent()
    msg = Msg(name="user", content=prompt, role="user")
    result = _run_async(agent(msg))
    text = result.get_text_content() or ""
    return ToolResponse(
        content=[{"type": "text", "text": text}],
    )


def detect_prompt_type(prompt: str) -> ToolResponse:
    """Detect the category of a development prompt.

    Classifies the prompt into one of: frontend, backend, fullstack,
    ML/AI, DevOps, data-engineering, mobile, or general.

    Args:
        prompt (str): The prompt to classify.

    Returns:
        ToolResponse: The detected prompt category.
    """
    prompt_lower = prompt.lower()

    keywords = {
        "frontend": [
            "react", "vue", "angular", "css", "html", "ui", "ux",
            "component", "dom", "tailwind", "next.js", "svelte",
            "browser", "responsive",
        ],
        "backend": [
            "api", "server", "database", "sql", "rest", "graphql",
            "microservice", "endpoint", "authentication", "django",
            "flask", "fastapi", "express", "spring",
        ],
        "ML/AI": [
            "model", "train", "dataset", "neural", "transformer",
            "pytorch", "tensorflow", "fine-tune", "embedding",
            "llm", "classification", "regression", "nlp",
            "machine learning", "deep learning",
        ],
        "DevOps": [
            "docker", "kubernetes", "ci/cd", "pipeline", "deploy",
            "terraform", "ansible", "monitoring", "helm", "jenkins",
            "github actions",
        ],
        "data-engineering": [
            "etl", "data pipeline", "spark", "airflow", "kafka",
            "warehouse", "lakehouse", "dbt", "snowflake",
        ],
        "mobile": [
            "ios", "android", "react native", "flutter", "swift",
            "kotlin", "mobile app",
        ],
    }

    scores = {category: 0 for category in keywords}
    for category, words in keywords.items():
        for word in words:
            if word in prompt_lower:
                scores[category] += 1

    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    detected = best if scores[best] > 0 else "general"

    return ToolResponse(
        content=[{"type": "text", "text": detected}],
    )

