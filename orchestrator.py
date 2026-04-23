# -*- coding: utf-8 -*-
"""Orchestrator agent that drives the multi-agent prompt improvement pipeline.

The orchestrator is itself a ``ReActAgent`` equipped with a ``Toolkit``
whose tools are the four specialist agents (Refiner, Critic, Finalizer,
Scorer) plus a ``detect_prompt_type`` utility.  It decides — via tool
calls — which agent to invoke and in what order.
"""

import asyncio
from typing import Any

from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.tool import Toolkit

from config import get_model_and_formatter
from tools import (
    refine_prompt,
    critique_prompt,
    finalize_prompt,
    score_prompt,
    detect_prompt_type,
)

# ---------------------------------------------------------------------------
# Orchestrator system prompts
# ---------------------------------------------------------------------------

_ADVANCED_SYS_PROMPT = """\
You are the **Orchestrator Agent** for a Prompt Engineering Assistant.

You have access to the following tools:
- ``detect_prompt_type``  – classify the prompt category
- ``refine_prompt``       – improve the prompt's structure and specificity
- ``critique_prompt``     – find weaknesses and missing details
- ``finalize_prompt``     – merge refinement + critique into the best prompt
- ``score_prompt``        – rate the final prompt on a 1-10 scale

**Workflow (follow this order exactly):**
1. Call ``detect_prompt_type`` with the user's original prompt.
2. Call ``refine_prompt`` with the user's original prompt.
3. Call ``critique_prompt`` with the refined prompt from step 2.
4. Call ``finalize_prompt`` with the original prompt, refined prompt, and \
   critique.
5. Call ``score_prompt`` with the final prompt from step 4.

After ALL five tool calls have completed, produce a final summary in **exactly** \
this format (use the markers precisely):

[PROMPT_TYPE]: <detected type>
[REFINED]: <refined prompt from step 2>
[CRITIQUE]: <critique from step 3>
[FINAL]: <final prompt from step 4>
[SCORE]: <score from step 5>
"""

_BASIC_SYS_PROMPT = """\
You are the **Orchestrator Agent** for a Prompt Engineering Assistant \
(basic mode).

You have access to the following tools:
- ``detect_prompt_type``  – classify the prompt category
- ``refine_prompt``       – improve the prompt's structure and specificity
- ``score_prompt``        – rate the prompt on a 1-10 scale

**Workflow (follow this order exactly):**
1. Call ``detect_prompt_type`` with the user's original prompt.
2. Call ``refine_prompt`` with the user's original prompt.
3. Call ``score_prompt`` with the refined prompt from step 2.

After ALL three tool calls have completed, produce a final summary in \
**exactly** this format (use the markers precisely):

[PROMPT_TYPE]: <detected type>
[REFINED]: <refined prompt from step 2>
[CRITIQUE]: N/A (basic mode)
[FINAL]: <refined prompt — same as [REFINED] in basic mode>
[SCORE]: <score from step 3>
"""


def _build_toolkit() -> Toolkit:
    """Build the toolkit with all agent-tool functions registered.

    Returns:
        `Toolkit`:
            A ``Toolkit`` with all tool functions registered.
    """
    toolkit = Toolkit()
    toolkit.register_tool_function(detect_prompt_type)
    toolkit.register_tool_function(refine_prompt)
    toolkit.register_tool_function(critique_prompt)
    toolkit.register_tool_function(finalize_prompt)
    toolkit.register_tool_function(score_prompt)
    return toolkit


def _create_orchestrator(mode: str = "advanced") -> ReActAgent:
    """Create the orchestrator ``ReActAgent``.

    Args:
        mode (`str`, optional):
            ``"advanced"`` (default) or ``"basic"``.

    Returns:
        `ReActAgent`:
            The orchestrator agent with attached toolkit.
    """
    model, formatter = get_model_and_formatter()
    sys_prompt = (
        _ADVANCED_SYS_PROMPT if mode == "advanced" else _BASIC_SYS_PROMPT
    )

    toolkit = _build_toolkit()

    return ReActAgent(
        name="Orchestrator",
        sys_prompt=sys_prompt,
        model=model,
        formatter=formatter,
        toolkit=toolkit,
        memory=InMemoryMemory(),
        max_iters=15,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _run_async(coro: Any) -> Any:
    """Run a coroutine from sync context, handling running event-loops.

    Args:
        coro: The coroutine to run.

    Returns:
        The coroutine's return value.
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


def _parse_section(text: str, marker: str) -> str:
    """Extract the content following ``[MARKER]:`` from the orchestrator output.

    Args:
        text (`str`):
            The full orchestrator response text.
        marker (`str`):
            The section marker (e.g. ``"REFINED"``).

    Returns:
        `str`:
            The extracted section content, or a fallback string.
    """
    tag = f"[{marker}]:"
    if tag not in text:
        return f"(could not extract {marker})"
    start = text.index(tag) + len(tag)
    # Find the next section marker or end of string
    next_markers = ["[PROMPT_TYPE]:", "[REFINED]:", "[CRITIQUE]:", "[FINAL]:", "[SCORE]:"]
    end = len(text)
    for nm in next_markers:
        if nm == tag:
            continue
        idx = text.find(nm, start)
        if idx != -1 and idx < end:
            end = idx
    return text[start:end].strip()


def run_pipeline(user_prompt: str, mode: str = "advanced") -> dict:
    """Run the full prompt-engineering pipeline.

    This is the main entry-point called by the FastAPI endpoint.

    Args:
        user_prompt (`str`):
            The raw user prompt.
        mode (`str`, optional):
            ``"advanced"`` or ``"basic"``.

    Returns:
        `dict`:
            A dictionary with keys: ``original_prompt``, ``refined_prompt``,
            ``critique``, ``final_prompt``, ``score``, ``prompt_type``,
            ``mode``.
    """
    orchestrator = _create_orchestrator(mode)
    msg = Msg(
        name="user",
        content=user_prompt,
        role="user",
    )

    result_msg = _run_async(orchestrator(msg))
    text = result_msg.get_text_content() or ""

    refined = _parse_section(text, "REFINED")
    critique = _parse_section(text, "CRITIQUE")
    final = _parse_section(text, "FINAL")
    score = _parse_section(text, "SCORE")
    prompt_type = _parse_section(text, "PROMPT_TYPE")

    return {
        "original_prompt": user_prompt,
        "refined_prompt": refined,
        "critique": critique,
        "final_prompt": final,
        "score": score,
        "prompt_type": prompt_type,
        "mode": mode,
    }

