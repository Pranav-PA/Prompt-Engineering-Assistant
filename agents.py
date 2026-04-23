# -*- coding: utf-8 -*-
"""Specialised agents for the Prompt Engineering Assistant.

Each agent wraps an ``agentscope.agent.ReActAgent`` with a dedicated
system prompt.  All LLM interactions flow through AgentScope's model
layer — there are **no** direct OpenAI/Gemini API calls.
"""

from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit

from config import get_model_and_formatter

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

REFINER_SYS_PROMPT = """\
You are the **Refiner Agent**, an expert at taking vague, incomplete, or \
poorly structured development prompts and transforming them into clear, \
well-structured prompts.

When you receive a prompt to refine:
1. Identify the core intent behind the user's request.
2. Add specificity: mention relevant technologies, constraints, \
   input/output formats, edge-cases, and acceptance criteria.
3. Organise the prompt with a clear structure (context → task → constraints \
   → expected output).
4. Keep the refined prompt concise but comprehensive.

Return ONLY the refined prompt text, nothing else.
"""

CRITIC_SYS_PROMPT = """\
You are the **Critic Agent**, a meticulous reviewer of development prompts.

When you receive a prompt:
1. Evaluate clarity, specificity, completeness, and structure.
2. Point out any missing details: technologies, edge-cases, error handling, \
   performance requirements, security considerations, testing expectations.
3. Check if the prompt is actionable — could a competent developer implement \
   it without asking follow-up questions?
4. Provide concrete, constructive feedback on how to improve the prompt.

Return your critique as a structured bullet-point list.
"""

FINALIZER_SYS_PROMPT = """\
You are the **Finalizer Agent**. You produce the final, polished prompt \
ready for use with an AI coding assistant.

You will receive:
- An original prompt
- A refined version
- Critic feedback

Your job:
1. Merge the best aspects of the refined prompt with the critic's feedback.
2. Ensure the final prompt is self-contained, specific, and actionable.
3. Preserve the user's original intent while maximising quality.
4. Format the prompt with clear sections (Context, Task, Constraints, \
   Expected Output) where appropriate.

Return ONLY the final prompt text.
"""

SCORER_SYS_PROMPT = """\
You are the **Scorer Agent**. You evaluate the quality of a development \
prompt on a 1–10 scale.

Scoring criteria:
- **Clarity** (0-2): Is it unambiguous?
- **Specificity** (0-2): Does it mention technologies, formats, constraints?
- **Completeness** (0-2): Does it cover edge-cases, error handling, testing?
- **Actionability** (0-2): Could a developer implement it without follow-ups?
- **Structure** (0-2): Is it well-organised and easy to follow?

Return your response in this EXACT format:
Score: <total>/10
Breakdown:
- Clarity: <n>/2
- Specificity: <n>/2
- Completeness: <n>/2
- Actionability: <n>/2
- Structure: <n>/2
Brief justification: <1-2 sentences>
"""


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_refiner_agent() -> ReActAgent:
    """Create and return the Refiner Agent.

    Returns:
        `ReActAgent`:
            An agent configured to refine vague prompts.
    """
    model, formatter = get_model_and_formatter()
    return ReActAgent(
        name="Refiner",
        sys_prompt=REFINER_SYS_PROMPT,
        model=model,
        formatter=formatter,
        memory=InMemoryMemory(),
        max_iters=1,
    )


def create_critic_agent() -> ReActAgent:
    """Create and return the Critic Agent.

    Returns:
        `ReActAgent`:
            An agent configured to critique prompts.
    """
    model, formatter = get_model_and_formatter()
    return ReActAgent(
        name="Critic",
        sys_prompt=CRITIC_SYS_PROMPT,
        model=model,
        formatter=formatter,
        memory=InMemoryMemory(),
        max_iters=1,
    )


def create_finalizer_agent() -> ReActAgent:
    """Create and return the Finalizer Agent.

    Returns:
        `ReActAgent`:
            An agent configured to produce the final prompt.
    """
    model, formatter = get_model_and_formatter()
    return ReActAgent(
        name="Finalizer",
        sys_prompt=FINALIZER_SYS_PROMPT,
        model=model,
        formatter=formatter,
        memory=InMemoryMemory(),
        max_iters=1,
    )


def create_scorer_agent() -> ReActAgent:
    """Create and return the Scorer Agent.

    Returns:
        `ReActAgent`:
            An agent configured to score prompt quality.
    """
    model, formatter = get_model_and_formatter()
    return ReActAgent(
        name="Scorer",
        sys_prompt=SCORER_SYS_PROMPT,
        model=model,
        formatter=formatter,
        memory=InMemoryMemory(),
        max_iters=1,
    )

