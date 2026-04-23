# -*- coding: utf-8 -*-
"""Pydantic models for request/response validation."""

from typing import Literal

from pydantic import BaseModel, Field


class PromptRequest(BaseModel):
    """Request model for the /generate endpoint.

    Args:
        prompt (`str`):
            The raw user prompt to be improved.
        mode (`Literal["basic", "advanced"]`, optional):
            The refinement mode. ``basic`` runs a single pass, ``advanced``
            runs the full multi-agent pipeline with critique and iteration.
    """

    prompt: str = Field(
        ...,
        min_length=1,
        description="The raw user prompt to be improved.",
    )
    mode: Literal["basic", "advanced"] = Field(
        default="advanced",
        description=(
            "Refinement mode: 'basic' for a single-pass refinement, "
            "'advanced' for full multi-agent pipeline."
        ),
    )


class PromptResponse(BaseModel):
    """Response model returned by the /generate endpoint.

    Args:
        original_prompt (`str`):
            The original user prompt.
        refined_prompt (`str`):
            The prompt after the Refiner Agent improved it.
        critique (`str`):
            Feedback from the Critic Agent on what could be improved.
        final_prompt (`str`):
            The final, polished prompt produced by the Finalizer Agent.
        score (`str`):
            A quality score and brief justification from the Scorer Agent.
        prompt_type (`str`):
            Detected category of the prompt (e.g. frontend, backend, ML).
        mode (`str`):
            The mode that was used for generation.
    """

    original_prompt: str
    refined_prompt: str
    critique: str
    final_prompt: str
    score: str
    prompt_type: str
    mode: str

