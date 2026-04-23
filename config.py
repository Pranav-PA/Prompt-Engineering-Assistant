# -*- coding: utf-8 -*-
"""AgentScope runtime configuration and model factory.

Initialises the AgentScope runtime and provides a helper to build
the ``OpenAIChatModel`` (with matching formatter) that every agent
in this application shares.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def get_model_and_formatter():
    """Create and return an OpenAI chat model and its formatter.

    The model reads ``OPENAI_API_KEY`` and ``OPENAI_MODEL_NAME`` from
    the environment (or ``.env`` file).  A ``stream=False`` model is
    returned so that the FastAPI endpoint can collect the full response
    synchronously.

    Returns:
        `tuple[OpenAIChatModel, OpenAIChatFormatter]`:
            A 2-tuple of (model, formatter).
    """
    from agentscope.model import OpenAIChatModel
    from agentscope.formatter import OpenAIChatFormatter

    api_key = os.getenv("OPENAI_API_KEY", "")
    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it in the .env file or your environment.",
        )

    model = OpenAIChatModel(
        model_name=model_name,
        api_key=api_key,
        stream=False,
    )

    formatter = OpenAIChatFormatter()

    return model, formatter


def init_agentscope() -> None:
    """Initialise the AgentScope runtime.

    This **must** be called once before any agents are created.
    """
    from agentscope import init

    init(
        logging_level="INFO",
    )

