"""BFCL v3 integration for SwayBot.

Exposes SwayBot's LLMBrain behind an OpenAI-compatible API endpoint
so that BFCL's ``bfcl generate`` can call it as a remote model.
BFCL handles its own tool execution loop internally; the server only
needs to translate messages and return function-calling responses.

Usage::

    # Start the SwayBot server
    python -m benchmarks.bfcl --serve --port 8000

    # Then in another shell (with bfcl-eval installed):
    export REMOTE_OPENAI_BASE_URL=http://localhost:8000/v1
    export REMOTE_OPENAI_API_KEY=swaybot
    bfcl generate --model swaybot --test-category multi_turn_base
    bfcl evaluate
"""
