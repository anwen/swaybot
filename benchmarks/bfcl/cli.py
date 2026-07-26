"""CLI entry point for running BFCL v3 evaluation with SwayBot.

Two modes:

1. **Server mode** — start the OpenAI-compatible API server, then run
   ``bfcl generate`` + ``bfcl evaluate`` manually::

       python -m benchmarks.bfcl --serve --port 8000

2. **One-shot mode** — start the server, run BFCL, and shut down::

       python -m benchmarks.bfcl --test-category multi_turn_base --model swaybot

Requires ``bfcl-eval`` (``pip install bfcl-eval``) and ``swaybot[api]``.
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BFCL v3 evaluation with SwayBot",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the SwayBot OpenAI-compatible server (no BFCL run)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--test-category",
        default=None,
        help="BFCL test category (e.g. multi_turn_base, simple). "
             "If set, runs BFCL generate + evaluate after starting the server.",
    )
    parser.add_argument(
        "--model",
        default="swaybot",
        help="Model name passed to BFCL (default: swaybot)",
    )
    parser.add_argument(
        "--bfcl-project-root",
        default=None,
        help="BFCL_PROJECT_ROOT directory. Required for bfcl-eval.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the SwayBot server (default: swaybot)",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="API base URL for the underlying LLM (or set SWAYBOT_API_BASE)",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Underlying LLM model name (or set SWAYBOT_MODEL)",
    )
    args = parser.parse_args()

    # Set environment variables for SwayBot's LLMBrain
    if args.api_base:
        os.environ["SWAYBOT_API_BASE"] = args.api_base
    if args.llm_model:
        os.environ["SWAYBOT_MODEL"] = args.llm_model
    if args.api_key:
        os.environ["SWAYBOT_API_KEY"] = args.api_key

    if args.serve and args.test_category is None:
        # Server-only mode
        _run_server(args.host, args.port)
    elif args.test_category:
        # One-shot: start server, run BFCL, shut down
        _run_oneshot(args)
    else:
        parser.print_help()


def _run_server(host: str, port: int) -> None:
    """Start the SwayBot OpenAI-compatible server."""
    try:
        import uvicorn
    except ImportError as exc:
        print(
            f"Error: {exc}\n\n"
            "uvicorn is required. Install it with:\n"
            "  pip install 'swaybot[api]'",
            file=sys.stderr,
        )
        sys.exit(1)

    from benchmarks.bfcl.server import create_app
    app = create_app()
    print(f"Starting SwayBot BFCL server on {host}:{port}")
    print(f"Set these environment variables for BFCL:")
    print(f"  REMOTE_OPENAI_BASE_URL=http://{host}:{port}/v1")
    print(f"  REMOTE_OPENAI_API_KEY=swaybot")
    uvicorn.run(app, host=host, port=port)


def _register_swaybot_model(model_name: str) -> None:
    """Register a custom BFCL model config that talks to our local server.

    BFCL's model mapping is static, so we inject a SwayBot entry that uses the
    generic OpenAI chat-completions handler. The actual LLM backend is selected
    by SwayBot's own environment variables (SWAYBOT_API_BASE, SWAYBOT_MODEL).
    """
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )

    MODEL_CONFIG_MAPPING[model_name] = ModelConfig(
        model_name=model_name,
        display_name=f"{model_name} (SwayBot)",
        url="",
        org="SwayBot",
        license="",
        model_handler=OpenAICompletionsHandler,
        is_fc_model=True,
    )


def _run_oneshot(args: argparse.Namespace) -> None:
    """Start server in background, run BFCL, then shut down."""
    import threading

    # Start server in a thread
    try:
        import uvicorn
    except ImportError as exc:
        print(
            f"Error: {exc}\n\n"
            "uvicorn is required. Install it with:\n"
            "  pip install 'swaybot[api]'",
            file=sys.stderr,
        )
        sys.exit(1)

    from benchmarks.bfcl.server import create_app
    app = create_app()

    server_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": app, "host": args.host, "port": args.port, "log_level": "warning"},
        daemon=True,
    )
    server_thread.start()

    # Wait for server to be ready
    import time
    time.sleep(2)

    # BFCL writes results under BFCL_PROJECT_ROOT.
    if args.bfcl_project_root:
        os.environ["BFCL_PROJECT_ROOT"] = args.bfcl_project_root

    # The OpenAI-compatible client used by BFCL should talk to our local server.
    # LLMBrain itself is configured separately via SWAYBOT_* env vars.
    os.environ["OPENAI_BASE_URL"] = f"http://{args.host}:{args.port}/v1"
    os.environ["OPENAI_API_KEY"] = args.api_key or "swaybot"

    # Register a custom model config for SwayBot so bfcl_eval knows how to call it.
    _register_swaybot_model(args.model)

    # Run BFCL generate directly in-process so our model-config patch is visible.
    print(f"Running bfcl generate --model {args.model} --test-category {args.test_category}")
    from bfcl_eval._llm_response_generation import main as generation_main
    gen_args = SimpleNamespace(
        model=[args.model],
        test_category=[args.test_category],
        temperature=0.001,
        include_input_log=False,
        exclude_state_log=False,
        num_gpus=1,
        num_threads=None,
        gpu_memory_utilization=0.9,
        backend="sglang",
        skip_server_setup=False,
        local_model_path=None,
        result_dir=args.bfcl_project_root or "",
        allow_overwrite=False,
        run_ids=False,
    )
    try:
        generation_main(gen_args)
    except Exception as exc:
        print(f"bfcl generate failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()

    # Run BFCL evaluate directly in-process.
    print("Running bfcl evaluate")
    from bfcl_eval.eval_checker.eval_runner import main as evaluation_main
    try:
        evaluation_main(
            model=[args.model],
            test_categories=[args.test_category],
            result_dir=args.bfcl_project_root or None,
            score_dir=None,
            partial_eval=False,
        )
    except Exception as exc:
        print(f"bfcl evaluate failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()

    print("Done. Server shutting down.")


if __name__ == "__main__":
    main()
