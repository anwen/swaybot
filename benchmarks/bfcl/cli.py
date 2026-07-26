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
import subprocess
import sys


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

    # Set BFCL environment variables
    os.environ["REMOTE_OPENAI_BASE_URL"] = f"http://{args.host}:{args.port}/v1"
    os.environ["REMOTE_OPENAI_API_KEY"] = args.api_key or "swaybot"
    if args.bfcl_project_root:
        os.environ["BFCL_PROJECT_ROOT"] = args.bfcl_project_root

    # Run BFCL generate
    print(f"Running bfcl generate --model {args.model} --test-category {args.test_category}")
    gen_cmd = [
        sys.executable, "-m", "bfcl_eval",
        "generate",
        "--model", args.model,
        "--test-category", args.test_category,
    ]
    try:
        result = subprocess.run(gen_cmd, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            print(f"bfcl generate failed with code {result.returncode}", file=sys.stderr)
    except FileNotFoundError:
        print(
            "Error: bfcl-eval not found. Install it with:\n"
            "  pip install bfcl-eval",
            file=sys.stderr,
        )
        sys.exit(1)

    # Run BFCL evaluate
    print("Running bfcl evaluate")
    eval_cmd = [
        sys.executable, "-m", "bfcl_eval",
        "evaluate",
    ]
    try:
        result = subprocess.run(eval_cmd, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    except FileNotFoundError:
        print("bfcl evaluate failed", file=sys.stderr)

    print("Done. Server shutting down.")


if __name__ == "__main__":
    main()
