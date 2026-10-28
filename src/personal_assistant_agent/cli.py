"""Command-line entry point."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="个人智能助理")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import truststore
    import uvicorn

    truststore.inject_into_ssl()
    uvicorn.run(
        "personal_assistant_agent.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
