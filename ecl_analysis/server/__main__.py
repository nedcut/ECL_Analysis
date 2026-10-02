"""Launch the local web UI server: `python -m ecl_analysis.server`."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .app import create_app

DEFAULT_PORT = 8765


def main() -> None:
    parser = argparse.ArgumentParser(description="Brightness Sorcerer web UI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--web-dist",
        default=str(Path(__file__).resolve().parents[2] / "web" / "dist"),
        help="Built frontend directory to serve at / (skipped when missing).",
    )
    args = parser.parse_args()

    app = create_app(web_dist=args.web_dist)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
