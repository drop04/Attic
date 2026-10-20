"""python -m attic  ->  run the server."""
import os
import sys

import uvicorn

from .server import app_from_env


def main() -> None:
    if len(os.getenv("ATTIC_TOKEN", "")) < 16:
        sys.exit(
            "Set ATTIC_TOKEN (16+ chars), e.g.\n"
            "  export ATTIC_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
        )
    uvicorn.run(
        app_from_env(),
        host=os.getenv("ATTIC_HOST", "127.0.0.1"),
        port=int(os.getenv("ATTIC_PORT", "8787")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
