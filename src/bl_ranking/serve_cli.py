"""Console entry point for the serving process."""
from __future__ import annotations

import uvicorn

from bl_ranking.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "bl_ranking.api:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.web_concurrency,
    )


if __name__ == "__main__":
    main()
