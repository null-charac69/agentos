"""
scripts/init_qdrant.py
───────────────────────
One-time initialisation script: creates Qdrant vector collections.

Run directly:
    python scripts/init_qdrant.py

Or via Docker Compose init service (automatic on first `docker compose up`).

Safe to run multiple times — existing collections are left untouched.
"""

from __future__ import annotations

import sys
import time

import structlog

# Ensure project root is in path when running as a script
sys.path.insert(0, ".")

from config.settings import get_settings
from memory.qdrant_store import ensure_collections_exist

logger = structlog.get_logger(__name__)


def main() -> None:
    settings = get_settings()

    print(f"Connecting to Qdrant at {settings.qdrant_url}…")

    # Retry a few times in case Qdrant is still starting up
    for attempt in range(1, 6):
        try:
            ensure_collections_exist()
            print(f"✓ Collections ready:")
            print(f"  - {settings.qdrant_collection_episodic}")
            print(f"  - {settings.qdrant_collection_knowledge}")
            print("Qdrant initialisation complete.")
            sys.exit(0)
        except Exception as exc:
            print(f"Attempt {attempt}/5 failed: {exc}")
            if attempt < 5:
                time.sleep(3)

    print("Failed to initialise Qdrant after 5 attempts.")
    sys.exit(1)


if __name__ == "__main__":
    main()
