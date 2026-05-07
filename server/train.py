"""Training-mode server entry point.

Run with:
    python -m server.train

Starts a training WebSocket server on TRAIN_PORT (8766 by default).
Training viewers connect to ws://host:8766 and receive real-time stats.
No real players are accepted — the world is bot-only.
"""
from __future__ import annotations

import asyncio
import logging

from . import config
from .training import TrainingWorld, TRAIN_PORT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


async def main() -> None:
    import websockets.asyncio.server as ws_server  # type: ignore

    world = TrainingWorld()

    tick_task = asyncio.create_task(world.tick_loop())

    server = await ws_server.serve(
        world.handle_viewer,
        config.HOST,
        TRAIN_PORT,
    )
    logger.info(f"Training server listening on ws://{config.HOST}:{TRAIN_PORT}")

    try:
        await server.serve_forever()
    finally:
        tick_task.cancel()
        server.close()
        await server.wait_closed()


if __name__ == '__main__':
    asyncio.run(main())
