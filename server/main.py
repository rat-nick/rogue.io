"""rogue.io - authoritative game server entry point."""
from __future__ import annotations

import asyncio
import logging

from . import config
from .game import GameWorld

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


async def main() -> None:
    import websockets.asyncio.server as ws_server  # type: ignore

    world = GameWorld()
    world.seed_food()
    world.seed_bots()
    logger.info(f"Food seeded: {world.food_mgr.count()} pellets")

    tick_task = asyncio.create_task(world.tick_loop())

    server = await ws_server.serve(
        world.handle_connection,
        config.HOST,
        config.PORT,
    )
    logger.info(f"Server listening on ws://{config.HOST}:{config.PORT}")

    try:
        await server.serve_forever()
    finally:
        tick_task.cancel()
        server.close()
        await server.wait_closed()


if __name__ == '__main__':
    asyncio.run(main())
