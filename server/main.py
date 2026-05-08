"""rogue.io - authoritative game server entry point."""
from __future__ import annotations

import asyncio
import functools
import http.server
import logging
import pathlib
import threading

from . import config
from .game import GameWorld

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

_CLIENT_DIR = pathlib.Path(__file__).parent.parent / 'client'
HTTP_PORT = 8080


def _run_http_server() -> None:
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(_CLIENT_DIR),
    )
    # Suppress request logs from the HTTP server
    handler.log_message = lambda *_: None  # type: ignore[method-assign]
    with http.server.HTTPServer(('', HTTP_PORT), handler) as httpd:
        httpd.serve_forever()


async def main() -> None:
    import websockets.asyncio.server as ws_server  # type: ignore

    threading.Thread(target=_run_http_server, daemon=True).start()
    logger.info(f"Game client at http://localhost:{HTTP_PORT}")

    world = GameWorld()
    world.seed_food()
    world.seed_viruses()
    world.seed_bots()
    logger.info(f"Food seeded: {world.food_mgr.count()} pellets")
    logger.info(f"Viruses seeded: {world.virus_mgr.count()} viruses")

    tick_task = asyncio.create_task(world.tick_loop())

    server = await ws_server.serve(
        world.handle_connection,
        config.HOST,
        config.PORT,
    )
    logger.info(f"WebSocket server on ws://{config.HOST}:{config.PORT}")

    try:
        await server.serve_forever()
    finally:
        tick_task.cancel()
        server.close()
        await server.wait_closed()


if __name__ == '__main__':
    asyncio.run(main())
