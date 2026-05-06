# rogue.io

A multiplayer browser-based .io game (agar.io-style) with an authoritative WebSocket server, real-time physics, and an evolutionary AI bot system.

## Gameplay

- Move your cell toward your cursor
- Eat food pellets and smaller players to grow
- Split into smaller, faster cells to catch prey
- Avoid larger players that can eat you
- Cells decay over time — stay aggressive

### Controls

| Key / Action | Effect |
|---|---|
| Mouse move | Steer your cell toward cursor |
| `Q` | Split cell(s) in the direction of cursor |
| `W` | Eject mass (costs 20 mass, ejects 15-mass pellet) |

## Requirements

- Python 3.10+
- `pip install -r requirements.txt`

```
websockets>=13.0
msgpack>=1.0.0
neat-python>=0.92
```

## Running the Project

### 1. Start the server

```bash
cd server
python -m main
```

The server starts on `ws://0.0.0.0:8765` at 20 ticks/second.

### 2. Serve the client

Open a second terminal from the project root:

```bash
python -m http.server 8000
```

### 3. Open the game

| Page | URL |
|---|---|
| Play | http://localhost:8000/client/index.html |
| Spectate | http://localhost:8000/client/spectate.html |

Enter a name and click **Play** to join. The spectate view shows all active players without participating.

## Project Structure

```
server/
  main.py        — Entry point; starts WebSocket server and game loop
  game.py        — Core game world (tick loop, collisions, spawning)
  player.py      — Player and cell logic
  physics.py     — Movement, splitting, merging, decay
  food.py        — Food spawning and management
  bot.py         — Bot AI controller (genome-driven decision loop)
  genetics.py    — Genome pool, crossover, mutation, evolution
  protocol.py    — Message type constants and packet layouts
  spatial.py     — Spatial grid for O(1) collision detection
  config.py      — All tunable server/game constants
  neat.cfg       — NEAT neural network configuration

client/
  index.html        — Main game page
  spectate.html     — Spectator page
  js/main.js        — Game loop (60 FPS render, 20 Hz input)
  js/network.js     — WebSocket + msgpack codec
  js/renderer.js    — Canvas rendering
  js/viewport.js    — Camera following player
  js/minimap.js     — World overview minimap
  js/leaderboard.js — Top 10 scores HUD
  js/input.js       — Mouse and keyboard handling
  js/state.js       — Client-side state reconciliation
```

## Configuration

Edit `server/config.py` to tune gameplay:

| Setting | Default | Description |
|---|---|---|
| `PORT` | `8765` | WebSocket server port |
| `WORLD_W` / `WORLD_H` | `30000` | World dimensions |
| `TICK_RATE` | `20` | Ticks per second |
| `FOOD_TARGET` | `40000` | Target number of food pellets |
| `BOT_START` | `150` | Bots spawned at startup |
| `BOT_MAX` | `300` | Maximum concurrent bots |
| `SPLIT_MIN_MASS` | `35.0` | Minimum mass required to split |
| `MERGE_TIME` | `12` | Seconds before split cells merge |
| `DECAY_RATE` | `0.004` | Mass lost per second (0.4%) |

## Bot / AI System

The server spawns AI bots controlled by an 8-gene genome (detection radii, mass thresholds, timing). Bots compete against players and each other; when a bot dies its fitness score is recorded and the genome pool evolves via selection, crossover, and mutation. The best genomes are saved to `neat_population.pkl` and persist across server restarts.

## Protocol

Communication uses binary [MessagePack](https://msgpack.org/) frames over WebSocket.

| Message | Direction | Content |
|---|---|---|
| `MSG_INPUT (0x01)` | Client → Server | Mouse world position + split/eject flags (10 bytes) |
| `MSG_INIT (0x10)` | Server → Client | Player ID, world size, tick rate |
| `MSG_TICK (0x11)` | Server → Client | Visible cells, food delta, leaderboard |
| `MSG_DEAD (0x12)` | Server → Client | Final score and killer name |
