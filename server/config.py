# Server bind
HOST = "0.0.0.0"
PORT = 8765

# World
WORLD_W = 30000
WORLD_H = 30000

# Tick rate
TICK_RATE = 20          # ticks per second
TICK_INTERVAL = 1.0 / TICK_RATE  # seconds

# Food
FOOD_TARGET = 40000
FOOD_MASS = 1.0
REMNANT_DECAY_RATE = 0.20   # 5% of mass lost per second
REMNANT_MIN_MASS = 1.0      # remnant removed when decayed below this
FOOD_COLORS = [
    (255, 85, 85),   # 0 red
    (255, 165, 0),   # 1 orange
    (255, 255, 85),  # 2 yellow
    (85, 255, 85),   # 3 green
    (85, 255, 255),  # 4 cyan
    (85, 85, 255),   # 5 blue
    (255, 85, 255),  # 6 magenta
    (255, 200, 200), # 7 pink
    (200, 255, 200), # 8 mint
    (200, 200, 255), # 9 lavender
    (255, 255, 200), # 10 cream
    (200, 230, 255), # 11 sky
]

# Cell physics
BASE_SPEED = 400.0       # world units/sec at mass=1
SPEED_EXPONENT = 0.2     # speed = BASE_SPEED / mass^SPEED_EXPONENT
MIN_SPLIT_MASS = 35.0
MAX_CELLS = 32           # max split cells per player
MERGE_TIME_BASE = 12.0   # seconds before split cells can merge
SPLIT_SPEED = 900.0      # initial ejection speed on split

# Split momentum
SPLIT_DECEL        = 3.5     # deceleration multiplier for split velocity (replaces hardcoded 8.0)
SPLIT_RECOIL       = 0.3     # fraction of SPLIT_SPEED applied as recoil to parent cell

# Merge attraction (kicks in when merge_timer <= 0)
MERGE_PULL_BASE     = 120.0  # initial pull speed in world units/sec
MERGE_PULL_MAX      = 550.0  # max pull speed at full ramp
MERGE_PULL_RAMP     = 6.0    # seconds to ramp from base to max pull
MERGE_PULL_MAX_TIME = 15.0   # how deep the timer goes negative (caps tracking)

DECAY_RATE = 0.004       # fraction of mass lost per second
DECAY_MIN_MASS = 100.0    # only decay above this
SPLIT_DECAY_RATE = 0.004  # faster decay for split cells (merge_timer > 0)
EAT_RATIO = 1.1          # must be this much larger to eat
MIN_CELL_MASS = 10.0     # remove cell if below this after decay

# Eject
EJECT_MASS_COST = 20.0
EJECT_SPEED = 800.0
EJECT_MASS = 15.0        # mass of ejected food pellet

# Spatial grid
GRID_CELL_SIZE = 200

# Viewport (world units visible at base mass)
VIEW_BASE_SIZE = 3840
VIEW_MASS_SCALE = 0.4    # viewport_w = VIEW_BASE_SIZE * (total_mass/100)^VIEW_MASS_SCALE

# Radius formula: radius = sqrt(mass) * 10
RADIUS_FACTOR = 10.0

# Leaderboard broadcast interval (in ticks)
LEADERBOARD_INTERVAL = 10
DETACHED_DECAY_MULTIPLIER = 3.0   # split cells lose mass 3x faster

# Bot population (NEAT)
BOT_START = 150          # bots spawned at server start
BOT_MAX   = 300          # hard cap on concurrent bots
BOT_BURST = 50           # bots added per burst cycle
BOT_BURST_INTERVAL = 60.0  # seconds between bursts

# NEAT
NEAT_SAVE_PATH = "neat_population.pkl"
