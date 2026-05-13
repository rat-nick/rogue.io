# Server bind
HOST = "0.0.0.0"
PORT = 8765

# World
WORLD_W = 20000
WORLD_H = 20000

# Tick rate
TICK_RATE = 20          # ticks per second
TICK_INTERVAL = 1.0 / TICK_RATE  # seconds

# Food
FOOD_TARGET = 4000
FOOD_MASS = 1.0
REMNANT_DECAY_RATE = 0.10   # 20% of mass lost per second
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
# Speed formula (from Ogar / agar.io): BASE_SPEED / ceil(10 * sqrt(mass))^SPEED_EXPONENT
# where BASE_SPEED = playerSpeed(30) * 1.6 * 25 = 1200 and SPEED_EXPONENT = 0.32 (applied to size)
BASE_SPEED     = 1200  # world units/sec at size=1
SPEED_EXPONENT = 0.32    # applied to cell size (= ceil(10*sqrt(mass))), not raw mass
MIN_SPEED      = 50.0    # absolute floor (world units/sec)

MIN_SPLIT_MASS = 36    # minimum mass to perform a split (playerMinMassSplit)
MAX_CELLS = 16           # max simultaneous pieces per player (playerMaxCells)

# Merge timer: merge_timer = MERGE_TIME_BASE + MERGE_TIME_MASS_FACTOR * cell_mass  (seconds)
# Matches Ogar: floor(playerRecombineTime + 0.02 * mass) seconds
MERGE_TIME_BASE        = 30.0   # base seconds (playerRecombineTime)
MERGE_TIME_MASS_FACTOR = 0.02   # extra seconds per unit of mass

SPLIT_SPEED = 2000.0     # initial split burst (world units/sec)
SPLIT_DECEL = 3.7        # exponential deceleration factor; total travel ≈ SPLIT_SPEED/SPLIT_DECEL ≈ 540 units

# Ticks after a split during which same-player cells do NOT push each other apart.
# Matches Ogar collisionRestoreTicks = 15 (at 25 Hz = 600 ms); 12 ticks ≈ 600 ms at 20 Hz.
COLLISION_RESTORE_TICKS = 12

DECAY_RATE     = 0.002   # fraction of mass lost per second (playerMassDecayRate 0.002 / 25 Hz)
DECAY_MIN_MASS = 9.0     # only cells above this mass decay (playerMinMassDecay)
EAT_RATIO      = 1.3     # attacker must have ≥ EAT_RATIO × target mass to eat (baseEatingMassRequired)
MIN_CELL_MASS  = 7.0     # cell removed when mass falls below this (below DECAY_MIN_MASS so decay alone can't kill a cell)

# Eject (Ogar: ejectMassLoss=15, ejectMass=13, playerMinMassEject=32)
EJECT_MIN_MASS  = 32.0   # minimum cell mass to allow ejecting
EJECT_MASS_COST = 15.0   # mass deducted from cell on eject
EJECT_MASS      = 13.0   # mass of the ejected pellet
EJECT_SPEED     = 900.0  # initial speed of ejected pellet (world units/sec)
EJECT_DECEL     = 1.0    # exponential deceleration factor for ejected pellets

# Viruses
VIRUS_MASS = 100.0        # mass of a virus
VIRUS_RADIUS = 100.0      # visual radius (fixed, not mass-based)
VIRUS_FEED_THRESHOLD = 7  # number of ejects needed to make virus shoot a new one
VIRUS_SHOOT_SPEED = 600.0 # speed of new virus ejected from fed virus
VIRUS_CORNER_MARGIN = 100.0    # distance from edge for corner viruses
VIRUS_TARGET = 8         # target number of viruses to maintain
VIRUS_SPLIT_THRESHOLD = 150.0  # cells larger than this get split by virus

# Spatial grid
GRID_CELL_SIZE = 400

# Viewport (world units visible at base mass)
VIEW_BASE_SIZE = 3840
VIEW_MASS_SCALE = 0.3    # viewport_w = VIEW_BASE_SIZE * (total_mass/100)^VIEW_MASS_SCALE

# Radius formula: radius = sqrt(mass) * 10
RADIUS_FACTOR = 10.0

# Corner kill zone
CORNER_KILL_RADIUS = 400  # cells within this distance of any corner are destroyed

# Leaderboard broadcast interval (in ticks)
LEADERBOARD_INTERVAL = 10
DETACHED_DECAY_MULTIPLIER = 3.0   # split cells lose mass 3x faster

# Bot population (NEAT)
BOT_START = 25          # bots spawned at server start
BOT_MAX   = 25          # hard cap on concurrent bots
BOT_BURST = 50         # bots added per burst cycle
BOT_BURST_INTERVAL = 30.0  # seconds between bursts

# NEAT
NEAT_SAVE_PATH = "neat_population.pkl"

# Perception version — 1 = legacy 16-sector (177 inputs), 2 = raycast+quadrant (74 inputs)
PERCEPTION_VERSION = 2
