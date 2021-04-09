
class Config:
    PADDING = 10  # Distance between screen edge and player paddles (px)
    MAX_SCORE = 100  # Points one side must win to finish game
    WIDTH = 160  # Game window width (px)
    HEIGHT = 192  # Game window height (px)
    SPEEDUP = 1  # Flat multiplier to game movement speeds
    ACTIONS = ["UP", "DOWN", "NONE"]
    STATE_PACKET_INTERVAL_MS = 5
    GAME_FPS = 15
    AI_INFERENCES_PER_SECOND = 15  # Inferences per second
    BALL_MARKER_SIZE = 4  # Pixel height and width of experimental position markers
    CUSTOM = 0
    ATARI = 1
    HIT_PRACTICE = 2
    ATARI_ACTIONS = [2, 3]  # Control indices for "UP", "DOWN"
    ATARI_ACTION_SIZE = 2
    CUSTOM_ACTION_SIZE = 2
    CUSTOM_STATE_SHAPE = HEIGHT // 2, WIDTH // 2
    ATARI_STATE_SHAPE = 80, 80
    CUSTOM_STATE_SIZE = HEIGHT // 2 * WIDTH // 2
    ATARI_STATE_SIZE = 80 * 80
    BALL_DIAMETER = 6
    PADDLE_HEIGHT = 20
    PADDLE_WIDTH = 2
    LEFT_PADDLE_X = PADDING
    RIGHT_PADDLE_X = WIDTH - PADDING
    ENV_TYPE = CUSTOM
