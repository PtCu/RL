from os import path
# 设置游戏屏幕大小
SCREEN_WIDTH = 480
SCREEN_HEIGHT = 600
FPS = 30
POWERUP_TIME = 5000
BAR_LENGTH = 100
BAR_HEIGHT = 10

# Define Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# 音视频文件夹
IMG_DIR = path.join(path.dirname(__file__), 'assets')
sound_folder = path.join(path.dirname(__file__), 'sounds')

# 玩家飞机图片
PLAYER_IMAGE = None

# 背景图片
BACKGROUND_IMAGE = None
BACKGROUND_RECT = None

# 陨石图片
METEOR_IMAGE = []

# 导弹、子弹图片
MISSILE_IMAGE = None
BULLET_IMAGE = None

# 爆炸动画（图片列表）
EXPLOSION_ANIM = {}

# powerup图片
POWERUP_IMAGE = {}

# 道具出现几率
POW_RATE = 0.1

# 每次生成的陨石数量
MOB_NUMS = 3

# 攻速
SHOOT_EVERY_F = 30
