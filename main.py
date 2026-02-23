import pygame
import sys
import os
import random
import math
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import json # Added import json

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1600, 900
FPS = 60
MAP_WIDTH, MAP_HEIGHT = 4000, 4000

# SPELL CLASSES
SPELL_LABELS = [
    "0_TrebleClef",
    "1_BassClef",
    "2_Sharp",
    "3_Flat",
    "4_QuarterNote",
    "5_Accent",
    "6_EighthNote",
    "7_QuarterRest",
    "8_HalfRest"
]

# --- PyTorch Model Setup ---
class SpellCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(SpellCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Resource Path Helper for PyInstaller
def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, relative_path)

# Init Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
spell_model = SpellCNN(num_classes=9).to(device)
try:
    model_path = get_resource_path("spell_model.pth")
    spell_model.load_state_dict(torch.load(model_path, map_location=device))
    spell_model.eval()
    print("Successfully loaded spell_model.pth")
except Exception as e:
    print(f"Warning: Could not load spell_model.pth: {e}")

# Transform for incoming Pygame surface
spell_transform = transforms.Compose([
    transforms.ToTensor()
])

def predict_spell(surface):
    # Get surface pixels
    arr = pygame.surfarray.pixels_red(surface)
    coords = np.argwhere(arr > 0)
    if coords.size == 0:
        return None, 0.0
        
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(surface.get_width(), x_max + padding)
    y_max = min(surface.get_height(), y_max + padding)
    
    rect = pygame.Rect(x_min, y_min, x_max - x_min, y_max - y_min)
    cropped = surface.subsurface(rect).copy()
    
    max_side = max(rect.width, rect.height)
    square_surf = pygame.Surface((max_side, max_side))
    square_surf.fill((0, 0, 0))
    
    offset_x = (max_side - rect.width) // 2
    offset_y = (max_side - rect.height) // 2
    square_surf.blit(cropped, (offset_x, offset_y))
    
    final_surf = pygame.transform.smoothscale(square_surf, (64, 64))
    
    # Convert to grayscale values [0.0, 1.0]
    img_array = pygame.surfarray.pixels_red(final_surf).astype(np.float32) / 255.0
    img_array = img_array.T # Pygame uses (x,y), transform to (y,x)
    
    # Needs channel dim: (1, 64, 64) -> add batch dim (1, 1, 64, 64)
    tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = spell_model(tensor)
        probs = F.softmax(outputs, dim=1)
        max_prob, predicted = torch.max(probs, 1)
        
    return SPELL_LABELS[predicted.item()], max_prob.item()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BG_COLOR = (40, 60, 40) # Dark greenish background

# Set up display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("띳띠 서바이벌")
clock = pygame.time.Clock()

# --- Configuration Loading ---
PLAYER_CONFIG = {}
ENEMY_CONFIG = {}

try:
    player_conf_path = get_resource_path("player_config.json")
    with open(player_conf_path, "r", encoding="utf-8") as f:
        PLAYER_CONFIG = json.load(f)
except Exception as e:
    print(f"Warning: Could not load player_config.json, using fallback stats. {e}")
    PLAYER_CONFIG = {
        "max_health": 100, "base_speed": 400, "projectile_count": 1,
        "max_mana": 100.0, "mana_regen": 5.0, "damage_multiplier": 1.0
    }

try:
    enemy_conf_path = get_resource_path("enemy_config.json")
    with open(enemy_conf_path, "r", encoding="utf-8") as f:
        ENEMY_CONFIG = json.load(f)
except Exception as e:
    print(f"Warning: Could not load enemy_config.json, using fallback stats. {e}")
    ENEMY_CONFIG = {
        "1": {"health": 30, "speed": 100, "damage": 5, "scale": 0.5}
    }

# Asset Paths
ASSETS_DIR = get_resource_path("assets")
PLAYER_DIR = os.path.join(ASSETS_DIR, "플로로")
ENEMY_DIR = os.path.join(ASSETS_DIR, "띳띠")

# --- Asset Loading ---
def load_gif_frames(filepath, scale=1.0):
    frames = []
    try:
        if not os.path.exists(filepath):
            return frames
        img = Image.open(filepath)
        for frame_idx in range(getattr(img, 'n_frames', 1)):
            img.seek(frame_idx)
            frame_rgba = img.convert("RGBA")
            pygame_image = pygame.image.fromstring(
                frame_rgba.tobytes(), frame_rgba.size, frame_rgba.mode
            ).convert_alpha()
            if scale != 1.0:
                new_size = (int(pygame_image.get_width() * scale), int(pygame_image.get_height() * scale))
                pygame_image = pygame.transform.smoothscale(pygame_image, new_size)
            frames.append(pygame_image)
    except Exception as e:
        print(f"Error loading GIF {filepath}: {e}")
        # dummy
        surf = pygame.Surface((32, 32))
        surf.fill((255, 0, 255))
        frames.append(surf)
    return frames

def load_image(filepath, scale=1.0):
    try:
        if not os.path.exists(filepath):
            surf = pygame.Surface((32, 32))
            surf.fill((255, 0, 255))
            return surf
        pygame_image = pygame.image.load(filepath).convert_alpha()
        if scale != 1.0:
            new_size = (int(pygame_image.get_width() * scale), int(pygame_image.get_height() * scale))
            pygame_image = pygame.transform.smoothscale(pygame_image, new_size)
        return pygame_image
    except Exception as e:
        print(f"Error loading image {filepath}: {e}")
        surf = pygame.Surface((32, 32))
        surf.fill((255, 0, 255))
        return surf

# Load Player Images (Scale down to 75% which is 1.5x of previous 50%)
player_frames_default = load_gif_frames(os.path.join(PLAYER_DIR, "플로로_와쿠와쿠.gif"), scale=0.75)
player_frames_skill1 = load_gif_frames(os.path.join(PLAYER_DIR, "플로로_방냥자.gif"), scale=0.75)
player_frame_skill2 = [load_image(os.path.join(PLAYER_DIR, "플로로_할짝.png"), scale=0.75)]

# Load Enemy Images and parse their "level" from filename (e.g., "1. 띳무.png" -> level 1)
enemy_data = [] # List of dicts: {'image': surface, 'level': int}
if os.path.exists(ENEMY_DIR):
    for filename in os.listdir(ENEMY_DIR):
        if filename.endswith(".png"):
            # Try to extract number from filename
            level = 1
            try:
                # expecting format like "1. 띳무.png"
                parts = filename.split('.')
                if parts[0].isdigit():
                    level = int(parts[0])
            except:
                pass
            
            # We will scale based on level dynamically in the Enemy class.
            # However, we need the original image surface first to scale it later without losing quality.
            img = load_image(os.path.join(ENEMY_DIR, filename))
            enemy_data.append({'image': img, 'level': level})

# Sort enemies by level just in case
enemy_data.sort(key=lambda x: x['level'])

if not enemy_data:
    surf = pygame.Surface((50, 50))
    surf.fill(RED)
    enemy_data.append({'image': surf, 'level': 1})

# Load Projectile Animation Frames
projectile_frames_flying = []
projectile_frames_impact = []
# Base scale: 0.25 for flying, 1.6x larger (0.4) for impact
FLYING_SCALE = 0.25 
IMPACT_SCALE = 0.40

for i in range(8):
    filepath = os.path.join(PLAYER_DIR, f"플로로_공격_이펙트{i}.png")
    if os.path.exists(filepath):
        if i <= 2:
            img = load_image(filepath, scale=FLYING_SCALE) 
            projectile_frames_flying.append(img)
        else:
            img = load_image(filepath, scale=IMPACT_SCALE) 
            projectile_frames_impact.append(img)

if not projectile_frames_flying:
    surf = pygame.Surface((20, 20), pygame.SRCALPHA)
    pygame.draw.circle(surf, (255, 255, 0), (10, 10), 10)
    projectile_frames_flying.append(surf)
if not projectile_frames_impact:
    surf = pygame.Surface((30, 30), pygame.SRCALPHA)
    pygame.draw.circle(surf, (255, 0, 0), (15, 15), 15)
    projectile_frames_impact.append(surf)

# Load Spell UI Icons
SPELL_ICONS_DIR = os.path.join(ASSETS_DIR, "spell")
spell_icons_queue = {} # ~30x30 for bottom right queue
spell_icons_floating = {} # ~60x60 for floating feedback

spell_filename_map = {
    "TrebleClef": "TrebleClef.png",
    "BassClef": "BassClef.png",
    "Sharp": "Sharp.png",
    "Flat": "Flat.png",
    "QuarterNote": "QuarterNote.png",
    "Accent": "accent.png",
    "EighthNote": "EighthNote.png",
    "QuarterRest": "QuarterRest.png",
    "HalfRest": "HalfRest.png"
}

for label in SPELL_LABELS:
    spell_name = label.split('_')[1]
    filename = spell_filename_map.get(spell_name, "")
    filepath = os.path.join(SPELL_ICONS_DIR, filename)
    
    img_queue = load_image(filepath) # default load
    if getattr(img_queue, 'get_size', None):
        if img_queue.get_size() != (32, 32): # Not the dummy pink square
            spell_icons_queue[label] = pygame.transform.smoothscale(img_queue, (30, 30))
            spell_icons_floating[label] = pygame.transform.smoothscale(img_queue, (80, 80))
        else:
            spell_icons_queue[label] = img_queue
            spell_icons_floating[label] = pygame.transform.scale(img_queue, (80, 80))
    else:
        # fallback dummy
        surf = pygame.Surface((30, 30))
        surf.fill((255, 0, 255))
        spell_icons_queue[label] = surf
        spell_icons_floating[label] = pygame.transform.scale(surf, (80, 80))


# --- Classes ---
class Camera:
    def __init__(self, width, height):
        self.camera = pygame.Rect(0, 0, width, height)
        self.width = width
        self.height = height

    def apply(self, entity):
        return entity.rect.move(self.camera.topleft)

    def apply_rect(self, rect):
        return rect.move(self.camera.topleft)

    def update(self, target):
        x = -target.rect.centerx + int(WIDTH / 2)
        y = -target.rect.centery + int(HEIGHT / 2)
        
        # Clamp camera to map boundaries
        x = min(0, x)
        y = min(0, y)
        x = max(-(self.width - WIDTH), x)
        y = max(-(self.height - HEIGHT), y)
        
        self.camera = pygame.Rect(x, y, self.width, self.height)

class Player(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.frames = player_frames_default if player_frames_default else [pygame.Surface((64, 64))]
        self.current_frame = 0
        self.image = self.frames[self.current_frame]
        self.rect = self.image.get_rect(center=(x, y))
        self.pos = pygame.math.Vector2(x, y)
        
        # Base Stats from Config
        self.base_speed = PLAYER_CONFIG.get("base_speed", 400)
        self.speed = self.base_speed
        self.max_health = PLAYER_CONFIG.get("max_health", 100)
        self.health = self.max_health
        
        # Combat Stats
        self.damage_multiplier = PLAYER_CONFIG.get("damage_multiplier", 1.0)
        self.attack_cooldown_multiplier = 1.0
        self.projectile_count = PLAYER_CONFIG.get("projectile_count", 1)
        self.last_attack_time = 0  # To manage attack speed later if needed, currently click-based
        
        # Progression
        self.level = 1
        self.exp = 0
        self.exp_to_next_level = 50
        
        # Mana (Inspiration / 악상)
        self.max_mana = PLAYER_CONFIG.get("max_mana", 100.0)
        self.mana = self.max_mana
        self.mana_regen = PLAYER_CONFIG.get("mana_regen", 5.0) # per second
        
        # Stance (Clefs)
        self.stance = "Treble" # Treble or Bass
        
        self.animation_timer = 0
        self.animation_speed = 0.1
        
        self.skill_active = False
        self.skill_timer = 0
        self.skill_duration = 3.0
        
        # Slightly smaller hitbox than the image
        self.hitbox = self.rect.inflate(-20, -20)

    def gain_exp(self, amount):
        self.exp += amount
        if self.exp >= self.exp_to_next_level:
            self.exp -= self.exp_to_next_level
            self.level += 1
            self.exp_to_next_level = int(self.exp_to_next_level * 1.5)
            # Heal slightly on level up
            self.health = min(self.max_health, self.health + 20)
            return True # Indicates level up happened
        return False

    def apply_upgrade(self, upgrade_type):
        if upgrade_type == "damage":
            self.damage_multiplier += 0.2
        elif upgrade_type == "speed":
            self.base_speed += 50
            self.speed = self.base_speed
        elif upgrade_type == "health":
            self.max_health += 50
            self.health += 50
        elif upgrade_type == "projectile":
            self.projectile_count += 1

    def activate_skill(self):
        if not self.skill_active:
            self.skill_active = True
            self.skill_timer = self.skill_duration
            self.frames = player_frames_skill1 if player_frames_skill1 else self.frames
            self.current_frame = 0

    def update(self, dt):
        # Handle input
        keys = pygame.key.get_pressed()
        dir_vector = pygame.math.Vector2(0, 0)
        
        if keys[pygame.K_w]: dir_vector.y = -1
        if keys[pygame.K_s]: dir_vector.y = 1
        if keys[pygame.K_a]: dir_vector.x = -1
        if keys[pygame.K_d]: dir_vector.x = 1
            
        if dir_vector.length_squared() > 0:
            dir_vector = dir_vector.normalize()
            
        self.pos += dir_vector * self.speed * dt
        
        # Keep within map boundaries
        self.pos.x = max(0, min(MAP_WIDTH, self.pos.x))
        self.pos.y = max(0, min(MAP_HEIGHT, self.pos.y))
        
        self.rect.centerx = round(self.pos.x)
        self.rect.centery = round(self.pos.y)
        self.hitbox.center = self.rect.center

        # Animation
        self.animation_timer += dt
        if self.animation_timer >= self.animation_speed:
            self.animation_timer = 0
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            self.image = self.frames[self.current_frame]

        # Regenerate Mana
        self.mana = min(self.max_mana, self.mana + self.mana_regen * dt)

        # Skill timer
        if self.skill_active:
            self.skill_timer -= dt
            if self.skill_timer <= 0:
                self.skill_active = False
                self.frames = player_frames_default
                self.current_frame = 0

class Projectile(pygame.sprite.Sprite):
    def __init__(self, x, y, target_x, target_y):
        super().__init__()
        self.state = "flying" # "flying" or "impact"
        self.frames = projectile_frames_flying
        self.current_frame = 0
        self.base_image = self.frames[self.current_frame]
        
        self.pos = pygame.math.Vector2(x, y)
        direction = pygame.math.Vector2(target_x - x, target_y - y)
        
        if direction.length_squared() > 0:
            self.direction = direction.normalize()
            # Calculate angle in degrees for rotation
            self.angle_deg = math.degrees(math.atan2(-self.direction.y, self.direction.x))
            self.image = pygame.transform.rotate(self.base_image, self.angle_deg)
        else:
            self.direction = pygame.math.Vector2(1, 0)
            self.angle_deg = 0
            self.image = self.base_image
            
        self.rect = self.image.get_rect(center=(x, y))
            
        self.speed = 800
        self.lifetime = 1.5
        self.damage = 25
        
        self.animation_timer = 0
        self.animation_speed = 0.08 # Slightly slower so we can see frames 0,1,2 spinning
        
        self.impact_scale_mod = 1.0
        self.impact_played = False

    def trigger_impact(self):
        if self.state == "flying":
            self.state = "impact"
            self.frames = projectile_frames_impact
            self.current_frame = 0
            self.animation_timer = 0
            self.animation_speed = 0.05
            self.speed = 0 # stop moving
            # Keep the rotation consistency, and apply impact_scale_mod
            self.base_image = self.frames[self.current_frame]
            if self.impact_scale_mod != 1.0:
                w, h = self.base_image.get_size()
                self.base_image = pygame.transform.smoothscale(self.base_image, (int(w * self.impact_scale_mod), int(h * self.impact_scale_mod)))
            
            self.image = pygame.transform.rotate(self.base_image, self.angle_deg)
            self.rect = self.image.get_rect(center=(round(self.pos.x), round(self.pos.y)))

    def update(self, dt):
        if self.state == "flying":
            self.pos += self.direction * self.speed * dt
            
            # Animation
            self.animation_timer += dt
            if self.animation_timer >= self.animation_speed:
                self.animation_timer = 0
                self.current_frame = (self.current_frame + 1) % len(self.frames)
                self.base_image = self.frames[self.current_frame]
                self.image = pygame.transform.rotate(self.base_image, self.angle_deg)
                
            self.rect = self.image.get_rect(center=(round(self.pos.x), round(self.pos.y)))
            
            self.lifetime -= dt
            if self.lifetime <= 0:
                self.kill()
                
        elif self.state == "impact":
            self.animation_timer += dt
            if self.animation_timer >= self.animation_speed:
                self.animation_timer = 0
                self.current_frame += 1
                if self.current_frame >= len(self.frames):
                    self.kill() # Animation finished
                else:
                    self.base_image = self.frames[self.current_frame]
                    if self.impact_scale_mod != 1.0:
                        w, h = self.base_image.get_size()
                        self.base_image = pygame.transform.smoothscale(self.base_image, (int(w * self.impact_scale_mod), int(h * self.impact_scale_mod)))
                    
                    self.image = pygame.transform.rotate(self.base_image, self.angle_deg)
                    self.rect = self.image.get_rect(center=(round(self.pos.x), round(self.pos.y)))

class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y, enemy_info):
        super().__init__()
        self.level = enemy_info['level']
        
        # Pull stats from ENEMY_CONFIG, fallback to level '1' if not found
        lvl_str = str(self.level)
        conf = ENEMY_CONFIG.get(lvl_str, ENEMY_CONFIG.get("1", {"health": 30, "speed": 100, "damage": 5, "scale": 0.5}))
        
        # Apply scaling visually right away
        scale_mod = conf.get("scale", 0.5)
        orig_img = enemy_info['image']
        new_size = (int(orig_img.get_width() * scale_mod), int(orig_img.get_height() * scale_mod))
        self.image = pygame.transform.smoothscale(orig_img, new_size)
        
        self.rect = self.image.get_rect(center=(x, y))
        self.pos = pygame.math.Vector2(x, y)
        
        # Slightly smaller hitbox than the image
        self.hitbox = self.rect.inflate(-20, -20)
        
        self.max_health = conf.get("health", 30.0)
        self.health = self.max_health
        self.speed = conf.get("speed", 100.0)
        self.damage = conf.get("damage", 5.0)
        
        # Flash effect handling
        self.is_flashing = False
        self.flash_timer = 0
        self.flash_duration = 0.1

    def update(self, dt, target_pos):
        direction = target_pos - self.pos
        if direction.length_squared() > 0:
            direction = direction.normalize()
            
        self.pos += direction * self.speed * dt
        self.rect.center = (round(self.pos.x), round(self.pos.y))
        self.hitbox.center = self.rect.center

# --- Game State & Entities ---
all_sprites = pygame.sprite.Group()
enemies = pygame.sprite.Group()
projectiles = pygame.sprite.Group()

player = Player(MAP_WIDTH // 2, MAP_HEIGHT // 2)
all_sprites.add(player)

camera = Camera(MAP_WIDTH, MAP_HEIGHT)

spawn_timer = 0
spawn_rate = 1.0

# Using default internal font if SysFont fails
try:
    font = pygame.font.SysFont("malgungothic", 36)
    ui_font = pygame.font.SysFont("malgungothic", 24)
    title_font = pygame.font.SysFont("malgungothic", 48)
except:
    font = pygame.font.Font(None, 36)
    ui_font = pygame.font.Font(None, 24)
    title_font = pygame.font.Font(None, 48)

score = 0
game_time = 0.0 # Track elapsed time for scaling

# Game States
STATE_PLAYING = 0
STATE_LEVEL_UP = 1
STATE_GAME_OVER = 2
current_state = STATE_PLAYING

# Upgrade Options
upgrade_pool = [
    {"id": "damage", "name": "공격력 증가", "desc": "피해량 20% 증가"},
    {"id": "speed", "name": "이동 속도 증가", "desc": "이동 속도 증가"},
    {"id": "health", "name": "최대 체력 증가", "desc": "최대 체력 50 증가 & 회복"},
    {"id": "projectile", "name": "발사체 추가", "desc": "공격 시 투사체 1개 추가"}
]
current_upgrades_offered = []
upgrade_rects = []

def create_upgrade_overlay():
    global current_upgrades_offered, upgrade_rects
    # Randomly pick 3 from the pool (can allow duplicates if desired, but let's sample without replacement if possible)
    if len(upgrade_pool) >= 3:
        current_upgrades_offered = random.sample(upgrade_pool, 3)
    else:
        # Fallback if less than 3 unique options
        current_upgrades_offered = [random.choice(upgrade_pool) for _ in range(3)]
        
    upgrade_rects = []
    card_width, card_height = 300, 400
    spacing = 50
    total_width = (card_width * 3) + (spacing * 2)
    start_x = (WIDTH - total_width) // 2
    start_y = (HEIGHT - card_height) // 2
    
    for i in range(3):
        rect = pygame.Rect(start_x + i * (card_width + spacing), start_y, card_width, card_height)
        upgrade_rects.append(rect)

spellbook_queue = []
MAX_SPELL_QUEUE = 5

is_drawing = False
drawing_points = []
drawing_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

# UI variables
show_damage_numbers = []
spell_feedback_timer = 0.0
spell_feedback_label = ""
spell_feedback_color = WHITE

# Score Execution State Machine
active_score = []
score_execution_timer = 0.0
SCORE_EXECUTION_DELAY = 0.5
global_dmg_mult = 1.0
global_count_mult = 1
accent_next_note = False

# --- Main Loop ---
running = True
game_time = 0.0
current_stage = 1
score = 0
spawn_timer = 0
spawn_rate = 1.0 # Base spawn rate
while running:
    # Always tick clock to avoid spiral of death, but dt will be used based on state
    raw_dt = clock.tick(FPS) / 1000.0
    
    if current_state == STATE_PLAYING:
        dt = raw_dt
        game_time += dt
        
        # Stage calculation (every 60 seconds is a new stage)
        current_stage = int(game_time // 60) + 1
        if spell_feedback_timer > 0:
            spell_feedback_timer -= dt

        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LSHIFT:
                    player.activate_skill()
                elif event.key == pygame.K_SPACE:
                    valid_points = [p for p in drawing_points if p is not None]
                    if len(valid_points) > 5: # Valid draw
                        # Render white lines on black surface for model prediction
                        test_surf = pygame.Surface((WIDTH, HEIGHT))
                        test_surf.fill((0, 0, 0))
                        for i in range(1, len(drawing_points)):
                            if drawing_points[i-1] is not None and drawing_points[i] is not None:
                                pygame.draw.line(test_surf, (255, 255, 255), drawing_points[i-1], drawing_points[i], 15)
                                pygame.draw.circle(test_surf, (255, 255, 255), drawing_points[i], 7)
                                
                        spell_name, prob = predict_spell(test_surf)
                        if spell_name and prob >= 0.50:
                            print(f"Spell Cast Success! {spell_name} (Prob: {prob:.2f})")
                            spell_feedback_label = spell_name
                            spell_feedback_color = GREEN
                            spell_feedback_timer = 2.0
                            if len(spellbook_queue) < MAX_SPELL_QUEUE:
                                spellbook_queue.append(spell_name)
                        else:
                            print(f"Spell Cast FAIL! Best match was {spell_name} but only (Prob: {prob:.2f})")
                            spell_feedback_label = "FAIL"
                            spell_feedback_color = RED
                            spell_feedback_timer = 2.0
                            player.health -= 20 # Punishment damage
                            if player.health <= 0:
                                current_state = STATE_GAME_OVER
                    
                    drawing_points.clear()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click - start drawing
                    is_drawing = True
                    # Do not clear if we are starting a new stroke for the same spell
                    drawing_points.append(event.pos)
                elif event.button == 3: # Right click - Execute Score
                    if len(spellbook_queue) > 0 and len(active_score) == 0:
                        active_score = spellbook_queue.copy()
                        spellbook_queue.clear()
                        
                        # Pre-calculate Globals (Sharps and Flats)
                        global_dmg_mult = 1.0
                        global_count_mult = 1
                        
                        for sp_full in active_score:
                            sp_name = sp_full.split('_')[1] if '_' in sp_full else sp_full
                            if sp_name == "Sharp":
                                global_dmg_mult *= 1.7
                            elif sp_name == "Flat":
                                global_count_mult *= 2
                                
                        score_execution_timer = 0.0
                        accent_next_note = False
            elif event.type == pygame.MOUSEMOTION:
                if is_drawing:
                    drawing_points.append(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and is_drawing:
                    is_drawing = False
                    drawing_points.append(None) # Mark the end of this stroke

        # Simple enemy spawning just outside camera view
        spawn_timer += dt
        if spawn_timer >= spawn_rate:
            spawn_timer = 0
            side = random.choice(["top", "bottom", "left", "right"])
            cam_rect = pygame.Rect(-camera.camera.x, -camera.camera.y, WIDTH, HEIGHT)
            margin = 100
            
            if side == "top":
                ex = random.randint(cam_rect.left - margin, cam_rect.right + margin)
                ey = cam_rect.top - margin
            elif side == "bottom":
                ex = random.randint(cam_rect.left - margin, cam_rect.right + margin)
                ey = cam_rect.bottom + margin
            elif side == "left":
                ex = cam_rect.left - margin
                ey = random.randint(cam_rect.top - margin, cam_rect.bottom + margin)
            else:
                ex = cam_rect.right + margin
                ey = random.randint(cam_rect.top - margin, cam_rect.bottom + margin)
                
            # Ensure spawn within map bounds
            ex = max(0, min(MAP_WIDTH, ex))
            ey = max(0, min(MAP_HEIGHT, ey))
            
            # --- Stage-based Probability Spawning ---
            available_levels = sorted([info['level'] for info in enemy_data])
            if not available_levels:
                available_levels = [1]
                
            weights = []
            max_level_allowed = min(len(available_levels), current_stage)
            
            for lvl in available_levels:
                if lvl > current_stage:
                    weights.append(0.0) # Cannot spawn enemies higher than current stage
                elif lvl == current_stage:
                    weights.append(20.0) # 20% chance for the newest, strongest enemy
                else:
                    # Distribute remaining 80% among lower levels, favoring slightly higher ones
                    weight = 80.0 / max(1, (current_stage - 1))
                    weights.append(weight)
                    
            if sum(weights) == 0:
                weights[0] = 100.0
                
            chosen_level = random.choices(available_levels, weights=weights, k=1)[0]
            
            # Find the corresponding info dictionary
            chosen_enemy_info = next((info for info in enemy_data if info['level'] == chosen_level), enemy_data[0])

            enemy = Enemy(ex, ey, chosen_enemy_info)
            all_sprites.add(enemy)
            enemies.add(enemy)
            
            # Difficulty ramp: Spawn rate caps at 0.1s
            spawn_rate = max(0.1, 1.0 - (current_stage * 0.05))

        # Process the Score (악보) sequential execution
        if len(active_score) > 0:
            score_execution_timer -= dt
            while len(active_score) > 0 and score_execution_timer <= 0:
                current_spell_full = active_score[0]
                sp_name = current_spell_full.split('_')[1] if '_' in current_spell_full else current_spell_full
                
                # Check for instant spells
                if sp_name in ["TrebleClef", "BassClef", "Sharp", "Flat", "Accent"]:
                    active_score.pop(0)
                    if sp_name == "TrebleClef":
                        player.stance = "Treble"
                        spell_feedback_label = "공격 모드 (Treble)"
                        spell_feedback_color = (0, 255, 0)
                        spell_feedback_timer = 2.0
                    elif sp_name == "BassClef":
                        player.stance = "Bass"
                        spell_feedback_label = "서포트 모드 (Bass)"
                        spell_feedback_color = (150, 150, 255)
                        spell_feedback_timer = 2.0
                    elif sp_name == "Sharp":
                        spell_feedback_label = "다음 공격 강화!"
                        spell_feedback_color = (255, 0, 0)
                        spell_feedback_timer = 1.0
                    elif sp_name == "Flat":
                        spell_feedback_label = "투사체 2배!"
                        spell_feedback_color = (0, 0, 255)
                        spell_feedback_timer = 1.0
                    elif sp_name == "Accent":
                        accent_next_note = True
                        spell_feedback_label = "다음 공격 확산(AoE)!"
                        spell_feedback_color = (255, 200, 0)
                        spell_feedback_timer = 1.0
                else:
                    # Sequential spells: Notes and Rests
                    base_cost = 0.0
                    if sp_name == "QuarterNote": base_cost = 30.0
                    elif sp_name == "EighthNote": base_cost = 15.0
                    
                    if player.mana >= base_cost:
                        player.mana -= base_cost
                        active_score.pop(0)
                        
                        # Execute
                        if sp_name == "QuarterRest":
                            amt = 50.0 if player.stance == "Bass" else 30.0
                            player.mana = min(player.max_mana, player.mana + amt)
                            spell_feedback_label = f"악상 회복 (+{int(amt)})"
                            spell_feedback_color = (100, 200, 255)
                            spell_feedback_timer = 2.0
                        elif sp_name == "HalfRest":
                            amt = 40.0 if player.stance == "Bass" else 20.0
                            player.health = min(player.max_health, player.health + amt)
                            spell_feedback_label = f"체력 회복 (+{int(amt)})"
                            spell_feedback_color = (100, 255, 100)
                            spell_feedback_timer = 2.0
                        elif sp_name in ["QuarterNote", "EighthNote"]:
                            shots_to_fire = 1
                            proj_dmg_mult = 1.0
                            proj_scale_mult = 1.0
                            is_aoe = False
                            
                            if sp_name == "QuarterNote":
                                shots_to_fire = 1
                                proj_dmg_mult = 1.5
                                proj_scale_mult = 1.5
                            elif sp_name == "EighthNote":
                                shots_to_fire = 2
                                proj_dmg_mult = 0.5
                                proj_scale_mult = 0.7
                                
                            if player.stance == "Bass":
                                proj_dmg_mult *= 0.6
                                
                            if accent_next_note:
                                is_aoe = True
                                proj_scale_mult *= 2.0
                                accent_next_note = False
                                
                            # Target selection logic
                            nearest_enemy = None
                            min_dist = float('inf')
                            for enemy in enemies:
                                dist = (enemy.pos - player.pos).length_squared()
                                if dist < min_dist:
                                    min_dist = dist
                                    nearest_enemy = enemy
                                    
                            if nearest_enemy:
                                p_count = player.projectile_count * shots_to_fire * global_count_mult
                                spread = 0.2
                                dx = nearest_enemy.pos.x - player.rect.centerx
                                dy = nearest_enemy.pos.y - player.rect.centery
                                base_angle = math.atan2(dy, dx)
                                start_angle = base_angle - (spread * (p_count - 1) / 2)
                                
                                for i in range(p_count):
                                    angle = start_angle + (i * spread)
                                    tx = player.rect.centerx + math.cos(angle) * 100
                                    ty = player.rect.centery + math.sin(angle) * 100
                                    proj = Projectile(player.rect.centerx, player.rect.centery, tx, ty)
                                    
                                    proj.damage = proj.damage * player.damage_multiplier * global_dmg_mult * proj_dmg_mult
                                    proj.is_aoe = is_aoe
                                    
                                    if proj_scale_mult != 1.0:
                                        w, h = proj.base_image.get_size()
                                        proj.base_image = pygame.transform.smoothscale(proj.base_image, (int(w * proj_scale_mult), int(h * proj_scale_mult)))
                                        proj.image = pygame.transform.rotate(proj.base_image, proj.angle_deg)
                                        proj.rect = proj.image.get_rect(center=(round(proj.pos.x), round(proj.pos.y)))
                                        proj.impact_scale_mod = proj_scale_mult
                                        
                                    all_sprites.add(proj)
                                    projectiles.add(proj)
                        
                        # Set delay for the NEXT iteration since we processed a Note/Rest
                        score_execution_timer = SCORE_EXECUTION_DELAY
                    else:
                        # Insufficient Mana aborts the rest of the score
                        spell_feedback_label = "실패 / 악상 부족!"
                        spell_feedback_color = (255, 0, 0)
                        spell_feedback_timer = 2.0
                        active_score.clear()
                        accent_next_note = False
                        break

        # Updates
        player.update(dt)
        projectiles.update(dt)
        
        # Check enemy-player collision and update enemies
        for enemy in enemies:
            enemy.update(dt, player.pos)
            if player.hitbox.colliderect(enemy.hitbox):
                player.health -= enemy.damage * dt
                if player.health <= 0:
                    current_state = STATE_GAME_OVER

        # Check projectile-enemy collection
        # Using lambda for hitbox collision. Do not kill projectile immediately.
        hits = pygame.sprite.groupcollide(enemies, projectiles, False, False, collided=lambda e, p: e.hitbox.colliderect(p.rect))
        for enemy, projs in hits.items():
            for p in projs:
                if p.state == "flying":
                    enemy.health -= p.damage
                    
                    # Process Splash AoE Damage if projectile has it
                    if getattr(p, 'is_aoe', False):
                        for other_enemy in enemies:
                            if other_enemy != enemy and (other_enemy.pos - p.pos).length_squared() <= 150**2:
                                other_enemy.health -= p.damage
                                
                    p.trigger_impact()
                    
        # Check deaths separately because of AoE processing
        for enemy in list(enemies):
            if enemy.health <= 0:
                enemy.kill()
                score += 10
                exp_amount = 10 + (enemy.level * 5)
                leveled_up = player.gain_exp(exp_amount)
                if leveled_up:
                    current_state = STATE_LEVEL_UP
                    create_upgrade_overlay()

        # Update Camera
        camera.update(player)

    elif current_state == STATE_LEVEL_UP:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()
                    for i, rect in enumerate(upgrade_rects):
                        if rect.collidepoint(mouse_pos):
                            chosen_upgrade = current_upgrades_offered[i]
                            player.apply_upgrade(chosen_upgrade["id"])
                            current_state = STATE_PLAYING
                            break

    elif current_state == STATE_GAME_OVER:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    # Draw Background
    screen.fill(BG_COLOR)
    
    # Draw Sprites (Sorted by Y coordinates to give false depth)
    for sprite in sorted(all_sprites, key=lambda s: s.rect.bottom):
        screen.blit(sprite.image, camera.apply(sprite))

    # Draw the current drawing line
    if len(drawing_points) > 1:
        for i in range(1, len(drawing_points)):
            if drawing_points[i-1] is not None and drawing_points[i] is not None:
                pygame.draw.line(screen, (0, 255, 255), drawing_points[i-1], drawing_points[i], 5)

    # Draw UI
    if current_state in (STATE_PLAYING, STATE_LEVEL_UP):
        # Health bar
        pygame.draw.rect(screen, BLACK, (18, 18, 204, 24))
        pygame.draw.rect(screen, RED, (20, 20, 200, 20))
        health_ratio = max(0, player.health / player.max_health)
        pygame.draw.rect(screen, GREEN, (20, 20, 200 * health_ratio, 20))
        
        # EXP bar
        pygame.draw.rect(screen, BLACK, (18, 58, 204, 14))
        pygame.draw.rect(screen, (50, 50, 50), (20, 60, 200, 10))
        exp_ratio = max(0, player.exp / player.exp_to_next_level)
        pygame.draw.rect(screen, (0, 150, 255), (20, 60, 200 * exp_ratio, 10))
        
        # Mana (악상) bar
        pygame.draw.rect(screen, BLACK, (18, 88, 204, 14))
        pygame.draw.rect(screen, (50, 50, 50), (20, 90, 200, 10))
        mana_ratio = max(0, player.mana / player.max_mana)
        pygame.draw.rect(screen, (150, 0, 255), (20, 90, 200 * mana_ratio, 10))
        
        # Spellboard UI (Queue or Active Score)
        queue_start_x = WIDTH - 290
        queue_start_y = HEIGHT - 80
        pygame.draw.rect(screen, (30, 30, 30), (queue_start_x, queue_start_y, 270, 60))
        
        # Decide which queue to display (Active Execution vs Preparation)
        display_queue = active_score if len(active_score) > 0 else spellbook_queue
        border_color = (255, 200, 0) if len(active_score) > 0 else WHITE
        pygame.draw.rect(screen, border_color, (queue_start_x, queue_start_y, 270, 60), 2)
        
        try:
            state_str = "실행 중" if len(active_score) > 0 else "대기열"
            spell_title = ui_font.render(f"Score - {state_str} ({player.stance}):", True, border_color)
            screen.blit(spell_title, (queue_start_x + 10, queue_start_y - 25))
            
            # Draw individual spell icons / text
            for idx, sp in enumerate(display_queue):
                center_x = queue_start_x + 30 + (idx*40)
                center_y = queue_start_y + 30
                
                # Active note highlights if executing
                if len(active_score) > 0 and idx == 0:
                    pygame.draw.circle(screen, (100, 100, 50), (center_x, center_y), 20)
                    pygame.draw.circle(screen, (255, 255, 100), (center_x, center_y), 20, 3)
                else:
                    pygame.draw.circle(screen, (50, 50, 50), (center_x, center_y), 18)
                    pygame.draw.circle(screen, (100, 255, 100), (center_x, center_y), 18, 2)
                
                # Draw the spell image
                if sp in spell_icons_queue:
                    icon = spell_icons_queue[sp]
                    icon_rect = icon.get_rect(center=(center_x, center_y))
                    screen.blit(icon, icon_rect)
                else:
                    sp_name = sp.split('_')[1] if '_' in sp else sp
                    badge = ui_font.render(sp_name[:2], True, WHITE)
                    screen.blit(badge, (center_x - badge.get_width()//2, center_y - badge.get_height()//2))
                
        except:
            pass
        
        try:
            level_text = font.render(f"Lv: {player.level}", True, WHITE)
            screen.blit(level_text, (230, 15))
            
            stage_text = font.render(f"Stage: {current_stage}", True, (255, 200, 100))
            screen.blit(stage_text, (230, 50))
            
            time_mins = int(game_time // 60)
            time_secs = int(game_time % 60)
            time_text = font.render(f"Time: {time_mins:02d}:{time_secs:02d}", True, WHITE)
            screen.blit(time_text, (380, 50))
            
            score_text = font.render(f"Score: {score}", True, WHITE)
            screen.blit(score_text, (20, 120))
            
            # --- Draw Spell Guide ---
            guide_y = 160
            guide_x = 20
            guide_title = font.render("- 악상 기호 가이드 -", True, (200, 200, 200))
            screen.blit(guide_title, (guide_x, guide_y))
            guide_y += 35
            
            guide_items = [
                ("0_TrebleClef", "높은음: 공격 모드 (기본)"),
                ("1_BassClef", "낮은음: 서포트 모드 (회복↑, 딜↓)"),
                ("2_Sharp", "샵(#): [전체] 데미지 1.7배"),
                ("3_Flat", "플랫(b): [전체] 탄 개수 2배"),
                ("5_Accent", "악센트(>): [다음 공격] 확산(AoE) 및 크기 증가"),
                ("4_QuarterNote", "4분음표: 강한 공격 1발 (악상 30)"),
                ("6_EighthNote", "8분음표: 약한 공격 2발 (악상 15)"),
                ("7_QuarterRest", "4분쉼표: 악상 회복 (기본30 / 서포트50)"),
                ("8_HalfRest", "2분쉼표: 체력 회복 (기본20 / 서포트40)")
            ]
            
            for g_label, g_desc in guide_items:
                if g_label in spell_icons_queue:
                    icon = spell_icons_queue[g_label]
                    screen.blit(icon, (guide_x, guide_y - 5))
                desc_surf = ui_font.render(g_desc, True, WHITE)
                screen.blit(desc_surf, (guide_x + 40, guide_y))
                guide_y += 40
            # -------------------------
            
            info_text = font.render("좌클릭 길게: 그리기 | 스페이스: 저장 | 우클릭: 사용 | LSHIFT: 스킬", True, WHITE)
            screen.blit(info_text, (WIDTH // 2 - info_text.get_width() // 2, 20))
            
            # Spell Feedback Floating UI
            if spell_feedback_timer > 0:
                floating_y = HEIGHT // 2 - 120
                floating_x = WIDTH // 2
                
                if spell_feedback_label == "FAIL":
                    feedback_surf = font.render("실패 / 악상 부족!", True, spell_feedback_color)
                    screen.blit(feedback_surf, (floating_x - feedback_surf.get_width() // 2, floating_y))
                # Text-based alerts without images (e.g. Clef changes, Mana recovery)
                elif "모드" in spell_feedback_label or "회복" in spell_feedback_label:
                    feedback_surf = font.render(spell_feedback_label, True, spell_feedback_color)
                    screen.blit(feedback_surf, (floating_x - feedback_surf.get_width() // 2, floating_y))
                else:
                    # Draw a nice backing circle
                    pygame.draw.circle(screen, (30, 30, 30), (floating_x, floating_y), 45)
                    pygame.draw.circle(screen, spell_feedback_color, (floating_x, floating_y), 45, 3)
                    
                    if spell_feedback_label in spell_icons_floating:
                        icon = spell_icons_floating[spell_feedback_label]
                        icon_rect = icon.get_rect(center=(floating_x, floating_y))
                        screen.blit(icon, icon_rect)
                        
                    # Add small text underneath
                    saved_text = ui_font.render("저장됨!", True, GREEN)
                    screen.blit(saved_text, (floating_x - saved_text.get_width() // 2, floating_y + 50))

        except:
            pass # Handle potential font issues silently

    # Level Up Overlay
    if current_state == STATE_LEVEL_UP:
        # Darken screen
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))
        
        title_text = title_font.render("레벨 업! 능력 선택", True, (255, 215, 0))
        screen.blit(title_text, (WIDTH//2 - title_text.get_width()//2, 100))
        
        for i, rect in enumerate(upgrade_rects):
            pygame.draw.rect(screen, (60, 60, 80), rect, border_radius=10)
            pygame.draw.rect(screen, WHITE, rect, 3, border_radius=10)
            
            upg = current_upgrades_offered[i]
            # Draw Name
            name_text = font.render(upg["name"], True, WHITE)
            # Center name horizontally
            screen.blit(name_text, (rect.centerx - name_text.get_width()//2, rect.top + 30))
            
            # Draw Desc
            desc_text = ui_font.render(upg["desc"], True, (200, 200, 200))
            screen.blit(desc_text, (rect.centerx - desc_text.get_width()//2, rect.top + 100))
            
            # Draw Click prompt
            click_text = ui_font.render("클릭하여 선택", True, (150, 150, 150))
            screen.blit(click_text, (rect.centerx - click_text.get_width()//2, rect.bottom - 40))

    # Game Over Overlay
    elif current_state == STATE_GAME_OVER:
        # Darken screen
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((150, 0, 0, 200))
        screen.blit(overlay, (0, 0))
        
        go_text = title_font.render("게임 오버", True, WHITE)
        screen.blit(go_text, (WIDTH//2 - go_text.get_width()//2, HEIGHT//2 - 100))
        
        final_score_text = font.render(f"최종 점수: {score}", True, (255, 215, 0))
        screen.blit(final_score_text, (WIDTH//2 - final_score_text.get_width()//2, HEIGHT//2))

    pygame.display.flip()

pygame.quit()
sys.exit()
