import pygame
import sys
import os
import json

pygame.init()

# Constants
WIDTH, HEIGHT = 1280, 720
MAP_WIDTH, MAP_HEIGHT = 4000, 4000
GRID_SIZE = 64
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
GRID_COLOR = (50, 50, 50)
UI_BG = (30, 30, 40)
HIGHLIGHT = (200, 200, 0)

# Paths
def get_resource_path(relative_path):
    base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

ASSETS_DIR = get_resource_path("assets")
MAP_DIR = os.path.join(ASSETS_DIR, "map")
GRASS_DIR = os.path.join(MAP_DIR, "Grass")
TREE_DIR = os.path.join(MAP_DIR, "tree")
SAVE_FILE = os.path.join(MAP_DIR, "map_data.json")

# Screen Setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Map Editor")
clock = pygame.time.Clock()

font = pygame.font.SysFont("malgungothic", 20)
if not pygame.font.get_init() or not font:
    font = pygame.font.Font(None, 24)

# Load Images
grass_images = {}
if os.path.exists(GRASS_DIR):
    for f in os.listdir(GRASS_DIR):
        if f.endswith(".png"):
            try:
                img = pygame.image.load(os.path.join(GRASS_DIR, f)).convert_alpha()
                # Scale to grid if necessary, though raw size is preferred. Let's force grid size for editor ease.
                img = pygame.transform.scale(img, (GRID_SIZE, GRID_SIZE))
                grass_images[f] = img
            except Exception as e:
                print(f"Error loading {f}: {e}")

tree_images = {}
if os.path.exists(TREE_DIR):
    for f in sorted(os.listdir(TREE_DIR)): # Sort to display consistently
        if f.endswith(".png"):
            try:
                img = pygame.image.load(os.path.join(TREE_DIR, f)).convert_alpha()
                # Trees might be larger than 64x64, keep original aspect ratio if possible, but let's keep original size
                # For UI display we will scale them down.
                tree_images[f] = img
            except Exception as e:
                print(f"Error loading {f}: {e}")

if not grass_images:
    print("Warning: No grass images found.")
if not tree_images:
    print("Warning: No tree images found.")

grass_keys = list(grass_images.keys())
tree_keys = list(tree_images.keys())

# Map Data: dict mapping (x, y) grid coordinates to filename
# Coordinates are grid indices (0 to MAP_WIDTH//GRID_SIZE)
map_data = {
    "grass": {}, # (gx, gy) -> filename
    "tree": {}   # (gx, gy) -> filename
}

# Editor State
cam_x, cam_y = 0, 0
current_layer = "grass" # "grass" or "tree"
show_grid = True

def load_map():
    global map_data
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r") as f:
                data = json.load(f)
                # Convert string keys back to tuples
                map_data["grass"] = {eval(k): v for k, v in data.get("grass", {}).items()}
                map_data["tree"] = {eval(k): v for k, v in data.get("tree", {}).items()}
            print("Map loaded successfully.")
        except Exception as e:
            print(f"Failed to load map: {e}")

def save_map():
    try:
        data = {
            "grass": {str(k): v for k, v in map_data["grass"].items()},
            "tree": {str(k): v for k, v in map_data["tree"].items()}
        }
        with open(SAVE_FILE, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Map saved to {SAVE_FILE}")
    except Exception as e:
        print(f"Failed to save map: {e}")

load_map()

# Main Loop
running = True
mouse_down = False
right_mouse_down = False

ui_height = 100
view_height = HEIGHT - ui_height

while running:
    dt = clock.tick(FPS) / 1000.0
    
    # Input Handling
    keys = pygame.key.get_pressed()
    cam_speed = 500 * dt
    if keys[pygame.K_w]: cam_y -= cam_speed
    if keys[pygame.K_s]: cam_y += cam_speed
    if keys[pygame.K_a]: cam_x -= cam_speed
    if keys[pygame.K_d]: cam_x += cam_speed
    
    # Clamp camera
    cam_x = max(0, min(cam_x, MAP_WIDTH - WIDTH))
    cam_y = max(0, min(cam_y, MAP_HEIGHT - view_height))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_g:
                show_grid = not show_grid
            elif event.key == pygame.K_s and (keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]):
                save_map()
            elif event.key == pygame.K_l and (keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]):
                load_map()
            elif event.key == pygame.K_1:
                current_layer = "grass"
                current_brush_idx = 0
            elif event.key == pygame.K_2:
                current_layer = "tree"
                current_brush_idx = 0
                
        elif event.type == pygame.MOUSEWHEEL:
            # Scroll to change brush
            if current_layer == "grass" and grass_keys:
                current_brush_idx = (current_brush_idx - event.y) % len(grass_keys)
            elif current_layer == "tree" and tree_keys:
                current_brush_idx = (current_brush_idx - event.y) % len(tree_keys)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: mouse_down = True
            elif event.button == 3: right_mouse_down = True
            
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1: mouse_down = False
            elif event.button == 3: right_mouse_down = False

    mouse_x, mouse_y = pygame.mouse.get_pos()
    
    # Paint logic
    if mouse_y < view_height:
        # Get grid coordinate
        world_x = mouse_x + cam_x
        world_y = mouse_y + cam_y
        gx = int(world_x // GRID_SIZE)
        gy = int(world_y // GRID_SIZE)
        
        if 0 <= gx < MAP_WIDTH // GRID_SIZE and 0 <= gy < MAP_HEIGHT // GRID_SIZE:
            if mouse_down:
                if current_layer == "grass" and grass_keys:
                    map_data["grass"][(gx, gy)] = grass_keys[current_brush_idx]
                elif current_layer == "tree" and tree_keys:
                    map_data["tree"][(gx, gy)] = tree_keys[current_brush_idx]
            elif right_mouse_down:
                if (gx, gy) in map_data[current_layer]:
                    del map_data[current_layer][(gx, gy)]

    # Drawing
    screen.fill((20, 20, 20))
    
    # 1. Draw Grass Layer
    for (gx, gy), filename in map_data["grass"].items():
        if filename in grass_images:
            px = gx * GRID_SIZE - cam_x
            py = gy * GRID_SIZE - cam_y
            if -GRID_SIZE <= px <= WIDTH and -GRID_SIZE <= py <= view_height:
                screen.blit(grass_images[filename], (px, py))

    # 2. Draw Grid (Optional)
    if show_grid:
        start_x = int(cam_x) % GRID_SIZE
        start_y = int(cam_y) % GRID_SIZE
        
        for x in range(-start_x, WIDTH, GRID_SIZE):
            pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, view_height))
        for y in range(-start_y, view_height, GRID_SIZE):
            pygame.draw.line(screen, GRID_COLOR, (0, y), (WIDTH, y))

    # 3. Draw Tree Layer
    for (gx, gy), filename in map_data["tree"].items():
        if filename in tree_images:
            img = tree_images[filename]
            # Center tree on grid cell bottom
            px = gx * GRID_SIZE + (GRID_SIZE // 2) - (img.get_width() // 2) - cam_x
            py = gy * GRID_SIZE + GRID_SIZE - img.get_height() - cam_y
            
            # Simple broad-phase culling
            if -img.get_width() <= px <= WIDTH and -img.get_height() <= py <= view_height:
                screen.blit(img, (px, py))

    # Highlight current cell
    if mouse_y < view_height:
        gx = int((mouse_x + cam_x) // GRID_SIZE)
        gy = int((mouse_y + cam_y) // GRID_SIZE)
        px = gx * GRID_SIZE - cam_x
        py = gy * GRID_SIZE - cam_y
        pygame.draw.rect(screen, HIGHLIGHT, (px, py, GRID_SIZE, GRID_SIZE), 2)

    # --- UI Drawing ---
    ui_rect = pygame.Rect(0, view_height, WIDTH, ui_height)
    pygame.draw.rect(screen, UI_BG, ui_rect)
    pygame.draw.line(screen, WHITE, (0, view_height), (WIDTH, view_height), 2)

    info_text = f"Pos: ({int(cam_x)}, {int(cam_y)}) | Layer(1/2): {current_layer.upper()} | Grid(G): {'ON' if show_grid else 'OFF'} | CTRL+S: Save | CTRL+L: Load | Scroll: Change Brush"
    txt_surf = font.render(info_text, True, WHITE)
    screen.blit(txt_surf, (10, view_height + 10))

    # Draw current brush
    brush_y = view_height + 40
    brush_label = font.render("Brush:", True, WHITE)
    screen.blit(brush_label, (10, brush_y + 10))
    
    if current_layer == "grass" and grass_keys:
        curr_name = grass_keys[current_brush_idx]
        img = grass_images[curr_name]
        screen.blit(img, (80, brush_y))
        name_surf = font.render(curr_name, True, (200, 200, 200))
        screen.blit(name_surf, (150, brush_y + 15))
        
    elif current_layer == "tree" and tree_keys:
        curr_name = tree_keys[current_brush_idx]
        img = tree_images[curr_name]
        # scale down if too big for UI
        if img.get_height() > 50:
            scale = 50 / img.get_height()
            ui_img = pygame.transform.smoothscale(img, (int(img.get_width() * scale), 50))
        else:
            ui_img = img
        screen.blit(ui_img, (80, brush_y))
        name_surf = font.render(curr_name, True, (200, 200, 200))
        screen.blit(name_surf, (150 + ui_img.get_width(), brush_y + 15))

    pygame.display.flip()

pygame.quit()
sys.exit()
