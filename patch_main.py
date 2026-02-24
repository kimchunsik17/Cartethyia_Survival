import sys
import re

with open("c:/Users/dydeo/OneDrive/Desktop/띳띠 서바이벌/main.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Imports and Argparse
imports_tgt = "import json # Added import json"
imports_rep = """import json # Added import json
import argparse

parser = argparse.ArgumentParser(description="띳띠 서바이벌")
parser.add_argument('--skip-title', action='store_true', help='Skip the title screen')
args = parser.parse_args()"""
code = code.replace(imports_tgt, imports_rep)

# 2. Pygame Init & Mixer
init_tgt = """# Initialize Pygame
pygame.init()"""
init_rep = """# Initialize Pygame
pygame.init()
pygame.mixer.init()"""
code = code.replace(init_tgt, init_rep)

# 3. Constants and Settings Variables
const_tgt = """# Constants
WIDTH, HEIGHT = 1600, 900
FPS = 60
MAP_WIDTH, MAP_HEIGHT = 4000, 4000"""

const_rep = """# Constants & Settings
RESOLUTIONS = [(1280, 720), (1600, 900), (1920, 1080)]
current_resolution_idx = 1
WIDTH, HEIGHT = RESOLUTIONS[current_resolution_idx]
FPS = 60
MAP_WIDTH, MAP_HEIGHT = 4000, 4000
is_fullscreen = False

game_volume = 1.0
LANGUAGES = ["한국어", "English", "日本語"]
current_language_idx = 0"""
code = code.replace(const_tgt, const_rep)


# 4. Global states
state_tgt = """# Game States
STATE_PLAYING = 0
STATE_LEVEL_UP = 1
STATE_GAME_OVER = 2
current_state = STATE_PLAYING"""
state_rep = """# Game States
STATE_TITLE = -1
STATE_SETTINGS = -2
STATE_PLAYING = 0
STATE_LEVEL_UP = 1
STATE_GAME_OVER = 2
current_state = STATE_PLAYING if args.skip_title else STATE_TITLE"""
code = code.replace(state_tgt, state_rep)

# 5. Load Logo in Asset Loading
asset_tgt = """if not enemy_data:
    surf = pygame.Surface((50, 50))
    surf.fill(RED)
    enemy_data.append({'image': surf, 'level': 1})"""
asset_rep = """if not enemy_data:
    surf = pygame.Surface((50, 50))
    surf.fill(RED)
    enemy_data.append({'image': surf, 'level': 1})

# Title Logo
logo_img = load_image(get_resource_path("assets/dditi_survival_logo.webp"))
"""
code = code.replace(asset_tgt, asset_rep)

# 6. Make screen global in loop context, handle Title/Settings events
event_tgt = """    if current_state == STATE_PLAYING:
        dt = raw_dt
        game_time += dt"""
        
event_rep = """    if current_state == STATE_TITLE:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()
                    # Check button clicks (hardcoded rects based on draw logic below)
                    start_rect = pygame.Rect(WIDTH//2 - 100, HEIGHT//2 + 50, 200, 50)
                    settings_rect = pygame.Rect(WIDTH//2 - 100, HEIGHT//2 + 120, 200, 50)
                    quit_rect = pygame.Rect(WIDTH//2 - 100, HEIGHT//2 + 190, 200, 50)
                    
                    if start_rect.collidepoint(mouse_pos):
                        current_state = STATE_PLAYING
                    elif settings_rect.collidepoint(mouse_pos):
                        current_state = STATE_SETTINGS
                    elif quit_rect.collidepoint(mouse_pos):
                        running = False

    elif current_state == STATE_SETTINGS:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()
                    
                    # Buttons
                    res_prev = pygame.Rect(WIDTH//2 + 30, HEIGHT//2 - 150, 40, 40)
                    res_next = pygame.Rect(WIDTH//2 + 230, HEIGHT//2 - 150, 40, 40)
                    fullscreen_btn = pygame.Rect(WIDTH//2 + 50, HEIGHT//2 - 80, 200, 40)
                    vol_down = pygame.Rect(WIDTH//2 + 30, HEIGHT//2 - 10, 40, 40)
                    vol_up = pygame.Rect(WIDTH//2 + 230, HEIGHT//2 - 10, 40, 40)
                    lang_prev = pygame.Rect(WIDTH//2 + 30, HEIGHT//2 + 60, 40, 40)
                    lang_next = pygame.Rect(WIDTH//2 + 230, HEIGHT//2 + 60, 40, 40)
                    back_btn = pygame.Rect(WIDTH//2 - 100, HEIGHT//2 + 250, 200, 50)
                    
                    if res_prev.collidepoint(mouse_pos):
                        current_resolution_idx = (current_resolution_idx - 1) % len(RESOLUTIONS)
                        WIDTH, HEIGHT = RESOLUTIONS[current_resolution_idx]
                        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN if is_fullscreen else 0)
                    elif res_next.collidepoint(mouse_pos):
                        current_resolution_idx = (current_resolution_idx + 1) % len(RESOLUTIONS)
                        WIDTH, HEIGHT = RESOLUTIONS[current_resolution_idx]
                        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN if is_fullscreen else 0)
                    elif fullscreen_btn.collidepoint(mouse_pos):
                        is_fullscreen = not is_fullscreen
                        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN if is_fullscreen else 0)
                    elif vol_down.collidepoint(mouse_pos):
                        game_volume = max(0.0, game_volume - 0.1)
                        # Example: pygame.mixer.music.set_volume(game_volume)
                    elif vol_up.collidepoint(mouse_pos):
                        game_volume = min(1.0, game_volume + 0.1)
                    elif lang_prev.collidepoint(mouse_pos):
                        current_language_idx = (current_language_idx - 1) % len(LANGUAGES)
                    elif lang_next.collidepoint(mouse_pos):
                        current_language_idx = (current_language_idx + 1) % len(LANGUAGES)
                    elif back_btn.collidepoint(mouse_pos):
                        current_state = STATE_TITLE

    elif current_state == STATE_PLAYING:
        dt = raw_dt
        game_time += dt"""
code = code.replace(event_tgt, event_rep)

# 7. Rendering Title & Settings Screens
draw_tgt = """    # Level Up Overlay
    if current_state == STATE_LEVEL_UP:"""
    
draw_rep = """    if current_state == STATE_TITLE:
        screen.fill((20, 20, 30))
        
        # Draw Logo
        if logo_img:
            logo_rect = logo_img.get_rect(center=(WIDTH//2, HEIGHT//2 - 150))
            screen.blit(logo_img, logo_rect)
            
        # Draw Buttons
        buttons = [("게임 시작", HEIGHT//2 + 50), ("설정", HEIGHT//2 + 120), ("종료", HEIGHT//2 + 190)]
        mouse_pos = pygame.mouse.get_pos()
        for label, y in buttons:
            rect = pygame.Rect(WIDTH//2 - 100, y, 200, 50)
            color = (80, 80, 100) if rect.collidepoint(mouse_pos) else (50, 50, 70)
            pygame.draw.rect(screen, color, rect, border_radius=10)
            pygame.draw.rect(screen, WHITE, rect, 2, border_radius=10)
            
            text_surf = font.render(label, True, WHITE)
            screen.blit(text_surf, (rect.centerx - text_surf.get_width()//2, rect.centery - text_surf.get_height()//2))

    elif current_state == STATE_SETTINGS:
        screen.fill((30, 30, 40))
        title_surf = title_font.render("설정 (Settings)", True, WHITE)
        screen.blit(title_surf, (WIDTH//2 - title_surf.get_width()//2, 80))
        
        options = [
            ("해상도", HEIGHT//2 - 150),
            ("창 모드", HEIGHT//2 - 80),
            ("게임 볼륨", HEIGHT//2 - 10),
            ("언어", HEIGHT//2 + 60),
            ("조작키", HEIGHT//2 + 130)
        ]
        
        for label, y in options:
            label_surf = font.render(label, True, (200, 200, 200))
            screen.blit(label_surf, (WIDTH//2 - 250, y))
            
        mouse_pos = pygame.mouse.get_pos()
            
        # Helper to draw interactive value
        def draw_val(y, val_text, has_arrows=True):
            if has_arrows:
                prev_rect = pygame.Rect(WIDTH//2 + 30, y, 40, 40)
                next_rect = pygame.Rect(WIDTH//2 + 230, y, 40, 40)
                
                pygame.draw.rect(screen, (80,80,100) if prev_rect.collidepoint(mouse_pos) else (50,50,70), prev_rect, border_radius=5)
                pygame.draw.rect(screen, (80,80,100) if next_rect.collidepoint(mouse_pos) else (50,50,70), next_rect, border_radius=5)
                
                lt = font.render("<", True, WHITE)
                rt = font.render(">", True, WHITE)
                screen.blit(lt, (prev_rect.centerx - lt.get_width()//2, prev_rect.centery - lt.get_height()//2))
                screen.blit(rt, (next_rect.centerx - rt.get_width()//2, next_rect.centery - rt.get_height()//2))
                
                vt = font.render(val_text, True, (255, 255, 100))
                screen.blit(vt, (WIDTH//2 + 150 - vt.get_width()//2, y))
            else:
                rect = pygame.Rect(WIDTH//2 + 50, y, 200, 40)
                pygame.draw.rect(screen, (80,80,100) if rect.collidepoint(mouse_pos) else (50,50,70), rect, border_radius=5)
                vt = font.render(val_text, True, (255, 255, 100))
                screen.blit(vt, (rect.centerx - vt.get_width()//2, rect.centery - vt.get_height()//2))
                
        # Draw dynamic values
        draw_val(HEIGHT//2 - 150, f"{WIDTH}x{HEIGHT}")
        draw_val(HEIGHT//2 - 80, "전체화면" if is_fullscreen else "창 모드", has_arrows=False)
        draw_val(HEIGHT//2 - 10, f"{int(game_volume * 100)}%")
        draw_val(HEIGHT//2 + 60, LANGUAGES[current_language_idx])
        
        # Keybinds (static display preview)
        kb_text = ui_font.render("W/A/S/D: 이동 | 스페이스: 저장 | 좌클릭: 그리기", True, (150, 150, 150))
        screen.blit(kb_text, (WIDTH//2 + 30, HEIGHT//2 + 135))
        
        # Back Button
        back_rect = pygame.Rect(WIDTH//2 - 100, HEIGHT//2 + 250, 200, 50)
        color = (80, 80, 100) if back_rect.collidepoint(mouse_pos) else (50, 50, 70)
        pygame.draw.rect(screen, color, back_rect, border_radius=10)
        back_surf = font.render("돌아가기", True, WHITE)
        screen.blit(back_surf, (back_rect.centerx - back_surf.get_width()//2, back_rect.centery - back_surf.get_height()//2))


    # Level Up Overlay
    if current_state == STATE_LEVEL_UP:"""
code = code.replace(draw_tgt, draw_rep)

with open("c:/Users/dydeo/OneDrive/Desktop/띳띠 서바이벌/main.py", "w", encoding="utf-8") as f:
    f.write(code)
print("Patch script complete.")
