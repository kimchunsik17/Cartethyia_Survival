import pygame
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Configuration ---
WINDOW_SIZE = 600
BG_COLOR = (0, 0, 0)
DRAW_COLOR = (255, 255, 255)
BRUSH_SIZE = 15

# SPELL CLASSES
SPELL_LABELS = [
    "0_TrebleClef",
    "1_BassClef",
    "2_Sharp",
    "3_Flat",
    "4_QuarterNote",
    "5_Accent"
]

# --- PyTorch Model Setup ---
class SpellCNN(nn.Module):
    def __init__(self, num_classes=len(SPELL_LABELS)):
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

def get_predictions(surface):
    # Pygame 서피스를 Numpy 배열로 변환
    arr = pygame.surfarray.pixels_red(surface)
    # 그려진 부분이 있는 좌표 찾기 (흰색 픽셀)
    coords = np.argwhere(arr > 0)
    
    if coords.size == 0:
        return None
        
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
    
    img_array = pygame.surfarray.pixels_red(final_surf).astype(np.float32) / 255.0
    img_array = img_array.T 
    
    tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = spell_model(tensor)
        probs = F.softmax(outputs, dim=1)[0]
        
    return probs.cpu().numpy()

pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE + 300, WINDOW_SIZE))
pygame.display.set_caption("스펠 인식률 테스트 도구")
clock = pygame.time.Clock()

try:
    font = pygame.font.SysFont("malgungothic", 20)
    title_font = pygame.font.SysFont("malgungothic", 30)
except:
    font = pygame.font.Font(None, 24)
    title_font = pygame.font.Font(None, 36)

# Init Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
spell_model = SpellCNN(num_classes=len(SPELL_LABELS)).to(device)
model_loaded = False
try:
    spell_model.load_state_dict(torch.load("spell_model.pth", map_location=device))
    spell_model.eval()
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")

canvas = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
canvas.fill(BG_COLOR)

drawing = False
last_pos = None

current_probs = None

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                # Only draw if inside canvas area
                if event.pos[0] < WINDOW_SIZE:
                    drawing = True
                    last_pos = event.pos
                    pygame.draw.circle(canvas, DRAW_COLOR, event.pos, BRUSH_SIZE // 2)
                
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                if drawing:
                    drawing = False
                
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                current_pos = event.pos
                if current_pos[0] < WINDOW_SIZE:
                    pygame.draw.line(canvas, DRAW_COLOR, last_pos, current_pos, BRUSH_SIZE)
                    pygame.draw.circle(canvas, DRAW_COLOR, current_pos, BRUSH_SIZE // 2)
                    last_pos = current_pos
                
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                canvas.fill(BG_COLOR)
                current_probs = None
            elif event.key == pygame.K_SPACE:
                if model_loaded:
                    current_probs = get_predictions(canvas)

    screen.fill((30, 30, 30))
    screen.blit(canvas, (0, 0))
    
    # Draw separator line
    pygame.draw.line(screen, (100, 100, 100), (WINDOW_SIZE, 0), (WINDOW_SIZE, WINDOW_SIZE), 2)
    
    # Draw Results UI on the right panel
    ui_x = WINDOW_SIZE + 20
    
    if not model_loaded:
        err_msg = font.render("spell_model.pth 없음!", True, (255, 0, 0))
        screen.blit(err_msg, (ui_x, 50))
    else:
        title = title_font.render("인식 결과", True, (255, 255, 255))
        screen.blit(title, (ui_x, 20))
        
        info = font.render("좌클릭 유지: 그리기", True, (200, 200, 200))
        clear_info = font.render("스페이스바: 인식 | C: 지우기", True, (200, 200, 200))
        screen.blit(info, (ui_x, 70))
        screen.blit(clear_info, (ui_x, 100))
        
        if current_probs is not None:
            # Sort probabilities descending
            sorted_indices = np.argsort(current_probs)[::-1]
            
            y_offset = 150
            for i, idx in enumerate(sorted_indices):
                label_name = SPELL_LABELS[idx].split('_')[1]
                prob = current_probs[idx] * 100
                
                # Highlight top prediction
                color = (0, 255, 0) if i == 0 and prob >= 50 else (255, 255, 255)
                if i == 0 and prob < 50:
                    color = (255, 100, 100) # Failed to recognize
                
                text = font.render(f"{i+1}. {label_name}: {prob:.1f}%", True, color)
                screen.blit(text, (ui_x, y_offset))
                
                # Draw bar
                bar_width = int(max(0, 200 * current_probs[idx]))
                pygame.draw.rect(screen, (50, 50, 50), (ui_x, y_offset + 30, 200, 10))
                pygame.draw.rect(screen, color, (ui_x, y_offset + 30, bar_width, 10))
                
                y_offset += 60

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
