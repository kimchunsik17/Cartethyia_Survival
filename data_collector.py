import pygame
import os
import sys
import uuid
import numpy as np

# --- Configuration ---
WINDOW_SIZE = 500
BG_COLOR = (0, 0, 0)
DRAW_COLOR = (255, 255, 255)
BRUSH_SIZE = 15
SAVE_DIR = "assets/spell_data"

# 사용할 악상 기호 (스펠) 클래스들
# 필요시 변경 가능합니다.
LABELS = [
    "0_TrebleClef",   # 높은음자리표
    "1_BassClef",     # 낮은음자리표
    "2_Sharp",        # 샵 (#)
    "3_Flat",         # 플랫 (b)
    "4_QuarterNote",  # 4분음표
    "5_Accent"        # 악센트 (>)
]

current_label_idx = 0

pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("스펠 데이터 수집기 (저장: S, 지우기: C, 라벨 변경: 좌/우 방향키)")
clock = pygame.time.Clock()

font = pygame.font.SysFont("malgungothic", 24)

# Create save directories
for label in LABELS:
    os.makedirs(os.path.join(SAVE_DIR, label), exist_ok=True)

drawing = False
last_pos = None
canvas = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
canvas.fill(BG_COLOR)

def pad_and_resize(surface, target_size=64):
    """
    그려진 이미지의 바운딩 박스를 찾아 자른 후,
    정사각형 비율을 유지하며 여백을 추가해 64x64 사이즈로 변환합니다.
    """
    # Pygame 서피스를 Numpy 배열로 변환
    arr = pygame.surfarray.pixels_red(surface)
    # 그려진 부분이 있는 좌표 찾기 (흰색 픽셀)
    coords = np.argwhere(arr > 0)
    
    if coords.size == 0:
        return None # 아무것도 그려지지 않음

    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    
    # 여백 추가
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(WINDOW_SIZE, x_max + padding)
    y_max = min(WINDOW_SIZE, y_max + padding)
    
    # 바운딩 박스 기준으로 크롭된 서피스 생성
    rect = pygame.Rect(x_min, y_min, x_max - x_min, y_max - y_min)
    cropped = surface.subsurface(rect).copy()
    
    # 정사각형으로 만들기 위한 캔버스 생성
    max_side = max(rect.width, rect.height)
    square_surf = pygame.Surface((max_side, max_side))
    square_surf.fill((0, 0, 0))
    
    # 중앙 정렬하여 붙여넣기
    offset_x = (max_side - rect.width) // 2
    offset_y = (max_side - rect.height) // 2
    square_surf.blit(cropped, (offset_x, offset_y))
    
    # 최종 64x64 사이즈로 크기 조절
    final_surf = pygame.transform.smoothscale(square_surf, (target_size, target_size))
    return final_surf

def get_saved_count(label):
    path = os.path.join(SAVE_DIR, label)
    if not os.path.exists(path): return 0
    return len([f for f in os.listdir(path) if f.endswith('.png')])

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                drawing = True
                last_pos = event.pos
                pygame.draw.circle(canvas, DRAW_COLOR, event.pos, BRUSH_SIZE // 2)
                
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False
                
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                current_pos = event.pos
                # 선을 부드럽게 그리기 위해 두 점 사이를 선으로 연결하고 끝을 둥글게 처리
                pygame.draw.line(canvas, DRAW_COLOR, last_pos, current_pos, BRUSH_SIZE)
                pygame.draw.circle(canvas, DRAW_COLOR, current_pos, BRUSH_SIZE // 2)
                last_pos = current_pos
                
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                # 초기화 (지우기)
                canvas.fill(BG_COLOR)
                
            elif event.key == pygame.K_s:
                # 저장
                processed_surf = pad_and_resize(canvas)
                if processed_surf:
                    current_label = LABELS[current_label_idx]
                    filename = f"{uuid.uuid4().hex[:8]}.png"
                    filepath = os.path.join(SAVE_DIR, current_label, filename)
                    pygame.image.save(processed_surf, filepath)
                    print(f"[{current_label}] 저장 완료: {filename}")
                    # 저장 후 자동 초기화
                    canvas.fill(BG_COLOR)
                else:
                    print("그려진 내용이 없습니다!")
                    
            elif event.key == pygame.K_RIGHT:
                current_label_idx = (current_label_idx + 1) % len(LABELS)
                canvas.fill(BG_COLOR)
                
            elif event.key == pygame.K_LEFT:
                current_label_idx = (current_label_idx - 1) % len(LABELS)
                canvas.fill(BG_COLOR)

    screen.fill(BG_COLOR)
    screen.blit(canvas, (0, 0))
    
    # UI Text
    current_label = LABELS[current_label_idx]
    saved_count = get_saved_count(current_label)
    
    try:
        text1 = font.render(f"현재 선택된 스펠: {current_label} ({current_label_idx + 1}/{len(LABELS)})", True, (255, 255, 0))
        text2 = font.render(f"현재 클래스 저장된 개수: {saved_count}개", True, (0, 255, 0))
        text3 = font.render("좌/우 방향키: 스펠 변경, S: 저장, C: 지우기", True, (200, 200, 200))
        
        screen.blit(text1, (10, 10))
        screen.blit(text2, (10, 40))
        screen.blit(text3, (10, WINDOW_SIZE - 40))
    except:
        pass

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
