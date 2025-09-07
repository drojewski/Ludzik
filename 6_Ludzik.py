import os
import shutil
import sys
import time
from datetime import datetime

import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
import pyautogui
from PIL import Image
import tempfile

INPUT_DIR = '.'
INPUT_FILE_NAME = os.environ.get('INPUT_FILE', 'part_4.json')
INPUT_FILE = os.path.join(INPUT_DIR, INPUT_FILE_NAME)
LOG_FILE = f"log_auto_walk_{INPUT_FILE_NAME}.txt"


OUT_DIR = "auto_walk_screenshots"
UP_IMG_DIR = os.path.join(OUT_DIR, "up_images")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(UP_IMG_DIR, exist_ok=True)
PAUSE = 2

def log(message):
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{time_str}] {message}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f_log:
        f_log.write(line + "\n")

def save_remaining_points(waypoints, last_index, filename=INPUT_FILE):
    remaining = waypoints[last_index + 1 :]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(remaining, f, ensure_ascii=False, indent=2)
    log(f"Zapisano {len(remaining)} pozostałych punktów do pliku {filename}")

def go_to_center():
    pyautogui.moveTo(center_x, center_y, duration=0.5)

def full_rotation(pano, step_px=600, degree_per_step=45):
    go_to_center()
    steps = int(360 / degree_per_step)
    for step in range(steps):
        pyautogui.mouseDown()
        pyautogui.moveRel(-step_px, 0, duration=1)
        pyautogui.mouseUp()
        go_to_center()
        take_photo(pano, is_up_image=False)
        pyautogui.mouseDown()
        pyautogui.moveRel(0, 500, duration=0.3)
        pyautogui.mouseUp()
        take_photo(pano, is_up_image=True)
        pyautogui.mouseDown()
        pyautogui.moveRel(0, -500, duration=0.3)
        pyautogui.mouseUp()
        go_to_center()
    go_to_center()

def resize_image(path, size=(640, 640)):
    image = Image.open(path)
    image.thumbnail(size)
    image.save(path)

def take_photo(pano_id, is_up_image=False):
    filename = (
        f"step_{i:03d}_{lat:.6f}_{lng:.6f}_{pano_id}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    if is_up_image:
        save_path = os.path.join(UP_IMG_DIR, filename)
    else:
        save_path = os.path.join(OUT_DIR, filename)
    canvas.screenshot(save_path)
    resize_image(save_path)

def canvas_to_rectangle_with_dimensioning():
    global rect
    rect = driver.execute_script("""
        const c = arguments[0];
        const rect = c.getBoundingClientRect();
        return {left: rect.left, top: rect.top, width: rect.width, height: rect.height};
        """, canvas)

def calculate_x_y_of_canvas_center():
    global center_x, center_y
    center_x = int(rect['left'] + rect['width'] / 2)
    center_y = int(rect['top'] + rect['height'] / 2)
    log(f"Canvas center at: x={center_x}, y={center_y}")

def setup_driver():
    opts = Options()
    # opts.add_argument("--headless")  # jeśli potrzebujesz uruchomić bez GUI
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument('--no-sandbox')
    opts.add_argument('--disable-dev-shm-usage')
    opts.add_argument('--disable-gpu')
    global temp_profile
    temp_profile = tempfile.mkdtemp()
    opts.add_argument(f'--user-data-dir={temp_profile}')
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=opts)

def find_last_processed_index():
    if not os.path.exists(LOG_FILE):
        return -1
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in reversed(lines):
        if "Otwieram punkt" in line:
            try:
                num = int(line.split("Otwieram punkt")[1].split("z")[0].strip())
                return num - 1
            except Exception:
                continue
    return -1

def run_auto_walk(waypoints):
    global canvas, driver, i, lat, lng
    total = len(waypoints)
    for i, point in enumerate(waypoints):
        lat, lng = point["coordinates"]
        pano_id = point.get("pano_id", "[brak pano_id]")
        url = (
            f"https://www.google.com/maps/@?api=1&map_action=pano"
            f"&viewpoint={lat},{lng}"
        )
        driver.get(url)
        log(f"Otwieram punkt {i + 1} z {total}: {lng}, {lat}, pano_id: {pano_id}")
        log(f"Pozostało punktów: {total - (i + 1)}")
        time.sleep(1)
        try:
            btn = driver.find_element(By.XPATH,
                "//button[normalize-space()='Zaakceptuj wszystko' or normalize-space()='I agree']")
            btn.click()
            log("Regulamin zaakceptowany")
            time.sleep(3)
        except Exception:
            pass
        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_all_elements_located((By.TAG_NAME, "canvas"))
            )
            canvases = driver.find_elements(By.TAG_NAME, "canvas")
            log(f"Znaleziono {len(canvases)} elementów <canvas>")
            canvas = canvases[0]
        except TimeoutException:
            log("Nie znaleziono elementu <canvas> w ciągu 20s.")
            #save_remaining_points(waypoints, i-1 if i > 0 else 0)
            log("Czekam 2 minuty przed ponownym uruchomieniem...")
            time.sleep(30)
            driver.quit()
            shutil.rmtree(temp_profile, ignore_errors=True)
            main()
            return
        canvas_to_rectangle_with_dimensioning()
        calculate_x_y_of_canvas_center()
        full_rotation(pano_id)
        log(f"[{i}] zapisano screenshot")

def main():
    global driver
    driver = setup_driver()
    log(f"Używany katalog profilu Chrome: {temp_profile}")
    log(f"Używany plik wejściowy: {INPUT_FILE}")

    last_idx = find_last_processed_index()
    if os.path.exists(INPUT_FILE):
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            waypoints = json.load(f)
        if last_idx >= 0 and last_idx < len(waypoints):
            waypoints = waypoints[last_idx:]
            with open(INPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(waypoints, f, ensure_ascii=False, indent=2)
            log(f"Przycięto plik wejściowy, zaczynając od punktu {last_idx + 1} (pominęto {last_idx} punktów).")

    if not os.path.exists(INPUT_FILE) or os.stat(INPUT_FILE).st_size == 0:
        log(f"Brak punktów do przetworzenia w pliku {INPUT_FILE}. Kończę działanie.")
        driver.quit()
        shutil.rmtree(temp_profile, ignore_errors=True)
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        waypoints = json.load(f)
    run_auto_walk(waypoints)
    driver.quit()
    shutil.rmtree(temp_profile, ignore_errors=True)
    log("Zakończono automatyczną wycieczkę Pegmanem.")

if __name__ == '__main__':
    main()
