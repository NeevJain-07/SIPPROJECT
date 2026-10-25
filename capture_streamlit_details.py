"""
capture_streamlit_details.py — Upload an image to Streamlit, process it, and capture full Quality Report and Debug View.
"""
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

SCREENSHOTS_DIR = Path("docs/screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1600,1100")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--force-device-scale-factor=1.5")
    driver = webdriver.Chrome(options=options)
    driver.set_window_size(1600, 1100)

    try:
        print("Navigating to Streamlit at http://127.0.0.1:8501 ...")
        driver.get("http://127.0.0.1:8501")
        
        # Wait for file uploader input
        file_input = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
        )
        sample_path = Path("IMG_3712.jpeg").resolve()
        print(f"Uploading {sample_path} ...")
        file_input.send_keys(str(sample_path))

        # Wait for "Scan & Dewarp Document" button
        time.sleep(3)
        scan_btn = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Scan & Dewarp Document')]"))
        )
        print("Clicking 'Scan & Dewarp Document' in Streamlit...")
        scan_btn.click()

        # Wait for processing to complete
        print("Waiting for Streamlit pipeline to finish...")
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Before / After') or contains(text(), 'Pipeline completed')]"))
        )
        time.sleep(4)
        driver.save_screenshot(str(SCREENSHOTS_DIR / "14_streamlit_processed_before_after.png"))
        print("Saved 14_streamlit_processed_before_after.png")

        # Helper to click radio
        def click_radio(name):
            labels = driver.find_elements(By.CSS_SELECTOR, "div[data-testid='stRadio'] label")
            for l in labels:
                if name in l.text:
                    l.click()
                    return True
            return False

        # Now click Quality Report
        print("Switching to Quality Report...")
        if click_radio("Quality Report"):
            time.sleep(4)
            driver.save_screenshot(str(SCREENSHOTS_DIR / "15_streamlit_quality_report_active.png"))
            print("Saved 15_streamlit_quality_report_active.png")

        # Now click Debug View
        print("Switching to Debug View...")
        if click_radio("Debug View"):
            time.sleep(4)
            driver.save_screenshot(str(SCREENSHOTS_DIR / "16_streamlit_debug_view_stages.png"))
            print("Saved 16_streamlit_debug_view_stages.png")

    finally:
        driver.quit()
    print("Done capturing Streamlit details!")

if __name__ == "__main__":
    main()
