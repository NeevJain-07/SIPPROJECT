"""
take_screenshots.py — Robust High-Resolution Browser Screenshot Capture
Captures real, live screenshots of both the Web Studio and the Streamlit App.
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

def get_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1600,1050")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--force-device-scale-factor=1.5")
    driver = webdriver.Chrome(options=options)
    driver.set_window_size(1600, 1050)
    return driver

def capture_web_studio(driver):
    print("Navigating to Web Studio at http://127.0.0.1:5000 ...")
    driver.get("http://127.0.0.1:5000")
    time.sleep(3)

    # 1. Initial Web Studio Dashboard
    driver.save_screenshot(str(SCREENSHOTS_DIR / "01_web_studio_dashboard.png"))
    print("Saved 01_web_studio_dashboard.png")

    try:
        # Click the "Scan & Dewarp Document" button for the default selected preset (IMG_3712.jpeg)
        btn_process = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "btnProcess"))
        )
        print("Clicking 'Scan & Dewarp Document' for IMG_3712.jpeg...")
        btn_process.click()

        # Wait for processing to finish (spinner appears and then disappears, and metrics card is displayed)
        time.sleep(2)
        WebDriverWait(driver, 40).until(
            lambda d: d.find_element(By.ID, "spinnerOverlay").value_of_css_property("display") == "none"
        )
        time.sleep(2)
        driver.save_screenshot(str(SCREENSHOTS_DIR / "02_web_studio_comparison_slider.png"))
        print("Saved 02_web_studio_comparison_slider.png")

        # 3. Switch to "100% Scanned" view (Clean Color)
        scanned_tab = driver.find_element(By.CSS_SELECTOR, "button[data-view='scanned']")
        scanned_tab.click()
        time.sleep(1.5)
        driver.save_screenshot(str(SCREENSHOTS_DIR / "03_web_studio_scanned_clean_color.png"))
        print("Saved 03_web_studio_scanned_clean_color.png")

        # 4. Switch mode to "B&W Scan"
        bw_mode_btn = driver.find_element(By.CSS_SELECTOR, "button[data-mode='clean_bw']")
        bw_mode_btn.click()
        time.sleep(1.5)
        driver.save_screenshot(str(SCREENSHOTS_DIR / "04_web_studio_scanned_clean_bw.png"))
        print("Saved 04_web_studio_scanned_clean_bw.png")

        # 5. Switch to "Side-by-Side" view
        side_tab = driver.find_element(By.CSS_SELECTOR, "button[data-view='sidebyside']")
        side_tab.click()
        time.sleep(1.5)
        driver.save_screenshot(str(SCREENSHOTS_DIR / "05_web_studio_side_by_side.png"))
        print("Saved 05_web_studio_side_by_side.png")

        # 6. Switch to "Page Quad Overlay" view
        overlay_tab = driver.find_element(By.CSS_SELECTOR, "button[data-view='overlay']")
        overlay_tab.click()
        time.sleep(1.5)
        driver.save_screenshot(str(SCREENSHOTS_DIR / "06_web_studio_quad_overlay.png"))
        print("Saved 06_web_studio_quad_overlay.png")

        # 7. Select preset 2 (IMG_3715.jpeg - Ruled Notebook on Bedsheet)
        sample_items = driver.find_elements(By.CSS_SELECTOR, ".sample-item")
        if len(sample_items) > 1:
            print("Selecting preset 2 (IMG_3715.jpeg)...")
            sample_items[1].click()
            time.sleep(1)
            # Switch scan mode back to Clean Color
            color_btn = driver.find_element(By.CSS_SELECTOR, "button[data-mode='clean_color']")
            color_btn.click()
            # Click process
            btn_process = driver.find_element(By.ID, "btnProcess")
            btn_process.click()
            time.sleep(2)
            WebDriverWait(driver, 40).until(
                lambda d: d.find_element(By.ID, "spinnerOverlay").value_of_css_property("display") == "none"
            )
            time.sleep(2)
            # Switch to comparison slider
            split_tab = driver.find_element(By.CSS_SELECTOR, "button[data-view='split']")
            split_tab.click()
            time.sleep(1.5)
            driver.save_screenshot(str(SCREENSHOTS_DIR / "07_web_studio_ruled_notebook_slider.png"))
            print("Saved 07_web_studio_ruled_notebook_slider.png")

        # 8. Select preset 3 (IMG_3716.jpeg - Notes on High-Contrast Floral Fabric)
        if len(sample_items) > 2:
            print("Selecting preset 3 (IMG_3716.jpeg)...")
            sample_items[2].click()
            time.sleep(1)
            btn_process = driver.find_element(By.ID, "btnProcess")
            btn_process.click()
            time.sleep(2)
            WebDriverWait(driver, 40).until(
                lambda d: d.find_element(By.ID, "spinnerOverlay").value_of_css_property("display") == "none"
            )
            time.sleep(2)
            driver.save_screenshot(str(SCREENSHOTS_DIR / "08_web_studio_floral_bedsheet_slider.png"))
            print("Saved 08_web_studio_floral_bedsheet_slider.png")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error capturing Web Studio features: {e}")

def capture_streamlit(driver):
    print("\nNavigating to Streamlit app at http://127.0.0.1:8501 ...")
    driver.get("http://127.0.0.1:8501")
    
    # Wait for Streamlit to hydrate and render radio buttons
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-testid='stRadio']"))
    )
    time.sleep(3)

    # 9. Streamlit Main Page
    driver.save_screenshot(str(SCREENSHOTS_DIR / "09_streamlit_upload_process.png"))
    print("Saved 09_streamlit_upload_process.png")

    # Helper to select radio option
    def click_radio_option(keyword):
        labels = driver.find_elements(By.CSS_SELECTOR, "div[data-testid='stRadio'] label")
        for lbl in labels:
            if keyword in lbl.text:
                lbl.click()
                return True
        return False

    # 10. Quality Report
    print("Switching to 'Quality Report'...")
    if click_radio_option("Quality Report"):
        time.sleep(4)
        driver.save_screenshot(str(SCREENSHOTS_DIR / "10_streamlit_quality_report.png"))
        print("Saved 10_streamlit_quality_report.png")

    # 11. Debug View
    print("Switching to 'Debug View'...")
    if click_radio_option("Debug View"):
        time.sleep(4)
        driver.save_screenshot(str(SCREENSHOTS_DIR / "11_streamlit_debug_view.png"))
        print("Saved 11_streamlit_debug_view.png")

    # 12. Batch Evaluate
    print("Switching to 'Batch Evaluate'...")
    if click_radio_option("Batch Evaluate"):
        time.sleep(4)
        driver.save_screenshot(str(SCREENSHOTS_DIR / "12_streamlit_batch_evaluate.png"))
        print("Saved 12_streamlit_batch_evaluate.png")

    # 13. About
    print("Switching to 'About'...")
    if click_radio_option("About"):
        time.sleep(3)
        driver.save_screenshot(str(SCREENSHOTS_DIR / "13_streamlit_about_architecture.png"))
        print("Saved 13_streamlit_about_architecture.png")

def main():
    driver = get_driver()
    try:
        capture_web_studio(driver)
        capture_streamlit(driver)
    finally:
        driver.quit()
    print("\nAll real screenshots captured successfully into docs/screenshots/!")

if __name__ == "__main__":
    main()
