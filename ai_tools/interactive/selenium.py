from .interactive_tool_class import InteractiveTool

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    WebDriverException, NoSuchElementException, NoAlertPresentException, JavascriptException
)
from selenium.webdriver.chrome.options import Options


import threading
import random
import subprocess


import time
import uuid

import io
import np
import cv2
import os
from datetime import datetime

from utils import url_parsing
import requests
import os
from urllib.parse import urlparse

def download_file(url: str, max_size_mb=25) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        return "Blocked: invalid URL scheme."

    headers = {
        "User-Agent": "Mozilla/5.0",
    }

    try:
        with requests.get(url, headers=headers, stream=True, timeout=10) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('Content-Length', 0))
            max_bytes = max_size_mb * 1024 * 1024

            if total_size and total_size > max_bytes:
                return f"Blocked: file too large ({total_size/1024/1024:.2f} MB > {max_size_mb} MB)."

            os.makedirs('./temp', exist_ok=True)
            fname = os.path.basename(parsed.path) or "downloaded_file"
            save_path = os.path.join('./temp', fname)

            size_downloaded = 0
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    size_downloaded += len(chunk)
                    if size_downloaded > max_bytes:
                        f.close()
                        os.remove(save_path)
                        return "Blocked: download exceeded 25MB during stream."
                    f.write(chunk)

            return f"Downloaded to `{save_path}`"

    except requests.RequestException as e:
        return f"Download failed: {str(e)}"


READABILITY_CDN = "https://cdnjs.cloudflare.com/ajax/libs/readability/0.6.0/Readability.js"
download_dir = os.path.abspath('./temp')


class SeleniumInteractiveTool(InteractiveTool):
    """
    Interactive Selenium tool supporting:
        - NAVIGATE <url>
        - CLICK <css_selector>
        - CLICK_LINK <index>               (click previously extracted link by index)
        - EXTRACT_TEXT <css_selector>
        - EXTRACT_LINKS <css_selector>
        - EXTRACT_ATTRIBUTES <css_selector> <attribute>
        - GET_PAGE_SOURCE
        - WAIT <seconds>
        - LIST_SELECTORS                  (all class names and IDs on page)
        - LIST_TAGS                       (distinct HTML tags on page)
        - FIND_SELECTOR_CONTAINING <text> (returns selector that contains text)
        - SCROLL <pixels>                 (scroll vertical offset)
        - SCROLL_TO_BOTTOM
        - SCROLL_TO_ELEMENT <css_selector>
        - TAKE_SCREENSHOT <filename>      (saved to memory, not disk)
        - ALERT_GET_TEXT                  (get alert/confirm/prompt message text)
        - ALERT_ACCEPT                   (accept alert/confirm/prompt)
        - ALERT_DISMISS                  (dismiss alert/confirm)
        - ALERT_SEND_KEYS <text>         (send keys to prompt and accept)
        - EXTRACT_ALL_TEXT               (extract main readable text using Readability.js)
        - EXIT

    You can batch multiple commands in one action by sending a SCRIPT block:
        SCRIPT\n<command1>\n<command2>\n...
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        chrome_opts = Options()
        prefs = {
            "download.default_directory": download_dir,
            "download.prompt_for_download": False, 
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "profile.default_content_setting_values.automatic_downloads": 2,  # BLOCK ALL

        }
        
        self.session_id = str(uuid.uuid4())

        self.debug_dir = os.path.abspath(f'./debug_frames_{self.session_id}')
        os.makedirs(self.debug_dir, exist_ok=True)

        chrome_opts.add_argument('--headless')
        chrome_opts.add_argument('--disable-gpu')
        chrome_opts.add_argument('--window-size=800,600')  # Keep it small
        chrome_opts.add_argument('--disable-extensions')
        chrome_opts.add_argument('--disable-dev-shm-usage')  # Avoid issues with /dev/shm

        chrome_opts.add_arguments('--disable-blink-features=AutomationControlled')
        chrome_opts.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/114.0.0.0 Safari/537.36")
        self.driver = webdriver.Chrome(options=chrome_opts)
        self._input_queue = []
        self._last_links = []
        self._last_selectors = []
        self._last_selector = None
        self._readability_loaded = False

        self.screenshots = []       # List of tuples: (filename, paths)
        self.detailed_logs = []     # List of dicts with step, command, result, timestamp

        self.debug_thread = threading.Thread(target=self._debug_capture_loop, daemon=True)
        self._debug_running = True
        self.debug_thread.start()

    def _debug_capture_loop(self):
        i = 0
        while self._debug_running:
            try:
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                # Include session_id and counter in filename
                filename = os.path.join(self.debug_dir, f"frame_{self.session_id}_{timestamp}_{i:05d}.jpg")
                png = self.driver.get_screenshot_as_png()

                img_np = np.frombuffer(png, dtype=np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                cv2.imwrite(filename, img)
                i += 1
                
            except Exception as e:
                self.log(f"[debug_capture] Error: {e}")

            time.sleep(random.uniform(0.09, 0.11))  # 9–11 FPS


    def log_step(self, command, result):
        self.detailed_logs.append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "step": len(self.detailed_logs) + 1,
            "command": command,
            "result": result
        })

    def send_input(self, input_data):
        """Queue a Selenium command"""
        self.log(f"Queueing: {input_data}")
        self._input_queue.append(input_data)

    def _inject_readability(self):
        """Inject Readability.js from CDN if not already loaded"""
        if self._readability_loaded:
            return
        self.log("Injecting Readability.js from CDN")
        is_loaded = self.driver.execute_script("return typeof Readability !== 'undefined';")
        if not is_loaded:
            inject_script = f"""
                var script = document.createElement('script');
                script.src = '{READABILITY_CDN}';
                document.head.appendChild(script);
            """
            self.driver.execute_script(inject_script)
            time.sleep(3)
        self._readability_loaded = True

    def receive_output(self, input_data: str):
        """
        Accepts one or multiple commands as a single string.
        If starts with SCRIPT, treats following lines as commands.
        Executes all commands immediately in order, returns list of results.
        """
        commands = []
        input_data = input_data.strip()
        if input_data.upper().startswith("SCRIPT"):
            # split all lines after SCRIPT keyword
            lines = input_data.splitlines()
            commands = [line.strip() for line in lines[1:] if line.strip()]
        else:
            # single command
            commands = [input_data]

        results = []
        for cmd in commands:
            parts = cmd.split(maxsplit=2)
            action = parts[0].upper()
            arg = parts[1] if len(parts) > 1 else None
            extra = parts[2] if len(parts) > 2 else None

            try:
                if action == "NAVIGATE":
                    if not url_parsing.is_safe_url(arg):
                        result = f"Blocked navigation to unsafe or internal address: {arg}"
                    else:
                        self.driver.get(arg)
                        self._readability_loaded = False
                        result = f"Navigated to {arg}."

                elif action == "CLICK":
                    try:
                        elem = self.driver.find_element(By.CSS_SELECTOR, arg)
                        href = elem.get_attribute("href")

                        if href:
                            if not url_parsing.is_safe_url(href):
                                return f"[blocked] Unsafe link target: {href}"

                        elem.click()
                        self._readability_loaded = False
                        result = f"Clicked element {arg}."
                    except Exception as e:
                        result = f"[error] Failed to click element: {str(e)}"


                elif action == "CLICK_LINK":
                    try:
                        links = self.driver.find_elements(By.TAG_NAME, "a")
                        index = int(arg.strip())

                        if index < 0 or index >= len(links):
                            return f"[error] Link index {index} is out of range."

                        href = links[index].get_attribute("href")
                        if href and not url_parsing.is_safe_url(href):
                            return f"[blocked] Unsafe link target: {href}"

                        links[index].click()
                        self._readability_loaded = False
                        result = f"Clicked link #{index}."
                    except Exception as e:
                        result = f"[error] Failed to click link: {str(e)}"
                        
                elif action == "DOWNLOAD":
                    if not url_parsing.is_safe_url(arg):
                        result = "Blocked: Unsafe download URL."
                    else:
                        result = download_file(arg)


                elif action == "EXTRACT_TEXT":
                    elems = self.driver.find_elements(By.CSS_SELECTOR, arg)
                    texts = [e.text for e in elems]
                    result = texts

                elif action == "EXTRACT_LINKS":
                    elems = self.driver.find_elements(By.CSS_SELECTOR, arg)
                    links = [e.get_attribute('href') for e in elems if e.get_attribute('href')]
                    self._last_links = links
                    self._last_selector = arg
                    result = links

                elif action == "EXTRACT_ATTRIBUTES":
                    selector, attr = arg, extra
                    elems = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    attrs = [e.get_attribute(attr) for e in elems]
                    result = attrs

                elif action == "GET_PAGE_SOURCE":
                    result = self.driver.page_source

                elif action == "WAIT":
                    seconds = float(arg)
                    time.sleep(seconds)
                    result = f"Waited {seconds} seconds."

                elif action == "LIST_SELECTORS":
                    elems = self.driver.find_elements(By.XPATH, "//*")
                    selectors = set()
                    for e in elems:
                        cls = e.get_attribute("class")
                        if cls:
                            selectors.update(cls.split())
                        id_ = e.get_attribute("id")
                        if id_:
                            selectors.add(id_)
                    self._last_selectors = list(selectors)
                    result = self._last_selectors

                elif action == "LIST_TAGS":
                    elems = self.driver.find_elements(By.XPATH, "//*")
                    tags = list({e.tag_name for e in elems})
                    result = tags

                elif action == "FIND_SELECTOR_CONTAINING":
                    text = arg
                    elems = self.driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
                    selectors = []
                    for e in elems:
                        tag = e.tag_name
                        cid = e.get_attribute("id")
                        classes = e.get_attribute("class").split() if e.get_attribute("class") else []
                        if cid:
                            selectors.append(f"#{cid}")
                        elif classes:
                            selectors.append(f"{tag}." + ".".join(classes))
                        else:
                            selectors.append(tag)
                    result = selectors

                elif action == "SCROLL":
                    pixels = int(arg)
                    self.driver.execute_script(f"window.scrollBy(0, {pixels});")
                    result = f"Scrolled by {pixels} pixels."

                elif action == "SCROLL_TO_BOTTOM":
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    result = "Scrolled to bottom."

                elif action == "SCROLL_TO_ELEMENT":
                    elem = self.driver.find_element(By.CSS_SELECTOR, arg)
                    self.driver.execute_script("arguments[0].scrollIntoView();", elem)
                    result = f"Scrolled to element {arg}."

                elif action == "TAKE_SCREENSHOT":
                    filename = arg
                    png_bytes = self.driver.get_screenshot_as_png()
                    self.screenshots.append((filename, png_bytes))
                    result = f"Screenshot captured and saved in memory as {filename}."

                elif action == "ALERT_GET_TEXT":
                    alert = self.driver.switch_to.alert
                    result = alert.text

                elif action == "ALERT_ACCEPT":
                    alert = self.driver.switch_to.alert
                    alert.accept()
                    result = "Alert accepted."

                elif action == "ALERT_DISMISS":
                    alert = self.driver.switch_to.alert
                    alert.dismiss()
                    result = "Alert dismissed."

                elif action == "ALERT_SEND_KEYS":
                    if arg is None:
                        result = "No text provided for ALERT_SEND_KEYS."
                    else:
                        alert = self.driver.switch_to.alert
                        alert.send_keys(arg)
                        alert.accept()
                        result = f"Sent keys to alert and accepted."

                elif action == "EXTRACT_ALL_TEXT":
                    self._inject_readability()
                    article_text = self.driver.execute_script("""
                        let documentClone = document.cloneNode(true);
                        let reader = new Readability(documentClone);
                        let article = reader.parse();
                        return article ? article.textContent : '';
                    """)
                    result = article_text

                elif action == "EXIT":
                    self.done = True
                    result = "Session terminated by EXIT."

                else:
                    result = f"Unknown command '{action}'."

            except NoSuchElementException:
                result = f"Element not found for selector '{arg}'."
            except NoAlertPresentException:
                result = "No alert present."
            except JavascriptException as e:
                result = f"JavaScript error during execution: {e}."
            except WebDriverException as e:
                result = f"WebDriver error: {e}."

            self.log_step(cmd, result)
            results.append(result)

            if getattr(self, "done", False):
                break

        if len(results) == 1:
            return results[0]
        return results

    def process_response(self, response):
        self.history.append({
            "step": self.step_count,
            "response": response
        })
        return not self.done

    def get_final_output(self):
        if self.history:
            return self.history[-1]["response"]
        return None

    def describe(self) -> dict:
        return {
            "name": "SeleniumLiveBrowser",
            "description": (
                "Headless browser emulator for live web interaction. Supports navigation, clicking, text/link extraction, scrolling, and other browser-like operations.\n\n"
                "For multi-step workflows, use `SCRIPT` followed by newline-separated commands. This allows batching multiple actions into a single tool call.\n"
                "SCRIPT sessions maintain browser state across commands. Only issue `EXIT` when you intend to close the session entirely—do not include `EXIT` unless the full task is complete.\n\n"
                "Modal/popup alert commands:\n"
                "  ALERT_GET_TEXT - get alert/confirm/prompt message\n"
                "  ALERT_ACCEPT - accept alert\n"
                "  ALERT_DISMISS - dismiss alert\n"
                "  ALERT_SEND_KEYS <text> - send text to prompt and accept\n\n"
                "EXTRACT_ALL_TEXT extracts the main readable text from the page using Mozilla's Readability.js for clean content extraction."
            ),
            "commands": [
                "NAVIGATE <url>",
                "CLICK <css_selector>",
                "CLICK_LINK <index>",
                "EXTRACT_TEXT <css_selector>",
                "EXTRACT_LINKS <css_selector>",
                "EXTRACT_ATTRIBUTES <css_selector> <attribute>",
                "GET_PAGE_SOURCE",
                "WAIT <seconds>",
                "LIST_SELECTORS",
                "LIST_TAGS",
                "FIND_SELECTOR_CONTAINING <text>",
                "SCROLL <pixels>",
                "SCROLL_TO_BOTTOM",
                "SCROLL_TO_ELEMENT <css_selector>",
                "TAKE_SCREENSHOT <filename>",
                "ALERT_GET_TEXT",
                "ALERT_ACCEPT",
                "ALERT_DISMISS",
                "ALERT_SEND_KEYS <text>",
                "EXTRACT_ALL_TEXT",
                "SCRIPT <commands>",
                "EXIT"
            ],
            "state_summary": {
                "current_url": self.driver.current_url,
                "last_extracted_links_preview": self._last_links[:5],
                "queued_commands_count": len(self._input_queue),
                "screenshots_in_memory": len(self.screenshots),
                "log_entries_count": len(self.detailed_logs),
            }
        }

    def cleanup(self):
        self._debug_running = False
        try:
            self.driver.quit()
        except Exception:
            pass

        try:
            output_path = f"./debug_video_{self.session_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.mp4"
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-framerate", "10",
                "-pattern_type", "glob",
                "-i", os.path.join(self.debug_dir, "*.jpg"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                output_path
            ]
            subprocess.run(ffmpeg_cmd, check=True)
            self.log(f"[debug_capture] Video saved to {output_path}")
        except Exception as e:
            self.log(f"[debug_capture] Failed to compile video: {e}")

        # Cleanup frames folder for this session only
        try:
            import shutil
            shutil.rmtree(self.debug_dir, ignore_errors=True)
        except Exception as e:
            self.log(f"[debug_capture] Failed to delete frames: {e}")


if __name__ == "__main__":
    import sys

    tool = SeleniumInteractiveTool()
    print(tool.describe())
    print("SeleniumInteractiveTool REPL started. Type commands to execute.")
    print("Use SCRIPT for multi-line commands. Type EXIT to quit.")

    try:
        while not getattr(tool, "done", False):
            user_input = input(">> ").strip()

            # Handle multiline SCRIPT input
            if user_input.upper() == "SCRIPT":
                print("Enter SCRIPT commands line by line. Empty line to end.")
                lines = []
                while True:
                    line = input()
                    if not line.strip():
                        break
                    lines.append(line)
                script_block = "SCRIPT\n" + "\n".join(lines)
                tool.send_input(script_block)
            else:
                tool.send_input(user_input)

            output = tool.receive_output()

            # Print output nicely (lists, etc)
            if isinstance(output, list):
                for i, item in enumerate(output):
                    print(f"[{i}] {item}")
            else:
                print(output)

            if getattr(tool, "done", False):
                print("Session ended.")
                break

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Exiting.")

    finally:
        tool.cleanup()
        print("Browser driver quit. Goodbye!")
