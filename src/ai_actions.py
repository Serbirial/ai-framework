from log import log

from . import classify
from ai_tools import VALID_ACTIONS
from .static import DummyTokenizer, Config
from .ai_processing import summarize_chunks, summarize_raw_scraped_data, compress_summary

config = Config()

tokenizer = DummyTokenizer()

import json
import re
import time


def log_action_execution(action_name, action_params, action_label, result):
    try:
        with open("executed_actions.txt", "a", encoding="utf-8") as f:
            f.write(f"Action: {action_name}\n")
            f.write(f"Parameters: {json.dumps(action_params)}\n")
            f.write(f"Label: {action_label}\n")
            f.write(f"Result: {result}\n")
            f.write("-" * 40 + "\n")
    except Exception as e:
        print(f"ERROR: Failed to write to executed_actions.txt: {str(e)}")

# Minimal per-action rate limit intervals in seconds
# You can adjust per-action or have a global default
RATE_LIMITS = {
    # action_name: min_seconds_between_calls
    "duckduckgo_query": 1.5,
    "get_current_time": 1.0,
    "wikipedia_query_scraper": 1.0,
    "coingecko_price": 1.0,
    "run_calculus_wolfram_alpha": 1.0,
    "simple_url_scraper": 1.0,
    "get_weather": 1.0
    # Add other actions here as needed
}

# Keep track of last call times globally
last_call_times = {}

def check_rate_limit(action_name):
    min_interval = RATE_LIMITS.get(action_name, 0)  # default 1 second if not configured
    now = time.monotonic()
    last_time = last_call_times.get(action_name, 0)
    elapsed = now - last_time
    if elapsed < min_interval:
        wait_time = min_interval - elapsed
        return False, wait_time
    # Update last call time
    last_call_times[action_name] = now
    return True, 0

def chunk_data(max_output_tokens, data):
    chunk_size = max_output_tokens / 4

    raw_token_count = len(tokenizer.encode(data))
    if raw_token_count >= max_output_tokens:
        all_tokens = tokenizer.encode(data)
        chunks = [all_tokens[i:i+chunk_size] for i in range(0, len(all_tokens), chunk_size)]
        return [tokenizer.decode(chunk) for chunk in chunks]
    else:
        return [tokenizer.decode(data)]
    
def format_chunks(chunks: list):
    combined = ""
    i = 0
    for chunk in chunks:
        i += 1
        combined += f"### Section {i}:\n"
        combined += f"{chunk}\n\n"
    return combined

def find_compression_ratio(max_output_tokens, token_buffer, data_tokens):
    compression_ratio = (max_output_tokens - token_buffer) / data_tokens
    # Clip between a floor and ceiling to avoid overcompression or undercompression
    return max(0.5, min(0.9, compression_ratio))

def find_token_limit(data_tokens, compression_ratio, minimum_summary_size):
    max(int(data_tokens * compression_ratio), minimum_summary_size)
    
    
    
def process_raw_web_output(model, max_output_tokens, summary_compression, summary_nested_compression):
    if "error" in result.keys():
        result = result
    else:
        token_buffer = min(512, int(0.1 * max_output_tokens)) # 'reserve' a buffer of 512 tokens
        minimum_summary_size = config.token_config["SUMMARY_MINIMUM_TOKEN_LIMIT"]
        raw_token_count = len(tokenizer.encode(result))
        if raw_token_count >= max_output_tokens:
            chunked_html_parts = chunk_data(max_output_tokens, result)

            chunked_summaries = []
            for i, chunk_text in enumerate(chunked_html_parts):
                _chunk_size = tokenizer.encode(chunk_text)
                chunk_summary_token_limit = find_token_limit(_chunk_size, summary_compression, minimum_summary_size)
                chunked_summaries.append(summarize_raw_scraped_data(model, chunk_text, chunk_summary_token_limit))

            combined = format_chunks(chunked_summaries)

            data_tokens = tokenizer.encode(combined)
            if data_tokens >= max_output_tokens:
                category_string = "Pre-Summarized sections of a Web Page"

                final_summary_token_limit = find_token_limit(_chunk_size, summary_nested_compression, minimum_summary_size)

                final_summary = summarize_chunks(model, combined, category=category_string, max_tokens=final_summary_token_limit)
                
                # One last try by compressing the summary
                final_tokens = tokenizer.encode(final_summary)
                if final_tokens >= max_output_tokens:
                    compression_ratio = find_compression_ratio(max_output_tokens, token_buffer, final_tokens)
                    final_summary_compressed_token_limit = find_token_limit(final_tokens, compression_ratio, minimum_summary_size)
                    final_summary_compressed = compress_summary(model, combined, category=category_string, max_tokens=final_summary_compressed_token_limit)
                    result = {"summary": final_summary_compressed, "url": result["url"]}
            else:
                result = {"summary": combined, "url": result["url"]}

        else:
            data_tokens = tokenizer.encode(result["raw_html"])
            summary_token_limit = max(int(data_tokens * summary_compression), minimum_summary_size)
            summary = summarize_raw_scraped_data(model, result["raw_html"], summary_token_limit)
            result = {"summary": summary, "url": result["url"]}
    return result


def check_for_actions_and_run(model, text, max_token_window, max_chat_window, prompt_size, summary_compression: float = config.token_config["SUMMARY_COMPRESSION_PERCENT"], summary_nested_compression: float = config.token_config["SUMMARY_NESTED_COMPRESSION_PERCENT"], streamer=None):
    results = []
    max_output_tokens = (max_token_window - max_chat_window) - prompt_size
    print(f"ACTION DEBUG: output token window: {max_output_tokens}")

    pos = 0
    text_len = len(text)
    while pos < text_len:
        start_tag_pos = text.find("<", pos)
        if start_tag_pos == -1:
            break
        end_tag_pos = text.find(">", start_tag_pos)
        if end_tag_pos == -1:
            break

        tag_name = text[start_tag_pos + 1:end_tag_pos].strip()
        if tag_name.lower() == "action":
            pattern = re.compile(r"</action>", re.IGNORECASE)
            close_match = pattern.search(text, end_tag_pos + 1)
            if not close_match:
                break
            close_tag_start = close_match.start()
            close_tag_end = close_match.end()

            json_str = text[end_tag_pos + 1 : close_tag_start]
            json_str = json_str[json_str.find('{') : json_str.rfind('}') + 1].strip()
            action_name = None
            action_label = None
            try:
                action_json = json.loads(json_str)
                action_name = action_json.get("action")
                action_params = action_json.get("parameters", {})
                action_label = action_json.get("label", None)
                if streamer:
                    streamer.add_special(f"Executing tool `{action_name}`")
                # Polite sleep ONLY for specific actions
                if action_name in RATE_LIMITS:
                    time.sleep(0.3)

                if not action_label:
                    action_label = f"action_{len(results) + 1}"

                if action_name in VALID_ACTIONS:
                    allowed, wait = check_rate_limit(action_name)
                    if not allowed:
                        time.sleep(wait)
                        print(f"DEBUG: Executing action: {action_name} with {action_params}")
                        result = VALID_ACTIONS[action_name]["callable"](action_params)

                        if action_name == "raw_url_scraper":
                            result = process_raw_web_output(model, max_output_tokens, summary_compression, summary_nested_compression)

                        # Replace <ActionResultX> with <|ipython|> block
                        output = f"<|ipython|>\n# {action_name} result\n{json.dumps(result, indent=2)}\n<|eot_id|>"
                        results.append(output)

                        if streamer:
                            streamer.add_special(f"Executed tool `{action_name}`")

                        log_action_execution(action_name, action_params, action_label, result)

            except Exception as e:
                error_msg = {"error": f"Failed to execute action: {str(e)}"}
                output = f"<|ipython|>\n# Error during {action_name or 'unknown'}\n{json.dumps(error_msg, indent=2)}\n<|eot_id|>"
                results.append(output)

                label = action_label if 'action_label' in locals() else f"action_{len(results) + 1}"
                if streamer:
                    streamer.add_special(f"Errored while executing tool `{action_name if action_name else 'Cant Get Action Name'}`")
                log_action_execution( "unknown", action_params if 'action_params' in locals() else {}, label, error_msg)

            pos = close_tag_end
        else:
            pos = end_tag_pos + 1

    if len(results) > 0:
        if len(results) == 1:
            return results[0]
        else:
            return results
    return "NOACTION"

