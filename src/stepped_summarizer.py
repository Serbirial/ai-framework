from log import log
from .static import StopOnSpeakerChange
from utils.helpers import DummyTokenizer
from utils import openai
from .ai_processing import compress_summary
from utils.tokens import get_token_availability, find_compression_ratio


def chunk_data(max_output_tokens, data):
    tokenizer = DummyTokenizer()
    
    chunk_size = int(max_output_tokens // 4)

    all_tokens = tokenizer.encode(data)
    raw_token_count = len(all_tokens)

    if raw_token_count >= max_output_tokens:
        chunks = [all_tokens[i:i + chunk_size] for i in range(0, raw_token_count, chunk_size)]
        return [tokenizer.decode(chunk) for chunk in chunks]
    else:
        return [data]


def find_token_limit(data_tokens, compression_ratio, minimum_summary_size):
    return max(int(data_tokens * compression_ratio), minimum_summary_size)

def format_chunks(chunks: list):
    combined = ""
    i = 0
    for chunk in chunks:
        i += 1
        combined += f"### Section {i}:\n"
        combined += f"{chunk}\n\n"
    return combined



class SteppedSummarizing:
    def __init__(self, model, config, tier: str, data: str, streamer=None):
        self.model = model  # Reference to ChatBot
        self.data = data
        self.config = config
        self.streamer = streamer
        self.summaries = []
        
        self.summary_compression = config.token_config["SUMMARY_COMPRESSION_PERCENT"]
        self.summary_nested_compression = config.token_config["SUMMARY_NESTED_COMPRESSION_PERCENT"]
        self.minimum_summary_size = self.config.token_config["SUMMARY_MINIMUM_TOKEN_LIMIT"]


        self.max_token_window = self.config.token_config[tier]["BASE_TOKEN_WINDOW"]
        self.max_chat_window = self.config.token_config[tier]["MAX_CHAT_WINDOW"]
        self.prompt_window = self.config.token_config["PROMPT_RESERVATION"]
        self.max_output_tokens = (self.max_token_window - self.max_chat_window) - self.prompt
        
        self.token_buffer = min(512, int(0.1 * self.max_output_tokens)) # 'reserve' a buffer of 512 tokens

        self.chunks = chunk_data(self.max_output_tokens, self.data)

    def build_prompt(self):
        base = (
            #"<|begin_of_text|>"

            "<|start_header_id|>system<|end_header_id|>\n"
            f"You are a smart summarizer.\n"
            "You summarize a single chunk of data into a concise summary that can be combined with all the previous summaries to paint a full picture.\n"

        )

        log("STEPPED SUMMARY PROMPT", base)
        return base
    
    def build_final_prompt(self):

        base = (
            #"<|begin_of_text|>"

            "<|start_header_id|>system<|end_header_id|>\n"
            f"You are a highly capable summarization engine.\n"
            "Your job is to read all sections and produce one clear, complete summary that captures the entire meaning and structure of the input.\n" 
        )
        base += format_chunks(self.summaries)

        log("FINAL TASK PROMPT", base)
        return base

    def think(self, username):
        tokenizer = DummyTokenizer()
        if self.streamer:
            self.streamer.add_special(f"Summarizing something")
        prompt = self.build_prompt()
        

        i = 0
        for chunk in self.chunks:
            i += 1
            if self.streamer:
                self.streamer.add_special(f"Section {i} out of {self.depth} on summary")
                
            current_prompt_tokens = tokenizer.count_tokens(prompt)
            raw_chunk_tokens = tokenizer.count_tokens(chunk)

            available_tokens, _ = get_token_availability(
                current_prompt_tokens=current_prompt_tokens,
                other_tokens=raw_chunk_tokens,
                total_token_window=self.max_token_window,
                reserved_buffer=1024,
            )

            compression_ratio = find_compression_ratio(available_tokens, raw_chunk_tokens)
            summary_token_limit = find_token_limit(
                data_tokens=raw_chunk_tokens,
                compression_ratio=compression_ratio,
                minimum_summary_size=128
            )

            summary_prompt = "<|begin_of_text|>" + prompt
            summary_prompt += f"### Section {i}:\n{chunk}\n"
            summary_prompt += "<|eot_id|>\n"
            summary_prompt += "<|start_header_id|>assistant<|end_header_id|>\n"

            log(f"SUMMARY {i} of {len(self.chunks)} PROMPT\n", summary_prompt)

            stop_criteria = StopOnSpeakerChange()

            response = self.model.create_completion(
                prompt=summary_prompt,
                max_tokens=summary_token_limit,
                temperature=0.7,
                top_p=0.9,
                stop_criteria=stop_criteria,
            )

            summary_output = openai.extract_generated_text(response).strip()
            clean_summary_output = summary_output.replace("<|begin_of_text|>", "").strip()

            self.summaries.append(clean_summary_output)
            log(f"DEBUG: SUMMARY STEP {i}", clean_summary_output)


        tokenizer = DummyTokenizer()

        formatted_summaries = format_chunks(self.summaries)
        base_prompt = self.build_final_prompt()
        suffix = "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"

        base_tokens = tokenizer.count_tokens(base_prompt)
        suffix_tokens = tokenizer.count_tokens(suffix)
        summary_tokens = tokenizer.count_tokens(formatted_summaries)

        generation_token_budget = self.max_token_window / 2
        prompt_budget = self.max_token_window - generation_token_budget

        available_summary_tokens = prompt_budget - base_tokens - suffix_tokens
        overflow = summary_tokens - available_summary_tokens

        if overflow > 0:
            if self.streamer:
                self.streamer.add_special(f"WARNING: Summary past alloted token window, trying compression...")
            # compress summaries to fit
            compressed_summary = compress_summary(
                self.model,
                formatted_summaries,
                category="summary",
                max_tokens=available_summary_tokens
            )
            compressed_summary_tokens = tokenizer.count_tokens(compressed_summary)
            overflow = compressed_summary_tokens - available_summary_tokens
            if overflow > 0:
                
                if self.streamer:
                    self.streamer.add_special(f"WARNING: Summary compression failed to fit summary in alloted token window- model may soft crash and not respond, a fix is in progress to prevent this.")
            formatted_summaries = compressed_summary
            summary_tokens = compressed_summary_tokens

        final_prompt = base_prompt + "\n" + formatted_summaries + suffix

        if self.streamer:
            self.streamer.add_special("Finalizing the summary!")

        stop_criteria = StopOnSpeakerChange()
        _final_summary = self.model.create_completion(
            max_tokens=generation_token_budget,
            temperature=0.7,
            top_p=0.9,
            stop_criteria=stop_criteria,
            prompt=final_prompt,
        )

        final_summary = openai.extract_generated_text(_final_summary).strip()
        final_tokens_used = tokenizer.count_tokens(final_prompt)

        log("DEBUG: FINAL PROMPT TOKENS:", final_tokens_used)
        log("DEBUG: FINAL SUMMARY PROMPT:", final_prompt)
        log("DEBUG: FINAL GENERATED SUMMARY:", final_summary)

        return final_prompt, final_summary


def fit_summary(model, config, tier, data, streamer=None, max_passes=3):
    current_data = data
    for pass_num in range(max_passes):
        thinker = SteppedSummarizing(model, config, tier, current_data, streamer)
        _, summary = thinker.think(username="recursive_summarizer")

        # Token count of the summary
        tokenizer = DummyTokenizer()
        tokens = tokenizer.count_tokens(summary)

        if tokens < config.token_config[tier]["BASE_TOKEN_WINDOW"] * 0.7:
            # Summary fits comfortably, stop here
            return summary

        # Otherwise, set current_data to this summary and compress again
        current_data = summary
    
    # After max_passes, return whatever summary we have
    return current_data
