from log import log
from .static import StopOnSpeakerChange
from utils.helpers import DummyTokenizer
from utils import openai


def chunk_data(max_output_tokens, data):
    tokenizer = DummyTokenizer()

    chunk_size = max_output_tokens / 4

    raw_token_count = len(tokenizer.encode(data))
    if raw_token_count >= max_output_tokens:
        all_tokens = tokenizer.encode(data)
        chunks = [all_tokens[i:i+chunk_size] for i in range(0, len(all_tokens), chunk_size)]
        return [tokenizer.decode(chunk) for chunk in chunks]
    else:
        return [tokenizer.decode(data)]

def find_compression_ratio(max_output_tokens, token_buffer, data_tokens):
    compression_ratio = (max_output_tokens - token_buffer) / data_tokens
    # Clip between a floor and ceiling to avoid overcompression or undercompression
    return max(0.5, min(0.9, compression_ratio))

def find_token_limit(data_tokens, compression_ratio, minimum_summary_size):
    max(int(data_tokens * compression_ratio), minimum_summary_size)
    
def format_chunks(chunks: list):
    combined = ""
    i = 0
    for chunk in chunks:
        i += 1
        combined += f"### Section {i}:\n"
        combined += f"{chunk}\n\n"
    return combined

class SteppedSummarizing: # TODO: check during steps if total tokens are reaching token limit- if they are: summarize all steps into a numbered summary then re-build the prompt using it and start (re-using the depth limit but not step numbers)
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
        self.prompt_window = self.config.token_config[tier]["PROMPT_RESERVATION"]
        self.max_output_tokens = (self.max_token_window - self.max_chat_window) - self.prompt
            
        self.token_buffer = min(512, int(0.1 * self.max_output_tokens)) # 'reserve' a buffer of 512 tokens


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

        self.streamer.add_special(f"Summarizing web output")
        prompt = self.build_prompt()

        full = f"{prompt}"

        to_add = ""
        i = 0
        for chunk in self.chunks:
            i += 1
            # start with the system prompt or base context
            summary_prompt = f"{full}"
            summary_prompt += f"### **To Summarize**:\n{chunk}\n"
            # end sys prompt
            summary_prompt += "<|eot_id|>"



            stop_criteria = StopOnSpeakerChange() 
            summary_prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
            # response begins here
            log(f"SUMMARY {i} of {len(self.chunks)} PROMPT\n", summary_prompt)
            
            summary_token_limit = find_token_limit(tokenizer.encode(chunk), self.summary_compression, self.minimum_summary_size)

            
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
            
            to_add += "<|start_header_id|>assistant<|end_header_id|>\n"
            # append the full step (header + content) to the full conversation log
            to_add += f"### Section {i}:\n{clean_summary_output}\n"
            to_add == "<|eot_id|>\n"

        formatted_summaries = format_chunks(self.summaries)

        final_prompt = self.build_final_prompt()
        final_prompt += "\n"
        final_prompt += formatted_summaries
        final_prompt += "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"



        tokenizer = DummyTokenizer()
        final_tokens = tokenizer.count_tokens(final_prompt)

        log(f"DEBUG: FINAL SUMMARY PROMPT:\n",final_prompt)
        log(f"\nDEBUG: FINAL PROMPT TOKENS", final_tokens)

        stop_criteria = StopOnSpeakerChange() 
        self.streamer.add_special(f"Finalizing the summary!")

        compression_ratio = find_compression_ratio(self, self.token_buffer, formatted_summaries)
        log("\nSUMMARY COMPRESSION RATIO", compression_ratio)
        final_summary_compressed_token_limit = find_token_limit(final_tokens, compression_ratio, self.minimum_summary_size)
        log("\SUMMARY FINAL TOKEN LIMIT", final_summary_compressed_token_limit)
        
        _final_summary = self.model.create_completion(
            max_tokens=final_summary_compressed_token_limit,
            temperature=0.7,
            top_p=0.9,
            stop_criteria=stop_criteria,
            prompt=final_prompt,
        )
        final_summary = openai.extract_generated_text(_final_summary).strip()
        log("\n\nDEBUG: SUMMARY WORK",final_summary)
        final_tokens_used = tokenizer.count_tokens(final_prompt)

        log(f"DEBUG: FINAL TOKEN SIZE:", final_tokens_used)


        return final_prompt, final_summary

