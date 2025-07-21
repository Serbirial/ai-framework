# AI Framework

An experimental AI chatbot framework designed for Discord integration, featuring mood-aware responses, memory management, and recursive thinking capabilities.

## Features

- **Recursive Thinking**: Models can simulate reasoning by generating internal thoughts before final responses.
- **Mood-Aware Chatbot**: Mood is tracked and influences tone and behavior (e.g. happy, annoyed, angry).
- **Function Routing (Tool Use)**: The model can return structured commands like `<Action>...</Action>` which are routed internally.
- **Discord Integration**: Includes a functional bot setup with message handling, streaming, and debugging tools.
- **Prompt Management**: Prompts are kept modular and organized for easy tuning. (MOVING TO JSON PROMPT STORAGE!)

## Goal

My main goal with this is to make a 'framework' that can be used with any compatible model (meta llama prompt styles, current is 3.2) and turns it into a selfhostable 'gpt' alternative, with the pro's of being open source and fully customizable for any task or workflow with a bit of elbow grease.

I essentially want this to end up being my own selfhosted version of chatGPT, matching its capabilities and functions but with *more*, and without being shackled with rules.

## Example Workflow

* User: What's the capital of France?

* Classifying: Decides this is a task
* Route -> Task Worker
* Step 1: Action: web_search("capital of France")
* Tool response: The capital is Paris.
* Step 2: Confirms and sets up filler for next step
* Step 3: Filler setting up for final reply

* Final Reply: The capital of France is Paris!
