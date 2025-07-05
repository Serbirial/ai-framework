from src import bot, static
import psutil
import os


def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    print(f"[Memory Usage] {mem_mb:.2f} MB")

if __name__ == "__main__":
    chatbot = bot.ChatBot("console-AI", "console.json")
    username = "serbirial" # temp name 
    history = []

    print("\nType 'exit' to quit.\n")
    while True:
        inp = input("You > ")
        if inp.strip().lower() == "exit":
            break

        history.append(f"{username}: {inp}")
        
        # Simulate growing context
        context_string = "\n".join(history[-20:])  # last 20 turns to keep it reasonable
        data = chatbot.chat(
            username, inp, username,
            max_new_tokens=400,
            temperature=0.7,
            top_p=0.9,
            context=context_string 
        )

        history.append(f"console-AI: {data}")
        print("console-AI >", data + "\n")

        print_memory_usage()
