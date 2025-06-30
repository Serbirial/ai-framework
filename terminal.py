from src import bot

history = []

if __name__ == "__main__":
    chatbot = bot.ChatBot("console-AI", "console.json")
    username = input("Your Username? >")
    while 1:
        inp = input("You >")
        history.append(f"{username}: " + inp)
        data = chatbot.chat(username,inp, username, 300, temperature=0.7, top_p=0.9 )
        history.append(f"console-AI: " + data)
        print(data + "\n")