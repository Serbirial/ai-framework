import datetime

def log(prefix, data):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    string = f"[{timestamp}] [{prefix}] {data}"
    print(string)
