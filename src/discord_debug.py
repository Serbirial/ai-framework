import io, json, discord, asyncio

def custom_debug(**data):
    # Pretty print the debug data
    pretty = json.dumps(data, indent=2, ensure_ascii=False)

    # Save to local file (timestamped for uniqueness)
    filename = f"debug.txt"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(pretty)
        print(f"DEBUG: Saved debug dump to {filename}")
    except Exception as e:
        print("ERROR: Failed to save debug file locally:", e)

