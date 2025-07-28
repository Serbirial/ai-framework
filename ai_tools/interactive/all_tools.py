
from .selenium import SeleniumInteractiveTool
from .discord_temp_1msg import DiscordInteractiveTool

def load_interactive_tools():
    tool_classes = [
        SeleniumInteractiveTool,
        DiscordInteractiveTool
    ]

    tools_dict = {}
    for tool_cls in tool_classes:
        try:
            instance = tool_cls()
            desc = instance.describe()
            desc["instance"] = instance
            tool_name = desc.get("name")
            if not tool_name:
                raise ValueError(f"Tool {tool_cls} .describe() missing 'name'")
            tools_dict[tool_name] = desc
        except Exception as e:
            print(f"Error loading tool {tool_cls}: {e}")
    return tools_dict


def load_interactive_tools():
    return  load_interactive_tools()

def describe_all_tools():
    tools = load_interactive_tools()
    print("Registered Interactive Tools:")
    for name, desc in tools.items():
        print(f"\n🔹 Tool Name: {name}")
        print(f"  Description: {desc.get('description', '(no description)')}")
        print(f"  Commands: {', '.join(desc.get('commands', []))}")
        state_summary = desc.get("state_summary")
        if state_summary:
            print(f"  State summary keys: {', '.join(state_summary.keys())}")

if __name__ == "__main__":
    describe_all_tools()
