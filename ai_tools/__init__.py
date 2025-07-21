from .safe_math import EXPORT as SAFE_MATH_EXPORT
from .get_device_time import EXPORT as LOCAL_TIME_EXPORT
from .safe_calculus import EXPORT as SAFE_CALC_EXPORT
from .cat_facts import EXPORT as CAT_FACT_EXPORT
from .get_weather import EXPORT as WEATHER_API_EXPORT
from .get_latlot import EXPORT as LATLOT_API_EXPORT
from .safe_web_query import EXPORT as SEARCH_ENGINE_API_EXPORT
from .get_time import EXPORT as GET_TIME_LIVE
from .safe_get_sysinfo import EXPORT as GET_DEVICE_INFO
from .safe_live_crypto import EXPORT as GET_CRYPTO_LIVE
from .safe_wikipedia import EXPORT as GET_WIKIPEDIA
from .safe_python_sandbox import EXPORT as SAFE_PYTHON_SANDBOX
from .unsafe_youtube_scraper import EXPORT as UNSANFE_YOUTUBE_SCRAPE
from .unsafe_scrape import EXPORT as UNSAFE_RAW_SCRAPER
from .website_ping import EXPORT as WEBSITE_PING


from src.static import WEB_ACCESS
VALID_ACTIONS = {}


#if WEB_ACCESS:
    #VALID_ACTIONS.update(GET_CRYPTO_LIVE)
VALID_ACTIONS.update(GET_DEVICE_INFO)

if WEB_ACCESS:
    VALID_ACTIONS.update(CAT_FACT_EXPORT)
    #VALID_ACTIONS.update(LATLOT_API_EXPORT) doesnt need this tool tbh


VALID_ACTIONS.update(LOCAL_TIME_EXPORT)
if WEB_ACCESS:
    #VALID_ACTIONS.update(WEATHER_API_EXPORT)
    #VALID_ACTIONS.update(GET_WIKIPEDIA) this is broken
    VALID_ACTIONS.update(SEARCH_ENGINE_API_EXPORT)
    VALID_ACTIONS.update(UNSANFE_YOUTUBE_SCRAPE)
    VALID_ACTIONS.update(WEBSITE_PING)
    VALID_ACTIONS.update(GET_TIME_LIVE)
    VALID_ACTIONS.update(UNSAFE_RAW_SCRAPER)


VALID_ACTIONS.update(SAFE_MATH_EXPORT)
VALID_ACTIONS.update(SAFE_CALC_EXPORT)
VALID_ACTIONS.update(SAFE_PYTHON_SANDBOX)

def describe_valid_actions(valid_actions):
    print("Registered Tools:")
    for key, tool in sorted(valid_actions.items()):
        help_text = tool.get("help", "(no help text)")
        print(f"\n🔹 Tool Name: {key}")
        print(f"   Description: {help_text}")

describe_valid_actions(VALID_ACTIONS)
# If you add more tool modules, import their EXPORT dict and update VALID_ACTIONS similarly

# Example:
# from another_tool import EXPORT as ANOTHER_EXPORT
# VALID_ACTIONS.update(ANOTHER_EXPORT)

