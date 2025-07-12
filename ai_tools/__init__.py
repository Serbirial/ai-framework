
from .safe_math import EXPORT as SAFE_MATH_EXPORT
from .get_device_time import EXPORT as LOCAL_TIME_EXPORT
from .safe_calculus import EXPORT as SAFE_CALC_EXPORT
from .cat_facts import EXPORT as CAT_FACT_EXPORT
from .get_weather import EXPORT as WEATHER_API_EXPORT
from .get_latlot import EXPORT as LATLOT_API_EXPORT


VALID_ACTIONS = {}

VALID_ACTIONS.update(SAFE_MATH_EXPORT)
VALID_ACTIONS.update(SAFE_CALC_EXPORT)


VALID_ACTIONS.update(CAT_FACT_EXPORT)
VALID_ACTIONS.update(WEATHER_API_EXPORT)
VALID_ACTIONS.update(LATLOT_API_EXPORT)


VALID_ACTIONS.update(LOCAL_TIME_EXPORT)

# If you add more tool modules, import their EXPORT dict and update VALID_ACTIONS similarly

# Example:
# from another_tool import EXPORT as ANOTHER_EXPORT
# VALID_ACTIONS.update(ANOTHER_EXPORT)
