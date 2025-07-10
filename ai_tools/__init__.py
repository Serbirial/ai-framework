
from .safe_math import EXPORT as SAFE_MATH_EXPORT
from .get_time import EXPORT as TIME_EXPORT


VALID_ACTIONS = {}

VALID_ACTIONS.update(SAFE_MATH_EXPORT)
VALID_ACTIONS.update(TIME_EXPORT)

# If you add more tool modules, import their EXPORT dict and update VALID_ACTIONS similarly

# Example:
# from another_tool import EXPORT as ANOTHER_EXPORT
# VALID_ACTIONS.update(ANOTHER_EXPORT)
