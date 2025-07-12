import requests

def coingecko_price(params: dict) -> dict:
    """
    Get the current price of one or more cryptocurrencies using CoinGecko's free public API.
    Example: {"coins": ["bitcoin", "ethereum"], "vs_currency": "usd", "include_market_data": true}
    """
    try:
        coins = params.get("coins")
        currency = params.get("vs_currency", "usd").lower()
        include_market = params.get("include_market_data", False)

        if not coins or not isinstance(coins, list):
            return {"error": "Missing or invalid 'coins' list."}

        coin_ids = ",".join(coins)
        url = "https://api.coingecko.com/api/v3/simple/price"
        res = requests.get(
            url,
            params={
                "ids": coin_ids,
                "vs_currencies": currency,
                "include_market_cap": str(include_market).lower(),
                "include_24hr_vol": str(include_market).lower(),
                "include_24hr_change": str(include_market).lower(),
                "include_last_updated_at": str(include_market).lower()
            },
            timeout=6
        )

        if not res.ok:
            return {"error": f"CoinGecko API error: {res.status_code}"}

        data = res.json()
        if not data:
            return {"error": "No data returned for requested coins."}

        results = {}
        for coin in coins:
            if coin not in data:
                results[coin] = {"error": "Not found"}
                continue

            entry = data[coin]
            formatted = f"{coin.capitalize()}: {entry.get(currency)} {currency.upper()}"
            if include_market:
                change = entry.get(f"{currency}_24h_change")
                cap = entry.get(f"{currency}_market_cap")
                volume = entry.get(f"{currency}_24h_vol")
                formatted += (
                    f"\n  - 24h Change: {change:.2f}%"
                    f"\n  - Market Cap: {cap:,.0f} {currency.upper()}"
                    f"\n  - Volume (24h): {volume:,.0f} {currency.upper()}"
                )
            results[coin] = {"result": formatted}

        return results

    except Exception as e:
        return {"error": f"Exception: {str(e)}"}

EXPORT = {
    "coingecko_price": {
        "help": (
            "Fetch real-time crypto prices from CoinGecko. Supports multiple coins.\n"
            "Params:\n"
            "- coins: list of coin IDs (e.g., ['bitcoin', 'ethereum'])\n"
            "- vs_currency: fiat currency to compare against (e.g., 'usd', 'eur')\n"
            "- include_market_data: true to include market cap, 24h volume, change"
        ),
        "callable": coingecko_price,
        "params": {
            "coins": "list of coin IDs (e.g. ['bitcoin'])",
            "vs_currency": "string, default 'usd'",
            "include_market_data": "bool, optional"
        }
    }
}
