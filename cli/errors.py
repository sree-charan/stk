"""Custom error types for better error messages."""


class StkError(Exception):
    """Base error for stock assistant."""
    pass


class InvalidTickerError(StkError):
    def __init__(self, ticker: str) -> None:
        super().__init__(f"Invalid ticker symbol: {ticker}")
        self.ticker = ticker


class NetworkError(StkError):
    def __init__(self, source: str, detail: str = "") -> None:
        msg = f"Network error fetching from {source}"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)
        self.source = source


class RateLimitError(StkError):
    def __init__(self, source: str) -> None:
        super().__init__(f"Rate limit exceeded for {source}. Try again in a few minutes.")
        self.source = source


class NoDataError(StkError):
    """No data available from any source (real or cached)."""
    def __init__(self, ticker: str, source: str = "") -> None:
        msg = f"No data available for {ticker}"
        if source:
            msg += f" from {source}"
        super().__init__(msg)
        self.ticker = ticker


class ConfigError(StkError):
    """Missing or invalid configuration."""
    def __init__(self, key: str, hint: str = "") -> None:
        msg = f"Configuration missing: {key}"
        if hint:
            msg += f". {hint}"
        super().__init__(msg)
        self.key = key
