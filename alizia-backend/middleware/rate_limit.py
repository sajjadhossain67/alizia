"""Alizia AI Backend - Rate Limiting Middleware"""

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
import time
import threading
from typing import Dict, Set
from core.config import settings


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: int = 100, burst: int = 20):
        """Initialize rate limiter.
        
        Args:
            rate: Maximum requests per time window
            burst: Burst allowance before throttling
        """
        self.rate = rate
        self.burst = burst
        self._counters: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"tokens": float(burst), "reset_time": time.time()}
        )
        self._lock = threading.Lock()
    
    def _get_key(self, request: Request) -> str:
        """Generate a unique key for the rate limit check."""
        # Use combination of IP and API key if available
        client_host = request.client.host if request.client else "unknown"
        return f"rate_limit:{client_host}"
    
    def check_rate(self, key: str) -> bool:
        """Check if the request is within rate limits.
        
        Returns:
            True if allowed, False if rate limited
        """
        with self._lock:
            now = time.time()
            counter = self._counters[key]
            
            # Reset tokens if window has passed
            if now - counter["reset_time"] >= 60:  # 1-minute window
                counter["tokens"] = float(self.burst)
                counter["reset_time"] = now
            
            if counter["tokens"] > 0:
                counter["tokens"] -= 1
                return True
            return False
    
    def reset_rate(self, key: str) -> None:
        """Reset the rate limiter for a key."""
        with self._lock:
            self._counters[key] = {"tokens": float(self.burst), "reset_time": time.time()}


# Global rate limiter instance
rate_limiter = RateLimiter(
    rate=settings.RATE_LIMIT_DEFAULT if hasattr(settings, 'RATE_LIMIT_DEFAULT') else 100,
    burst=settings.RATE_LIMIT_BURST if hasattr(settings, 'RATE_LIMIT_BURST') else 20,
)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""
    
    def __init__(self, app, rate: int = 100, burst: int = 20):
        super().__init__(app)
        self.rate_limiter = RateLimiter(rate=rate, burst=burst)
    
    async def dispatch(self, request: Request, call_next):
        """Process request through rate limiter."""
        key = self.rate_limiter._get_key(request)
        
        if not self.rate_limiter.check_rate(key):
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                        "message": "Request limit exceeded. Please try again later.",
                    }
                },
                headers={
                    "x-ratelimit-limit-requests": str(self.rate_limiter.rate),
                    "x-ratelimit-remaining-requests": 
                        str(max(0, self.rate_limiter._counters[key]["tokens"])),
                    "x-ratelimit-reset-requests": str(int(self.rate_limiter._counters[key]["reset_time"])),
                    "retry-after": "60",
                },
            )
        
        response: Response = await call_next(request)
        
        # Add rate limit headers to successful responses
        if hasattr(response, 'headers'):
            key = self.rate_limiter._get_key(request)
            response.headers['x-ratelimit-limit-requests'] = str(self.rate_limiter.rate)
            response.headers['x-ratelimit-remaining-requests'] = str(
                max(0, self.rate_limiter._counters[key]["tokens"] - 1)
            )
            response.headers['x-ratelimit-reset-requests'] = str(
                int(self.rate_limiter._counters[key]["reset_time"])
            )
        
        return response


# Dependency for per-endpoint rate limiting
async def check_rate_limit(request: Request) -> bool:
    """Dependency to check rate limit for a specific endpoint."""
    key = rate_limiter._get_key(request)
    return rate_limiter.check_rate(key)