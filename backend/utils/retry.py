import time
import asyncio
from typing import Callable, Any, TypeVar

T = TypeVar('T')

def retry(fn: Callable[[], T], attempts: int = 3, backoff: float = 2.0) -> T:
    """Synchronous retry helper with exponential backoff."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if i == attempts - 1:
                raise e
            time.sleep(backoff ** i)

async def retry_async(fn: Callable[[], Any], attempts: int = 3, backoff: float = 2.0) -> Any:
    """Asynchronous retry helper with exponential backoff."""
    for i in range(attempts):
        try:
            if asyncio.iscoroutinefunction(fn):
                return await fn()
            else:
                return fn()
        except Exception as e:
            if i == attempts - 1:
                raise e
            await asyncio.sleep(backoff ** i)
