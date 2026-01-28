"""
熔断机制模块 (Circuit Breaker)

熔断机制是一种保护系统的设计模式，类似于电路保险丝：
1. CLOSED（闭合）：正常状态，请求正常通过
2. OPEN（断开）：连续失败达到阈值后，直接拒绝请求，避免雪崩
3. HALF_OPEN（半开）：经过一段时间后，允许部分请求通过，测试服务是否恢复

核心参数：
- failure_threshold: 触发熔断的连续失败次数
- recovery_timeout: 熔断后多久进入半开状态
- half_open_max_calls: 半开状态允许的最大请求数
"""

import time
import asyncio
from enum import Enum
from typing import Callable, TypeVar, ParamSpec
from functools import wraps
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"      # 正常
    OPEN = "open"          # 熔断
    HALF_OPEN = "half_open"  # 半开（测试恢复）


@dataclass
class CircuitBreakerStats:
    """熔断器统计"""
    total_calls: int = 0
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    last_failure_time: float = 0
    state_changes: list = field(default_factory=list)


class CircuitBreaker:
    """
    熔断器
    
    使用方式：
    ```python
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
    
    @breaker
    def call_api():
        return requests.get("https://api.example.com")
    ```
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        fallback: Callable | None = None,
        name: str = "default",
    ):
        """
        初始化熔断器
        
        Args:
            failure_threshold: 连续失败多少次后触发熔断
            recovery_timeout: 熔断后多少秒进入半开状态
            half_open_max_calls: 半开状态允许尝试的请求数
            fallback: 熔断时的降级函数
            name: 熔断器名称（用于日志）
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.fallback = fallback
        self.name = name
        
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> CircuitState:
        """获取当前状态（自动检查是否应该转换）"""
        if self._state == CircuitState.OPEN:
            # 检查是否应该进入半开状态
            if time.time() - self._stats.last_failure_time >= self.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state
    
    @property
    def stats(self) -> CircuitBreakerStats:
        """获取统计信息"""
        return self._stats
    
    def _transition_to(self, new_state: CircuitState):
        """状态转换"""
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            self._stats.state_changes.append({
                "from": old_state.value,
                "to": new_state.value,
                "time": time.time(),
            })
            logger.info(f"[CircuitBreaker:{self.name}] {old_state.value} -> {new_state.value}")
            
            if new_state == CircuitState.HALF_OPEN:
                self._half_open_calls = 0
    
    def _record_success(self):
        """记录成功"""
        self._stats.total_calls += 1
        self._stats.success_count += 1
        self._stats.consecutive_failures = 0
        
        # 半开状态下成功，恢复到闭合
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            if self._half_open_calls >= self.half_open_max_calls:
                self._transition_to(CircuitState.CLOSED)
    
    def _record_failure(self, error: Exception):
        """记录失败"""
        self._stats.total_calls += 1
        self._stats.failure_count += 1
        self._stats.consecutive_failures += 1
        self._stats.last_failure_time = time.time()
        
        logger.warning(f"[CircuitBreaker:{self.name}] 调用失败: {error}")
        
        # 检查是否需要熔断
        if self._state == CircuitState.CLOSED:
            if self._stats.consecutive_failures >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.HALF_OPEN:
            # 半开状态下失败，重新熔断
            self._transition_to(CircuitState.OPEN)
    
    def __call__(self, func: Callable[P, T]) -> Callable[P, T]:
        """装饰器模式"""
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return await self.call_async(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return self.call_sync(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    async def call_async(self, func: Callable, *args, **kwargs):
        """异步调用"""
        async with self._lock:
            state = self.state
        
        if state == CircuitState.OPEN:
            logger.warning(f"[CircuitBreaker:{self.name}] 熔断中，使用降级策略")
            if self.fallback:
                return self.fallback(*args, **kwargs)
            raise CircuitOpenError(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            async with self._lock:
                self._record_success()
            return result
            
        except Exception as e:
            async with self._lock:
                self._record_failure(e)
            
            if self.fallback:
                return self.fallback(*args, **kwargs)
            raise
    
    def call_sync(self, func: Callable, *args, **kwargs):
        """同步调用"""
        state = self.state
        
        if state == CircuitState.OPEN:
            logger.warning(f"[CircuitBreaker:{self.name}] 熔断中，使用降级策略")
            if self.fallback:
                return self.fallback(*args, **kwargs)
            raise CircuitOpenError(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            if self.fallback:
                return self.fallback(*args, **kwargs)
            raise
    
    def reset(self):
        """重置熔断器"""
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._half_open_calls = 0
        logger.info(f"[CircuitBreaker:{self.name}] 已重置")


class CircuitOpenError(Exception):
    """熔断器打开异常"""
    pass


# ===== 带重试的熔断器 =====

def with_retry(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    exponential_backoff: bool = True,
    retryable_exceptions: tuple = (Exception,),
):
    """
    重试装饰器
    
    Args:
        max_retries: 最大重试次数
        retry_delay: 初始重试延迟（秒）
        exponential_backoff: 是否使用指数退避
        retryable_exceptions: 可重试的异常类型
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    return func(*args, **kwargs)
                    
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = retry_delay * (2 ** attempt if exponential_backoff else 1)
                        logger.warning(f"重试 {attempt + 1}/{max_retries}，延迟 {delay}s: {e}")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"重试耗尽: {e}")
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = retry_delay * (2 ** attempt if exponential_backoff else 1)
                        logger.warning(f"重试 {attempt + 1}/{max_retries}，延迟 {delay}s: {e}")
                        time.sleep(delay)
                    else:
                        logger.error(f"重试耗尽: {e}")
            
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# ===== 全局熔断器实例 =====

# LLM 调用熔断器
llm_circuit_breaker = CircuitBreaker(
    name="llm",
    failure_threshold=3,
    recovery_timeout=60.0,
    fallback=lambda *args, **kwargs: {
        "type": "response",
        "content": "抱歉，系统暂时繁忙，请稍后重试。您也可以联系人工客服获取帮助。",
        "tool_calls": None,
    },
)
