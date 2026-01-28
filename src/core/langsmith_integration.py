"""
LangSmith 集成模块

LangSmith 是 LangChain 官方的可观测性平台，提供：
1. 全链路追踪：每个 Agent/Chain 的调用过程
2. 性能监控：延迟、token 消耗、错误率
3. Prompt 管理：版本控制和 A/B 测试
4. 评估工具：自动化评测

配置方式（.env）：
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-api-key
LANGCHAIN_PROJECT=ecommerce-chatbot
```
"""

import os
import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)


def configure_langsmith(
    project_name: str = "ecommerce-chatbot",
    enabled: bool = True,
):
    """
    配置 LangSmith
    
    Args:
        project_name: 项目名称（在 LangSmith 控制台显示）
        enabled: 是否启用
    """
    if not enabled:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.info("LangSmith 已禁用")
        return
    
    api_key = os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        logger.warning("未配置 LANGCHAIN_API_KEY，LangSmith 追踪已禁用")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return
    
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project_name
    
    logger.info(f"LangSmith 已启用: project={project_name}")


def is_langsmith_enabled() -> bool:
    """检查 LangSmith 是否启用"""
    return os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"


def trace_function(name: str | None = None, metadata: dict | None = None):
    """
    函数追踪装饰器
    
    使用 LangSmith 追踪函数执行
    
    Args:
        name: 追踪名称，默认使用函数名
        metadata: 附加元数据
    
    Example:
        @trace_function(name="process_order", metadata={"team": "backend"})
        def process_order(order_id: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not is_langsmith_enabled():
                return await func(*args, **kwargs)
            
            try:
                from langsmith import traceable
                traced_func = traceable(name=name or func.__name__, metadata=metadata)(func)
                return await traced_func(*args, **kwargs)
            except ImportError:
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not is_langsmith_enabled():
                return func(*args, **kwargs)
            
            try:
                from langsmith import traceable
                traced_func = traceable(name=name or func.__name__, metadata=metadata)(func)
                return traced_func(*args, **kwargs)
            except ImportError:
                return func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def create_run_tree(name: str, run_type: str = "chain", inputs: dict | None = None):
    """
    创建 LangSmith Run Tree（手动追踪）
    
    用于追踪非 LangChain 的代码段
    
    Example:
        from langsmith import RunTree
        
        with create_run_tree("custom_process") as rt:
            result = do_something()
            rt.end(outputs={"result": result})
    """
    if not is_langsmith_enabled():
        # 返回一个空的上下文管理器
        from contextlib import nullcontext
        return nullcontext()
    
    try:
        from langsmith import RunTree
        return RunTree(
            name=name,
            run_type=run_type,
            inputs=inputs or {},
        )
    except ImportError:
        from contextlib import nullcontext
        return nullcontext()


class LangSmithMetrics:
    """
    LangSmith 指标收集器
    
    收集并上报自定义指标到 LangSmith
    """
    
    @staticmethod
    def log_feedback(
        run_id: str,
        key: str,
        score: float,
        comment: str | None = None,
    ):
        """
        记录用户反馈
        
        Args:
            run_id: LangSmith run ID
            key: 反馈类型（如 "correctness", "helpfulness"）
            score: 分数（0-1）
            comment: 评论
        """
        if not is_langsmith_enabled():
            return
        
        try:
            from langsmith import Client
            client = Client()
            client.create_feedback(
                run_id=run_id,
                key=key,
                score=score,
                comment=comment,
            )
            logger.debug(f"反馈已记录: run_id={run_id}, {key}={score}")
        except Exception as e:
            logger.warning(f"反馈记录失败: {e}")
    
    @staticmethod
    def get_project_stats(project_name: str) -> dict | None:
        """
        获取项目统计信息
        
        Returns:
            项目统计数据
        """
        if not is_langsmith_enabled():
            return None
        
        try:
            from langsmith import Client
            client = Client()
            # 获取项目信息
            projects = list(client.list_projects(project_name=project_name))
            if projects:
                return {
                    "name": projects[0].name,
                    "run_count": projects[0].run_count,
                }
            return None
        except Exception as e:
            logger.warning(f"获取项目统计失败: {e}")
            return None
