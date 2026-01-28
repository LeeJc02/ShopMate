"""
多模型支持模块

支持多个 LLM 提供商，自动故障转移：
1. 主模型（primary）：优先使用
2. 备用模型（fallback）：主模型失败时自动切换

使用方式：
```python
from src.core.llm_router import get_llm, LLMRouter

# 简单用法（自动选择可用模型）
llm = get_llm()

# 高级用法（手动切换）
router = LLMRouter()
llm = router.get_primary()
# 如果失败，自动切换
llm = router.switch_to_fallback()
```
"""

import logging
from typing import Literal
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置"""
    provider: str
    model: str
    api_key: str
    base_url: str | None = None
    temperature: float = 0.7
    
    @property
    def is_available(self) -> bool:
        """检查 API Key 是否配置"""
        return bool(self.api_key)


class LLMRouter:
    """
    LLM 路由器
    
    管理多个 LLM 提供商，支持自动故障转移
    """
    
    def __init__(self):
        # 配置所有支持的模型
        self.models = {
            "dashscope": ModelConfig(
                provider="dashscope",
                model=settings.dashscope_model,
                api_key=settings.dashscope_api_key,
            ),
            "openai": ModelConfig(
                provider="openai",
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            ),
        }
        
        # 当前活跃的提供商
        self._active_provider = settings.llm_provider
        self._fallback_used = False
        
        logger.info(f"LLMRouter 初始化: 主模型={self._active_provider}")
    
    @property
    def available_providers(self) -> list[str]:
        """获取所有可用的提供商"""
        return [name for name, cfg in self.models.items() if cfg.is_available]
    
    @property
    def active_provider(self) -> str:
        """当前活跃的提供商"""
        return self._active_provider
    
    @property
    def is_using_fallback(self) -> bool:
        """是否正在使用备用模型"""
        return self._fallback_used
    
    def _create_llm(self, provider: str) -> BaseChatModel:
        """创建 LLM 实例"""
        config = self.models.get(provider)
        if not config or not config.is_available:
            raise ValueError(f"Provider '{provider}' 不可用或未配置")
        
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.model,
                api_key=config.api_key,
                base_url=config.base_url,
                temperature=config.temperature,
            )
        elif provider == "dashscope":
            from langchain_community.chat_models import ChatTongyi
            return ChatTongyi(
                model=config.model,
                api_key=config.api_key,
                temperature=config.temperature,
            )
        else:
            raise ValueError(f"不支持的 provider: {provider}")
    
    def get_llm(self, provider: str | None = None) -> BaseChatModel:
        """
        获取 LLM 实例
        
        Args:
            provider: 指定提供商，为 None 则使用当前活跃的
        """
        target = provider or self._active_provider
        return self._create_llm(target)
    
    def get_primary(self) -> BaseChatModel:
        """获取主模型"""
        return self.get_llm(settings.llm_provider)
    
    def get_fallback(self) -> BaseChatModel | None:
        """获取备用模型"""
        # 找到第一个不是主模型且可用的提供商
        for provider, config in self.models.items():
            if provider != settings.llm_provider and config.is_available:
                return self.get_llm(provider)
        return None
    
    def switch_to_fallback(self) -> BaseChatModel | None:
        """切换到备用模型"""
        fallback = self.get_fallback()
        if fallback:
            for provider, config in self.models.items():
                if provider != settings.llm_provider and config.is_available:
                    self._active_provider = provider
                    self._fallback_used = True
                    logger.warning(f"已切换到备用模型: {provider}")
                    return fallback
        logger.error("没有可用的备用模型")
        return None
    
    def switch_to_primary(self) -> BaseChatModel:
        """切换回主模型"""
        self._active_provider = settings.llm_provider
        self._fallback_used = False
        logger.info(f"已切换回主模型: {settings.llm_provider}")
        return self.get_primary()
    
    def get_status(self) -> dict:
        """获取路由器状态"""
        return {
            "active_provider": self._active_provider,
            "using_fallback": self._fallback_used,
            "available_providers": self.available_providers,
            "models": {
                name: {
                    "model": cfg.model,
                    "available": cfg.is_available,
                }
                for name, cfg in self.models.items()
            },
        }


# ===== 全局实例 =====

_router_instance: LLMRouter | None = None


def get_llm_router() -> LLMRouter:
    """获取全局 LLM 路由器"""
    global _router_instance
    if _router_instance is None:
        _router_instance = LLMRouter()
    return _router_instance


def get_llm(provider: str | None = None) -> BaseChatModel:
    """
    获取 LLM 实例（便捷函数）
    
    这是对 chitchat_agent.get_llm 的增强版本，支持多模型
    """
    return get_llm_router().get_llm(provider)


def get_llm_with_fallback() -> BaseChatModel:
    """
    获取 LLM，失败时自动切换到备用模型
    
    Returns:
        可用的 LLM 实例
    """
    router = get_llm_router()
    
    try:
        return router.get_primary()
    except Exception as e:
        logger.warning(f"主模型获取失败: {e}，尝试备用模型")
        fallback = router.switch_to_fallback()
        if fallback:
            return fallback
        raise
