"""应用配置管理"""

from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """应用配置类，从环境变量读取配置"""

    # LLM 提供商选择: "openai" 或 "dashscope"
    llm_provider: Literal["openai", "dashscope"] = "dashscope"

    # OpenAI 配置（保留备用）
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None

    # 通义千问 DashScope 配置
    dashscope_api_key: str = ""
    dashscope_model: str = "qwen-plus"

    # LangSmith 配置
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "ecommerce-customer-service"

    # Redis 配置
    redis_url: str = "redis://localhost:6379"

    # 应用配置
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
