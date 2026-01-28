"""Embeddings 模块 - 支持多种 Embedding 模型"""

from langchain_core.embeddings import Embeddings

from src.config import settings


def get_embeddings() -> Embeddings:
    """
    根据配置获取 Embeddings 实例
    
    Returns:
        Embeddings 实例
    """
    if settings.llm_provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model="text-embedding-3-small",
        )
    else:  # dashscope
        from langchain_community.embeddings import DashScopeEmbeddings
        return DashScopeEmbeddings(
            model="text-embedding-v3",
            dashscope_api_key=settings.dashscope_api_key,
        )
