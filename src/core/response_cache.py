"""
响应缓存模块

基于语义相似度的缓存，减少重复 LLM 调用：
1. 精确匹配缓存（query hash）
2. 语义相似度缓存（embedding 相似度）

使用方式：
```python
from src.core.response_cache import ResponseCache, get_cache

cache = get_cache()
# 查询缓存
cached = cache.get(query)
if cached:
    return cached

# 生成响应后存入缓存
response = llm.invoke(query)
cache.set(query, response)
```
"""

import hashlib
import time
from typing import Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """缓存条目"""
    query: str
    response: Any
    embedding: list[float] | None = None
    created_at: float = field(default_factory=time.time)
    hits: int = 0
    ttl: float = 3600.0  # 默认 1 小时过期


class ResponseCache:
    """
    响应缓存
    
    支持两种缓存模式：
    1. 精确匹配：基于 query hash
    2. 语义相似：基于 embedding 余弦相似度（需要 embeddings 模块）
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: float = 3600.0,
        similarity_threshold: float = 0.95,
        enable_semantic: bool = False,
    ):
        """
        初始化缓存
        
        Args:
            max_size: 最大缓存条目数
            ttl: 缓存过期时间（秒）
            similarity_threshold: 语义相似度阈值（0-1）
            enable_semantic: 是否启用语义缓存
        """
        self.max_size = max_size
        self.ttl = ttl
        self.similarity_threshold = similarity_threshold
        self.enable_semantic = enable_semantic
        
        # 精确匹配缓存（hash -> entry）
        self._exact_cache: dict[str, CacheEntry] = {}
        
        # 语义缓存（用于相似度匹配）
        self._semantic_cache: list[CacheEntry] = []
        
        # 统计
        self._stats = {
            "hits": 0,
            "misses": 0,
            "semantic_hits": 0,
        }
        
        # Embedding 函数（懒加载）
        self._embeddings = None
    
    def _get_hash(self, query: str) -> str:
        """计算 query hash"""
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]
    
    def _get_embedding(self, query: str) -> list[float] | None:
        """获取 query 的 embedding"""
        if not self.enable_semantic:
            return None
        
        if self._embeddings is None:
            try:
                from src.rag.embeddings import get_embeddings
                self._embeddings = get_embeddings()
            except Exception as e:
                logger.warning(f"无法加载 embeddings: {e}")
                self.enable_semantic = False
                return None
        
        try:
            return self._embeddings.embed_query(query)
        except Exception as e:
            logger.warning(f"Embedding 计算失败: {e}")
            return None
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
    
    def _cleanup_expired(self):
        """清理过期条目"""
        now = time.time()
        
        # 清理精确缓存
        expired_keys = [
            k for k, v in self._exact_cache.items()
            if now - v.created_at > v.ttl
        ]
        for k in expired_keys:
            del self._exact_cache[k]
        
        # 清理语义缓存
        self._semantic_cache = [
            e for e in self._semantic_cache
            if now - e.created_at <= e.ttl
        ]
    
    def _evict_lru(self):
        """LRU 淘汰"""
        if len(self._exact_cache) >= self.max_size:
            # 按命中次数和时间排序，淘汰最少使用的
            sorted_items = sorted(
                self._exact_cache.items(),
                key=lambda x: (x[1].hits, x[1].created_at),
            )
            # 淘汰 10%
            to_remove = max(1, len(sorted_items) // 10)
            for k, _ in sorted_items[:to_remove]:
                del self._exact_cache[k]
    
    def get(self, query: str) -> Any | None:
        """
        获取缓存的响应
        
        Args:
            query: 用户查询
            
        Returns:
            缓存的响应，未命中返回 None
        """
        self._cleanup_expired()
        
        # 1. 精确匹配
        query_hash = self._get_hash(query)
        if query_hash in self._exact_cache:
            entry = self._exact_cache[query_hash]
            entry.hits += 1
            self._stats["hits"] += 1
            logger.debug(f"缓存命中(精确): {query[:30]}...")
            return entry.response
        
        # 2. 语义匹配
        if self.enable_semantic and self._semantic_cache:
            query_embedding = self._get_embedding(query)
            if query_embedding:
                best_match = None
                best_score = 0.0
                
                for entry in self._semantic_cache:
                    if entry.embedding:
                        score = self._cosine_similarity(query_embedding, entry.embedding)
                        if score > best_score:
                            best_score = score
                            best_match = entry
                
                if best_match and best_score >= self.similarity_threshold:
                    best_match.hits += 1
                    self._stats["hits"] += 1
                    self._stats["semantic_hits"] += 1
                    logger.debug(f"缓存命中(语义): {query[:30]}... (相似度: {best_score:.2f})")
                    return best_match.response
        
        self._stats["misses"] += 1
        return None
    
    def set(self, query: str, response: Any, ttl: float | None = None):
        """
        设置缓存
        
        Args:
            query: 用户查询
            response: 响应内容
            ttl: 缓存过期时间（秒），为 None 使用默认值
        """
        self._evict_lru()
        
        query_hash = self._get_hash(query)
        entry = CacheEntry(
            query=query,
            response=response,
            ttl=ttl or self.ttl,
        )
        
        # 存入精确缓存
        self._exact_cache[query_hash] = entry
        
        # 存入语义缓存
        if self.enable_semantic:
            entry.embedding = self._get_embedding(query)
            if entry.embedding:
                self._semantic_cache.append(entry)
        
        logger.debug(f"缓存写入: {query[:30]}...")
    
    def invalidate(self, query: str):
        """使特定查询的缓存失效"""
        query_hash = self._get_hash(query)
        if query_hash in self._exact_cache:
            del self._exact_cache[query_hash]
    
    def clear(self):
        """清空所有缓存"""
        self._exact_cache.clear()
        self._semantic_cache.clear()
        logger.info("缓存已清空")
    
    def get_stats(self) -> dict:
        """获取缓存统计"""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        
        return {
            "size": len(self._exact_cache),
            "semantic_size": len(self._semantic_cache),
            "max_size": self.max_size,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "semantic_hits": self._stats["semantic_hits"],
            "hit_rate": round(hit_rate, 4),
        }


# ===== 全局缓存实例 =====

_cache_instance: ResponseCache | None = None


def get_cache() -> ResponseCache:
    """获取全局缓存实例"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = ResponseCache(
            max_size=1000,
            ttl=3600.0,
            enable_semantic=False,  # 默认禁用语义缓存（节省 embedding 调用）
        )
    return _cache_instance
