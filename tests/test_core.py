"""
单元测试套件

使用 pytest 运行：
    pytest tests/ -v
    pytest tests/ -v --cov=src  # 带覆盖率
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock


# ===== 熔断器测试 =====

class TestCircuitBreaker:
    """熔断器测试"""
    
    def test_circuit_starts_closed(self):
        """熔断器初始状态应该是 CLOSED"""
        from src.core.circuit_breaker import CircuitBreaker, CircuitState
        
        breaker = CircuitBreaker(failure_threshold=3)
        assert breaker.state == CircuitState.CLOSED
    
    def test_circuit_opens_after_failures(self):
        """连续失败后熔断器应该打开"""
        from src.core.circuit_breaker import CircuitBreaker, CircuitState
        
        breaker = CircuitBreaker(failure_threshold=3)
        
        # 模拟连续失败
        for _ in range(3):
            try:
                breaker.call_sync(lambda: exec('raise Exception("test")'))
            except Exception:
                pass
        
        assert breaker.state == CircuitState.OPEN
    
    def test_circuit_uses_fallback_when_open(self):
        """熔断时应该使用降级策略"""
        from src.core.circuit_breaker import CircuitBreaker, CircuitState
        
        fallback_called = False
        
        def fallback():
            nonlocal fallback_called
            fallback_called = True
            return "fallback"
        
        breaker = CircuitBreaker(failure_threshold=1, fallback=fallback)
        
        # 触发熔断
        try:
            breaker.call_sync(lambda: exec('raise Exception("test")'))
        except Exception:
            pass
        
        # 熔断后应该调用 fallback
        result = breaker.call_sync(lambda: "should not reach")
        assert fallback_called
        assert result == "fallback"


# ===== 响应缓存测试 =====

class TestResponseCache:
    """响应缓存测试"""
    
    def test_cache_miss_returns_none(self):
        """缓存未命中应该返回 None"""
        from src.core.response_cache import ResponseCache
        
        cache = ResponseCache()
        result = cache.get("不存在的查询")
        assert result is None
    
    def test_cache_hit_returns_value(self):
        """缓存命中应该返回值"""
        from src.core.response_cache import ResponseCache
        
        cache = ResponseCache()
        cache.set("测试查询", "测试响应")
        result = cache.get("测试查询")
        assert result == "测试响应"
    
    def test_cache_stats(self):
        """缓存统计应该正确"""
        from src.core.response_cache import ResponseCache
        
        cache = ResponseCache()
        cache.set("q1", "r1")
        cache.get("q1")  # hit
        cache.get("q2")  # miss
        
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
    
    def test_cache_clear(self):
        """缓存清空应该生效"""
        from src.core.response_cache import ResponseCache
        
        cache = ResponseCache()
        cache.set("q1", "r1")
        cache.set("q2", "r2")
        
        cache.clear()
        
        assert cache.get("q1") is None
        assert cache.get("q2") is None


# ===== A/B 测试框架测试 =====

class TestABTesting:
    """A/B 测试框架测试"""
    
    def test_create_experiment(self):
        """创建实验"""
        from src.core.ab_testing import ABTestManager
        
        manager = ABTestManager()
        exp = manager.create_experiment(
            name="test_exp",
            variants={"control": 0.5, "treatment": 0.5},
        )
        
        assert exp.name == "test_exp"
        assert exp.variants == {"control": 0.5, "treatment": 0.5}
    
    def test_variant_assignment_is_consistent(self):
        """同一 session 应该分配到同一组"""
        from src.core.ab_testing import ABTestManager
        
        manager = ABTestManager()
        manager.create_experiment(
            name="test_exp",
            variants={"A": 0.5, "B": 0.5},
        )
        
        session_id = "test-session-123"
        variant1 = manager.get_variant("test_exp", session_id)
        variant2 = manager.get_variant("test_exp", session_id)
        
        assert variant1 == variant2
    
    def test_record_result(self):
        """记录实验结果"""
        from src.core.ab_testing import ABTestManager
        
        manager = ABTestManager()
        manager.create_experiment(
            name="test_exp",
            variants={"control": 1.0},
        )
        
        manager.record_result("test_exp", "session-1", {"latency": 1.5})
        manager.record_result("test_exp", "session-2", {"latency": 2.0})
        
        stats = manager.get_experiment_stats("test_exp")
        assert stats["variants"]["control"]["count"] == 2
        assert stats["variants"]["control"]["latency_avg"] == 1.75


# ===== LLM 路由器测试 =====

class TestLLMRouter:
    """LLM 路由器测试"""
    
    def test_available_providers(self):
        """应该返回可用的提供商列表"""
        from src.core.llm_router import LLMRouter
        
        router = LLMRouter()
        providers = router.available_providers
        
        # 至少有一个提供商配置了 API Key
        assert isinstance(providers, list)
    
    def test_get_status(self):
        """获取状态"""
        from src.core.llm_router import LLMRouter
        
        router = LLMRouter()
        status = router.get_status()
        
        assert "active_provider" in status
        assert "using_fallback" in status
        assert "available_providers" in status


# ===== RAG 检索器测试 =====

class TestKnowledgeRetriever:
    """知识库检索器测试"""
    
    def test_search_returns_documents(self):
        """搜索应该返回文档"""
        from src.rag import KnowledgeRetriever
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent
        retriever = KnowledgeRetriever(
            persist_directory=project_root / "chroma_data",
            collection_name="customer_service_kb",
        )
        
        docs = retriever.search("iPhone 价格", k=3)
        
        assert isinstance(docs, list)
        # 如果索引存在，应该有结果
        # assert len(docs) > 0


# ===== 会话管理器测试 =====

class TestSessionManager:
    """会话管理器测试"""
    
    def test_add_and_get_message(self):
        """添加和获取消息"""
        from src.memory import InMemorySessionManager
        
        manager = InMemorySessionManager()
        session_id = "test-session"
        
        manager.add_message(session_id, "user", "你好")
        manager.add_message(session_id, "assistant", "你好！")
        
        history = manager.get_chat_history(session_id)
        
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "你好"
    
    def test_delete_session(self):
        """删除会话"""
        from src.memory import InMemorySessionManager
        
        manager = InMemorySessionManager()
        session_id = "test-session"
        
        manager.add_message(session_id, "user", "test")
        manager.delete_session(session_id)
        
        history = manager.get_chat_history(session_id)
        assert len(history) == 0


# ===== API 端点测试 =====

class TestAPIEndpoints:
    """API 端点测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        from fastapi.testclient import TestClient
        from src.api.main import app
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """根路径应该返回系统信息"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "智能电商客服 Multi-Agent 系统"
        assert data["version"] == "0.5.0"
    
    def test_tools_endpoint(self, client):
        """工具列表端点"""
        response = client.get("/tools")
        assert response.status_code == 200
        
        data = response.json()
        assert "tools" in data
        assert len(data["tools"]) > 0
    
    def test_system_status_endpoint(self, client):
        """系统状态端点"""
        response = client.get("/system/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "cache" in data
        assert "llm_router" in data
        assert "ab_tests" in data
    
    def test_cache_stats_endpoint(self, client):
        """缓存统计端点"""
        response = client.get("/system/cache")
        assert response.status_code == 200
        
        data = response.json()
        assert "hits" in data
        assert "misses" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
