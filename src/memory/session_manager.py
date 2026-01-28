"""
会话状态管理
支持内存存储和 Redis 持久化存储
"""

import json
from abc import ABC, abstractmethod
from typing import Any
from datetime import datetime, timedelta

from src.config import settings


class SessionManager(ABC):
    """会话管理器抽象基类"""
    
    @abstractmethod
    def get_session(self, session_id: str) -> dict | None:
        """获取会话数据"""
        pass
    
    @abstractmethod
    def save_session(self, session_id: str, data: dict) -> None:
        """保存会话数据"""
        pass
    
    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """删除会话"""
        pass
    
    @abstractmethod
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """添加消息到会话历史"""
        pass
    
    @abstractmethod
    def get_chat_history(self, session_id: str) -> list[dict]:
        """获取会话的聊天历史"""
        pass


class InMemorySessionManager(SessionManager):
    """
    基于内存的会话管理器
    适用于开发和测试环境
    """
    
    def __init__(self, max_history: int = 20, ttl_minutes: int = 60):
        """
        初始化内存会话管理器
        
        Args:
            max_history: 保留的最大历史消息数
            ttl_minutes: 会话过期时间（分钟）
        """
        self._sessions: dict[str, dict] = {}
        self.max_history = max_history
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def _is_expired(self, session: dict) -> bool:
        """检查会话是否过期"""
        last_activity = session.get("last_activity")
        if last_activity:
            last_time = datetime.fromisoformat(last_activity)
            return datetime.now() - last_time > self.ttl
        return True
    
    def _clean_expired_sessions(self) -> None:
        """清理过期会话"""
        expired = [
            sid for sid, sess in self._sessions.items()
            if self._is_expired(sess)
        ]
        for sid in expired:
            del self._sessions[sid]
    
    def get_session(self, session_id: str) -> dict | None:
        """获取会话数据"""
        self._clean_expired_sessions()
        
        session = self._sessions.get(session_id)
        if session and not self._is_expired(session):
            return session
        return None
    
    def save_session(self, session_id: str, data: dict) -> None:
        """保存会话数据"""
        data["last_activity"] = datetime.now().isoformat()
        self._sessions[session_id] = data
    
    def delete_session(self, session_id: str) -> None:
        """删除会话"""
        if session_id in self._sessions:
            del self._sessions[session_id]
    
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """添加消息到会话历史"""
        session = self.get_session(session_id) or {
            "chat_history": [],
            "created_at": datetime.now().isoformat(),
        }
        
        # 添加消息
        session["chat_history"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        
        # 限制历史长度
        if len(session["chat_history"]) > self.max_history:
            session["chat_history"] = session["chat_history"][-self.max_history:]
        
        self.save_session(session_id, session)
    
    def get_chat_history(self, session_id: str) -> list[dict]:
        """获取会话的聊天历史"""
        session = self.get_session(session_id)
        if session:
            return session.get("chat_history", [])
        return []


class RedisSessionManager(SessionManager):
    """
    基于 Redis 的会话管理器
    适用于生产环境，支持分布式部署
    """
    
    def __init__(
        self,
        redis_url: str | None = None,
        max_history: int = 20,
        ttl_seconds: int = 3600,
    ):
        """
        初始化 Redis 会话管理器
        
        Args:
            redis_url: Redis 连接 URL
            max_history: 保留的最大历史消息数
            ttl_seconds: 会话过期时间（秒）
        """
        import redis
        
        self.redis_url = redis_url or settings.redis_url
        self.client = redis.from_url(self.redis_url, decode_responses=True)
        self.max_history = max_history
        self.ttl = ttl_seconds
        self.prefix = "session:"
    
    def _get_key(self, session_id: str) -> str:
        """获取 Redis key"""
        return f"{self.prefix}{session_id}"
    
    def get_session(self, session_id: str) -> dict | None:
        """获取会话数据"""
        key = self._get_key(session_id)
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return None
    
    def save_session(self, session_id: str, data: dict) -> None:
        """保存会话数据"""
        key = self._get_key(session_id)
        data["last_activity"] = datetime.now().isoformat()
        self.client.setex(key, self.ttl, json.dumps(data, ensure_ascii=False))
    
    def delete_session(self, session_id: str) -> None:
        """删除会话"""
        key = self._get_key(session_id)
        self.client.delete(key)
    
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """添加消息到会话历史"""
        session = self.get_session(session_id) or {
            "chat_history": [],
            "created_at": datetime.now().isoformat(),
        }
        
        # 添加消息
        session["chat_history"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        
        # 限制历史长度
        if len(session["chat_history"]) > self.max_history:
            session["chat_history"] = session["chat_history"][-self.max_history:]
        
        self.save_session(session_id, session)
    
    def get_chat_history(self, session_id: str) -> list[dict]:
        """获取会话的聊天历史"""
        session = self.get_session(session_id)
        if session:
            return session.get("chat_history", [])
        return []


def get_session_manager() -> SessionManager:
    """
    获取会话管理器实例
    
    Returns:
        SessionManager 实例
    """
    # 目前使用内存存储，可以通过配置切换到 Redis
    # TODO: 根据配置自动选择
    return InMemorySessionManager()
