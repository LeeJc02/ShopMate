"""
流式响应 API

使用 Server-Sent Events (SSE) 实现流式输出，
让用户能够实时看到 AI 的响应过程。
"""

import asyncio
import json
import time
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.core import get_llm, get_cache, get_ab_manager, llm_circuit_breaker
from src.core.langsmith_integration import trace_function
from src.memory import InMemorySessionManager

router = APIRouter(prefix="/stream", tags=["流式响应"])

# 会话管理器（与 main.py 共享或独立）
session_manager = InMemorySessionManager(max_history=20, ttl_minutes=60)


class StreamChatRequest(BaseModel):
    """流式聊天请求"""
    message: str = Field(..., description="用户消息")
    session_id: str | None = Field(None, description="会话 ID")


async def generate_stream_response(
    message: str,
    session_id: str,
    chat_history: list[dict],
) -> AsyncGenerator[str, None]:
    """
    生成流式响应
    
    Yields:
        SSE 格式的数据块
    """
    start_time = time.time()
    full_response = ""
    
    try:
        # 获取 LLM（支持流式）
        llm = get_llm()
        
        # 构建消息
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        
        messages = [
            SystemMessage(content="你是一个专业的电商客服助手，请用中文回复。"),
        ]
        
        # 添加历史消息
        for msg in chat_history[-10:]:  # 限制历史长度
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        # 添加当前消息
        messages.append(HumanMessage(content=message))
        
        # 流式调用
        async for chunk in llm.astream(messages):
            if hasattr(chunk, "content") and chunk.content:
                content = chunk.content
                full_response += content
                
                # SSE 格式
                yield f"data: {json.dumps({'type': 'chunk', 'content': content}, ensure_ascii=False)}\n\n"
        
        # 发送完成信号
        latency = time.time() - start_time
        yield f"data: {json.dumps({'type': 'done', 'latency': round(latency, 2)}, ensure_ascii=False)}\n\n"
        
        # 保存到会话历史
        session_manager.add_message(session_id, "user", message)
        session_manager.add_message(session_id, "assistant", full_response)
        
        # 记录 A/B 测试结果
        ab_manager = get_ab_manager()
        ab_manager.record_result("llm_provider", session_id, {"latency": latency})
        
    except Exception as e:
        # 发送错误
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def stream_chat(request: StreamChatRequest):
    """
    流式聊天接口
    
    使用 SSE 实时返回响应内容
    
    响应格式：
    - `{"type": "chunk", "content": "..."}` - 内容块
    - `{"type": "done", "latency": 1.5}` - 完成
    - `{"type": "error", "message": "..."}` - 错误
    """
    import uuid
    
    session_id = request.session_id or str(uuid.uuid4())
    chat_history = session_manager.get_chat_history(session_id)
    
    return StreamingResponse(
        generate_stream_response(request.message, session_id, chat_history),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-ID": session_id,
        },
    )


@router.get("/test")
async def test_stream():
    """
    测试流式响应
    
    返回一个简单的计数流
    """
    async def generate():
        for i in range(5):
            yield f"data: {json.dumps({'count': i + 1})}\n\n"
            await asyncio.sleep(0.5)
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )
