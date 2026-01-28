"""FastAPI 主应用入口"""

import uuid
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.api.schemas import ChatRequest, ChatResponse, ToolCallInfo, ToolResultInfo
from src.rag import KnowledgeRetriever
from src.graphs import CustomerServiceGraph
from src.memory import InMemorySessionManager
from src.tools import get_tool_agent, TOOL_SCHEMAS, ToolResult

# 导入新的核心模块
from src.core import (
    get_cache,
    get_ab_manager,
    get_llm_router,
    configure_langsmith,
)

# 配置 LangSmith（如果配置了 API Key）
configure_langsmith(project_name="ecommerce-chatbot")

# 创建 FastAPI 应用
app = FastAPI(
    title="智能电商客服 Multi-Agent 系统",
    description="基于 LangChain + LangGraph 的多 Agent 协作客服系统（生产级增强版）",
    version="0.5.0",
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 初始化知识库检索器
retriever = KnowledgeRetriever(
    persist_directory=PROJECT_ROOT / "chroma_data",
    collection_name="customer_service_kb",
)

# 初始化 LangGraph 工作流（标准模式）
customer_service_graph = CustomerServiceGraph(retriever=retriever)

# 初始化会话管理器
session_manager = InMemorySessionManager(max_history=20, ttl_minutes=60)

# 注册流式响应路由
from src.api.stream import router as stream_router
app.include_router(stream_router)


@app.get("/")
async def root():
    """根路径，返回 API 信息"""
    return {
        "name": "智能电商客服 Multi-Agent 系统",
        "version": "0.5.0",
        "status": "running",
        "modes": {
            "standard": "LangGraph Multi-Agent 工作流（内置模拟数据）",
            "tool_call": "Tool Call 透传模式（可插拔集成）",
            "stream": "流式响应模式 (SSE)",
        },
        "agents": ["Supervisor", "ProductAgent", "OrderAgent", "AfterSalesAgent", "ChitchatAgent"],
        "features": [
            "LangGraph 工作流",
            "LLM 智能路由",
            "会话管理",
            "Tool Call 透传",
            "流式响应 (SSE)",
            "响应缓存",
            "熔断机制",
            "多模型支持",
            "A/B 测试",
            "LangSmith 追踪",
        ],
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    处理用户聊天请求
    
    两种模式：
    - use_tools=False（默认）：使用 LangGraph Multi-Agent 工作流，内置模拟数据
    - use_tools=True：启用 Tool Call 透传模式，返回需要调用的工具给后端执行
    
    两轮对话流程（use_tools=True 时）：
    1. 后端发送用户消息 → 返回 tool_call
    2. 后端执行工具 → 发送 tool_results → 返回最终回复
    """
    session_id = request.session_id or str(uuid.uuid4())

    # 获取会话历史
    if request.chat_history:
        chat_history = [
            {"role": msg.role, "content": msg.content, "tool_call_id": msg.tool_call_id}
            for msg in request.chat_history
        ]
    else:
        chat_history = session_manager.get_chat_history(session_id)

    # ===== Tool Call 透传模式 =====
    if request.use_tools:
        tool_agent = get_tool_agent()
        
        # 转换 tool_results
        tool_results = None
        if request.tool_results:
            tool_results = [
                ToolResult(
                    tool_name=r.tool_name,
                    success=r.success,
                    data=r.data,
                    error=r.error,
                )
                for r in request.tool_results
            ]
        
        # 调用 ToolEnabledAgent
        result = tool_agent.chat(
            user_input=request.message,
            chat_history=chat_history,
            tool_results=tool_results,
        )
        
        # 保存消息到会话历史
        if request.message:
            session_manager.add_message(session_id, "user", request.message)
        
        if result["type"] == "response" and result["content"]:
            session_manager.add_message(session_id, "assistant", result["content"])
            return ChatResponse(
                type="response",
                message=result["content"],
                session_id=session_id,
                agent_used="ToolEnabledAgent",
                tool_calls=None,
            )
        else:
            # 返回 tool_call
            tool_calls = [
                ToolCallInfo(
                    tool_name=tc.tool_name,
                    tool_args=tc.tool_args,
                    reason=tc.reason,
                )
                for tc in result["tool_calls"]
            ] if result["tool_calls"] else None
            
            return ChatResponse(
                type="tool_call",
                message=result["content"],  # 可能有中间回复
                session_id=session_id,
                agent_used="ToolEnabledAgent",
                tool_calls=tool_calls,
            )
    
    # ===== 标准模式（LangGraph 工作流）=====
    result = customer_service_graph.invoke(
        user_input=request.message,
        chat_history=chat_history,
    )

    session_manager.add_message(session_id, "user", request.message)
    session_manager.add_message(session_id, "assistant", result["message"])

    return ChatResponse(
        type="response",
        message=result["message"],
        session_id=session_id,
        agent_used=result["agent_used"],
        tool_calls=None,
    )


@app.get("/tools")
async def list_tools():
    """
    获取所有可用工具的 Schema 定义
    
    后端可以根据这些 Schema 实现对应的工具执行逻辑
    """
    return {
        "tools": [
            {
                "name": name,
                "description": schema["description"],
                "parameters": schema["parameters"],
                "returns": schema.get("returns", {}),
            }
            for name, schema in TOOL_SCHEMAS.items()
        ]
    }


@app.get("/tools/{tool_name}")
async def get_tool_schema(tool_name: str):
    """获取指定工具的 Schema 定义"""
    if tool_name not in TOOL_SCHEMAS:
        return {"error": f"Tool '{tool_name}' not found"}
    
    schema = TOOL_SCHEMAS[tool_name]
    return {
        "name": tool_name,
        "description": schema["description"],
        "parameters": schema["parameters"],
        "returns": schema.get("returns", {}),
    }


@app.get("/knowledge/search")
async def search_knowledge(query: str, k: int = 3):
    """直接搜索知识库（调试用）"""
    docs = retriever.search(query, k=k)
    results = [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("filename", ""),
            "category": doc.metadata.get("category", ""),
        }
        for doc in docs
    ]
    return {"query": query, "results": results}


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """获取会话历史（调试用）"""
    history = session_manager.get_chat_history(session_id)
    return {"session_id": session_id, "history": history}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """删除会话（调试用）"""
    session_manager.delete_session(session_id)
    return {"message": f"Session {session_id} deleted"}


# ===== 系统状态端点 =====

@app.get("/system/status")
async def system_status():
    """获取系统状态"""
    cache = get_cache()
    ab_manager = get_ab_manager()
    llm_router = get_llm_router()
    
    return {
        "cache": cache.get_stats(),
        "llm_router": llm_router.get_status(),
        "ab_tests": ab_manager.get_all_experiments(),
    }


@app.get("/system/cache")
async def cache_stats():
    """获取缓存统计"""
    return get_cache().get_stats()


@app.post("/system/cache/clear")
async def clear_cache():
    """清空缓存"""
    get_cache().clear()
    return {"message": "缓存已清空"}


@app.get("/system/ab-tests")
async def ab_tests():
    """获取 A/B 测试配置"""
    return get_ab_manager().get_all_experiments()


@app.post("/system/ab-tests/{experiment_name}/update")
async def update_ab_test(experiment_name: str, variants: dict):
    """更新 A/B 测试流量分配"""
    try:
        get_ab_manager().update_traffic(experiment_name, variants)
        return {"message": f"实验 {experiment_name} 已更新"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
    )
