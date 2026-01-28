"""API 数据模型定义"""

from typing import Literal, Any
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """聊天消息模型"""
    role: str = Field(..., description="消息角色: user/assistant/tool")
    content: str = Field(..., description="消息内容")
    tool_call_id: str | None = Field(None, description="工具调用ID（role=tool时）")


class ToolCallInfo(BaseModel):
    """工具调用信息"""
    tool_name: str = Field(..., description="工具名称")
    tool_args: dict[str, Any] = Field(default_factory=dict, description="工具参数")
    reason: str = Field(default="", description="调用原因说明")


class ToolResultInfo(BaseModel):
    """工具执行结果"""
    tool_name: str = Field(..., description="工具名称")
    success: bool = Field(..., description="是否成功")
    data: dict[str, Any] | None = Field(None, description="返回数据")
    error: str | None = Field(None, description="错误信息")


class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str = Field(default="", description="用户输入的消息")
    session_id: str | None = Field(None, description="会话ID，用于保持上下文")
    chat_history: list[ChatMessage] = Field(default_factory=list, description="对话历史")
    tool_results: list[ToolResultInfo] | None = Field(None, description="工具执行结果（两轮对话时传入）")
    use_tools: bool = Field(default=False, description="是否启用 Tool Call 模式")


class ChatResponse(BaseModel):
    """聊天响应模型"""
    type: Literal["response", "tool_call"] = Field(default="response", description="响应类型")
    message: str | None = Field(None, description="助手回复的消息（type=response时）")
    session_id: str = Field(..., description="会话ID")
    agent_used: str = Field(..., description="处理该消息的 Agent 名称")
    tool_calls: list[ToolCallInfo] | None = Field(None, description="需要调用的工具列表（type=tool_call时）")


# ===== Tool Schema 查询接口 =====

class ToolSchemaResponse(BaseModel):
    """工具 Schema 响应"""
    name: str
    description: str
    parameters: dict
    returns: dict
