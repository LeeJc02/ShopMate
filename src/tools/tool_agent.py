"""
Tool-enabled Agent
支持 Tool Call 透传的 Agent 实现
"""

import json
from typing import Literal

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.agents.chitchat_agent import get_llm
from src.tools.schemas import TOOL_SCHEMAS, ToolCall, ToolResult, get_tool_schema_for_llm


class ToolEnabledAgent:
    """
    支持 Tool Call 透传的智能 Agent
    
    当需要调用外部工具时，返回 tool_call 给调用方执行
    调用方执行完成后，将结果回传继续对话
    """

    SYSTEM_PROMPT = """你是一个专业的电商客服助手。

你可以通过调用工具来获取实时信息。可用的工具包括：
- query_order: 查询订单详情
- query_logistics: 查询物流信息
- query_user_orders: 查询用户订单列表
- query_user_info: 查询用户信息
- query_product_stock: 查询商品库存
- query_product_price: 查询商品价格
- create_aftersales: 创建售后申请
- query_aftersales: 查询售后进度
- transfer_to_human: 转接人工客服

当你需要查询用户的订单、物流、账户等实时信息时，请调用相应的工具。
当用户未提供必要信息（如订单号、用户ID）时，请先询问获取。

请用中文回复。"""

    def __init__(self):
        """初始化 Tool-enabled Agent"""
        self.llm = get_llm()
        
        # 绑定工具到 LLM（如果 LLM 支持 function calling）
        self.tools = get_tool_schema_for_llm()
        
        # 尝试绑定工具（某些 LLM 可能不支持）
        try:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        except Exception:
            # 如果不支持 bind_tools，使用普通 LLM
            self.llm_with_tools = self.llm
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ])

    def chat(
        self,
        user_input: str,
        chat_history: list[dict] | None = None,
        tool_results: list[ToolResult] | None = None,
    ) -> dict:
        """
        处理用户输入，可能返回 tool_call 或最终回复
        
        Args:
            user_input: 用户输入（如果是继续对话，可以为空）
            chat_history: 对话历史
            tool_results: 上次返回的 tool_call 执行结果
            
        Returns:
            {
                "type": "response" | "tool_call",
                "content": "回复内容" | None,
                "tool_calls": [ToolCall] | None,
            }
        """
        if chat_history is None:
            chat_history = []
        
        # 构建消息列表
        messages = []
        
        # 添加历史消息
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "tool":
                # 工具返回结果
                messages.append(ToolMessage(
                    content=msg["content"],
                    tool_call_id=msg.get("tool_call_id", ""),
                ))
        
        # 添加工具执行结果（如果有）- 以用户消息格式传递
        # 注意：通义千问等模型不支持 ToolMessage，改为 HumanMessage 格式
        if tool_results:
            result_texts = []
            for result in tool_results:
                if result.success:
                    result_texts.append(
                        f"【工具执行结果】{result.tool_name} 执行成功：\n"
                        f"{json.dumps(result.data, ensure_ascii=False, indent=2)}"
                    )
                else:
                    result_texts.append(
                        f"【工具执行结果】{result.tool_name} 执行失败：{result.error}"
                    )
            messages.append(HumanMessage(content="\n\n".join(result_texts)))
        
        # 添加用户输入（如果有）
        if user_input:
            messages.append(HumanMessage(content=user_input))
        
        # 调用 LLM
        chain = self.prompt | self.llm_with_tools
        response = chain.invoke({"messages": messages})
        
        # 检查是否有 tool_call
        if hasattr(response, "tool_calls") and response.tool_calls:
            # LLM 请求调用工具
            tool_calls = []
            for tc in response.tool_calls:
                tool_calls.append(ToolCall(
                    tool_name=tc["name"],
                    tool_args=tc.get("args", {}),
                    reason=f"需要{TOOL_SCHEMAS.get(tc['name'], {}).get('description', tc['name'])}",
                ))
            
            return {
                "type": "tool_call",
                "content": response.content if response.content else None,
                "tool_calls": tool_calls,
            }
        else:
            # 直接返回回复
            return {
                "type": "response",
                "content": response.content,
                "tool_calls": None,
            }


# 单例实例
_tool_agent_instance = None


def get_tool_agent() -> ToolEnabledAgent:
    """获取 ToolEnabledAgent 单例"""
    global _tool_agent_instance
    if _tool_agent_instance is None:
        _tool_agent_instance = ToolEnabledAgent()
    return _tool_agent_instance
