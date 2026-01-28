"""闲聊 Agent - 处理非业务问题和兜底回复"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel

from src.config import settings


def get_llm() -> BaseChatModel:
    """
    根据配置获取 LLM 实例

    Returns:
        LLM 实例（OpenAI 或 通义千问）
    """
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0.7,
        )
    else:  # dashscope
        from langchain_community.chat_models import ChatTongyi
        return ChatTongyi(
            model=settings.dashscope_model,
            api_key=settings.dashscope_api_key,
            temperature=0.7,
        )


class ChitchatAgent:
    """闲聊 Agent，用于处理非业务相关的用户对话"""

    SYSTEM_PROMPT = """你是一个友好的电商客服助手。你的主要职责是：
1. 友好地与用户打招呼和闲聊
2. 引导用户描述他们的需求
3. 对于无法处理的问题，礼貌地告知用户

注意事项：
- 保持友好、专业的态度
- 回复要简洁明了
- 如果用户有明确的业务需求（商品咨询、订单查询、售后问题），提示他们具体说明

请用中文回复。"""

    def __init__(self):
        """初始化 ChitchatAgent"""
        self.llm = get_llm()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        self.chain = self.prompt | self.llm

    def chat(self, user_input: str, chat_history: list | None = None) -> str:
        """
        处理用户输入并返回回复

        Args:
            user_input: 用户输入的消息
            chat_history: 对话历史记录

        Returns:
            Agent 的回复内容
        """
        if chat_history is None:
            chat_history = []

        # 转换历史记录格式
        formatted_history = []
        for msg in chat_history:
            if msg["role"] == "user":
                formatted_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                formatted_history.append(AIMessage(content=msg["content"]))

        response = self.chain.invoke({
            "input": user_input,
            "chat_history": formatted_history,
        })

        return response.content
