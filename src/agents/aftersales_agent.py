"""售后服务 Agent - 处理退换货、投诉等售后问题"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from src.agents.chitchat_agent import get_llm
from src.rag import KnowledgeRetriever


class AfterSalesAgent:
    """售后服务 Agent，处理退换货、维修、投诉等问题"""

    SYSTEM_PROMPT = """你是一个专业的电商客服助手，专门负责处理售后服务相关的问题。

你的职责包括：
1. 处理退货、换货申请
2. 解答售后政策问题
3. 处理投诉和建议
4. 指导用户进行售后操作

售后政策参考信息：
{policy_info}

回答要求：
- 基于售后政策信息回答用户问题
- 对于退换货申请，引导用户按照正确流程操作
- 态度诚恳，积极解决用户问题
- 必要时可以表示会升级处理或转接人工客服

请用中文回复。"""

    def __init__(self, retriever: KnowledgeRetriever | None = None):
        """
        初始化 AfterSalesAgent
        
        Args:
            retriever: 知识库检索器（可选，用于检索售后政策）
        """
        self.llm = get_llm()
        self.retriever = retriever

    def _get_policy_info(self, message: str) -> str:
        """检索相关售后政策信息"""
        if self.retriever:
            docs = self.retriever.search(message, k=2)
            if docs:
                policy_texts = []
                for doc in docs:
                    source = doc.metadata.get("filename", "")
                    content = doc.page_content.strip()
                    policy_texts.append(f"【{source}】\n{content}")
                return "\n\n---\n\n".join(policy_texts)
        
        # 如果没有检索器或没有找到相关内容，返回基本政策
        return """
基本售后政策：
1. 七天无理由退货：签收后7天内，商品未拆封可申请退货
2. 质量问题：保修期内免费维修或换货
3. 退款时效：3-7个工作日原路返回
4. 客服热线：400-123-4567
"""

    def chat(self, user_input: str, chat_history: list | None = None) -> str:
        """
        处理用户输入并返回回复
        """
        if chat_history is None:
            chat_history = []

        # 获取售后政策信息
        policy_info = self._get_policy_info(user_input)

        # 构建 prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        chain = prompt | self.llm

        # 转换历史记录格式
        formatted_history = []
        for msg in chat_history:
            if msg["role"] == "user":
                formatted_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                formatted_history.append(AIMessage(content=msg["content"]))

        response = chain.invoke({
            "policy_info": policy_info,
            "input": user_input,
            "chat_history": formatted_history,
        })

        return response.content
