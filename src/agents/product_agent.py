"""商品咨询 Agent - 使用 RAG 检索知识库回答商品相关问题"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.config import settings
from src.agents.chitchat_agent import get_llm
from src.rag import KnowledgeRetriever


class ProductAgent:
    """商品咨询 Agent，使用 RAG 从知识库检索信息回答用户问题"""

    SYSTEM_PROMPT = """你是一个专业的电商客服助手，专门负责回答商品相关的咨询。

你的职责包括：
1. 回答商品规格、价格、功能等问题
2. 根据用户需求推荐合适的商品
3. 解答促销活动、优惠政策相关问题
4. 解答售后政策、物流配送相关问题

回答要求：
- 基于提供的知识库信息进行回答
- 如果知识库中没有相关信息，诚实告知用户并建议联系人工客服
- 回答要准确、专业、简洁
- 使用友好的语气

以下是从知识库中检索到的相关信息：
{context}

请用中文回复。"""

    def __init__(self, retriever: KnowledgeRetriever):
        """
        初始化 ProductAgent
        
        Args:
            retriever: 知识库检索器
        """
        self.llm = get_llm()
        self.retriever = retriever
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        # 构建 RAG Chain
        self.chain = (
            {
                "context": lambda x: self._format_docs(
                    self.retriever.search(x["input"], k=3)
                ),
                "chat_history": lambda x: x.get("chat_history", []),
                "input": lambda x: x["input"],
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _format_docs(self, docs) -> str:
        """格式化检索到的文档"""
        if not docs:
            return "未找到相关信息。"
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("filename", "未知来源")
            category = doc.metadata.get("category", "其他")
            content = doc.page_content.strip()
            formatted.append(f"【{category}】({source})\n{content}")
        
        return "\n\n---\n\n".join(formatted)
    
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
        
        return response
    
    def search_knowledge(self, query: str, k: int = 3) -> list[dict]:
        """
        仅检索知识库，不生成回复
        
        Args:
            query: 查询文本
            k: 返回文档数量
            
        Returns:
            检索结果列表
        """
        docs = self.retriever.search(query, k=k)
        
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("filename", ""),
                "category": doc.metadata.get("category", ""),
            })
        
        return results
