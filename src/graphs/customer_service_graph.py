"""
客服系统 Multi-Agent 工作流
使用 LangGraph 构建 Supervisor 模式的多 Agent 协作系统
"""

from typing import TypedDict, Literal, Annotated
import operator

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END

from src.agents.chitchat_agent import get_llm
from src.agents import ChitchatAgent, ProductAgent
from src.agents.order_agent import OrderAgent
from src.agents.aftersales_agent import AfterSalesAgent
from src.rag import KnowledgeRetriever


# 定义 Agent 类型
AgentType = Literal["ProductAgent", "OrderAgent", "AfterSalesAgent", "ChitchatAgent"]


class AgentState(TypedDict):
    """Agent 状态定义"""
    # 用户原始输入
    user_input: str
    # 对话历史
    chat_history: list[dict]
    # 当前选择的 Agent
    current_agent: AgentType
    # Agent 的回复
    agent_response: str
    # 是否需要继续处理
    should_continue: bool


class SupervisorAgent:
    """
    Supervisor Agent - 使用 LLM 进行智能意图识别和路由
    """
    
    ROUTING_PROMPT = """你是一个智能客服系统的路由器，需要根据用户的输入判断应该由哪个专业 Agent 来处理。

可用的 Agent：
1. ProductAgent - 处理商品咨询、价格查询、商品推荐、促销活动、优惠政策
2. OrderAgent - 处理订单查询、物流追踪、订单状态
3. AfterSalesAgent - 处理退货、换货、退款、投诉、维修、售后政策
4. ChitchatAgent - 处理闲聊、问候、无法分类的问题

用户输入: {user_input}

请只输出一个 Agent 名称（ProductAgent/OrderAgent/AfterSalesAgent/ChitchatAgent），不要输出其他内容。"""

    def __init__(self):
        self.llm = get_llm()
    
    def route(self, user_input: str) -> AgentType:
        """
        根据用户输入路由到对应的 Agent
        
        Args:
            user_input: 用户输入
            
        Returns:
            Agent 名称
        """
        # 使用 LLM 进行意图识别
        response = self.llm.invoke(
            self.ROUTING_PROMPT.format(user_input=user_input)
        )
        
        agent_name = response.content.strip()
        
        # 验证返回的 Agent 名称
        valid_agents = ["ProductAgent", "OrderAgent", "AfterSalesAgent", "ChitchatAgent"]
        
        for valid_agent in valid_agents:
            if valid_agent.lower() in agent_name.lower():
                return valid_agent
        
        # 默认返回 ChitchatAgent
        return "ChitchatAgent"


class CustomerServiceGraph:
    """
    客服系统 Multi-Agent 工作流
    使用 LangGraph 实现 Supervisor 模式
    """
    
    def __init__(self, retriever: KnowledgeRetriever):
        """
        初始化客服系统工作流
        
        Args:
            retriever: 知识库检索器
        """
        # 初始化各个 Agent
        self.supervisor = SupervisorAgent()
        self.chitchat_agent = ChitchatAgent()
        self.product_agent = ProductAgent(retriever=retriever)
        self.order_agent = OrderAgent()
        self.aftersales_agent = AfterSalesAgent(retriever=retriever)
        
        # 构建工作流图
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        
        # 创建状态图
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("product_agent", self._product_agent_node)
        workflow.add_node("order_agent", self._order_agent_node)
        workflow.add_node("aftersales_agent", self._aftersales_agent_node)
        workflow.add_node("chitchat_agent", self._chitchat_agent_node)
        
        # 设置入口点
        workflow.set_entry_point("supervisor")
        
        # 添加条件边
        workflow.add_conditional_edges(
            "supervisor",
            self._route_to_agent,
            {
                "ProductAgent": "product_agent",
                "OrderAgent": "order_agent",
                "AfterSalesAgent": "aftersales_agent",
                "ChitchatAgent": "chitchat_agent",
            }
        )
        
        # 所有 Agent 完成后结束
        workflow.add_edge("product_agent", END)
        workflow.add_edge("order_agent", END)
        workflow.add_edge("aftersales_agent", END)
        workflow.add_edge("chitchat_agent", END)
        
        return workflow.compile()
    
    def _supervisor_node(self, state: AgentState) -> AgentState:
        """Supervisor 节点：进行意图识别和路由"""
        agent_type = self.supervisor.route(state["user_input"])
        return {**state, "current_agent": agent_type}
    
    def _route_to_agent(self, state: AgentState) -> AgentType:
        """路由函数：返回下一个要执行的 Agent"""
        return state["current_agent"]
    
    def _product_agent_node(self, state: AgentState) -> AgentState:
        """ProductAgent 节点"""
        response = self.product_agent.chat(
            user_input=state["user_input"],
            chat_history=state["chat_history"],
        )
        return {**state, "agent_response": response}
    
    def _order_agent_node(self, state: AgentState) -> AgentState:
        """OrderAgent 节点"""
        response = self.order_agent.chat(
            user_input=state["user_input"],
            chat_history=state["chat_history"],
        )
        return {**state, "agent_response": response}
    
    def _aftersales_agent_node(self, state: AgentState) -> AgentState:
        """AfterSalesAgent 节点"""
        response = self.aftersales_agent.chat(
            user_input=state["user_input"],
            chat_history=state["chat_history"],
        )
        return {**state, "agent_response": response}
    
    def _chitchat_agent_node(self, state: AgentState) -> AgentState:
        """ChitchatAgent 节点"""
        response = self.chitchat_agent.chat(
            user_input=state["user_input"],
            chat_history=state["chat_history"],
        )
        return {**state, "agent_response": response}
    
    def invoke(self, user_input: str, chat_history: list[dict] | None = None) -> dict:
        """
        执行工作流
        
        Args:
            user_input: 用户输入
            chat_history: 对话历史
            
        Returns:
            包含回复和使用的 Agent 的字典
        """
        if chat_history is None:
            chat_history = []
        
        # 初始状态
        initial_state: AgentState = {
            "user_input": user_input,
            "chat_history": chat_history,
            "current_agent": "ChitchatAgent",
            "agent_response": "",
            "should_continue": True,
        }
        
        # 执行工作流
        result = self.graph.invoke(initial_state)
        
        return {
            "message": result["agent_response"],
            "agent_used": result["current_agent"],
        }
