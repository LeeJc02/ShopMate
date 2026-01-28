"""订单服务 Agent - 处理订单查询、物流追踪等"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from src.agents.chitchat_agent import get_llm


# 模拟订单数据库
MOCK_ORDERS = {
    "ORD20240001": {
        "order_id": "ORD20240001",
        "user_id": "U10001",
        "product": "iPhone 15 Pro Max 256GB 黑色钛金属",
        "price": 9999,
        "status": "已发货",
        "create_time": "2024-01-20 14:30:00",
        "logistics": {
            "company": "顺丰速运",
            "tracking_no": "SF1234567890",
            "status": "运输中",
            "current_location": "深圳转运中心",
            "expected_delivery": "2024-01-23",
            "history": [
                {"time": "2024-01-20 16:00", "status": "已揽收", "location": "广州"},
                {"time": "2024-01-21 08:00", "status": "运输中", "location": "广州转运中心"},
                {"time": "2024-01-21 18:00", "status": "运输中", "location": "深圳转运中心"},
            ]
        }
    },
    "ORD20240002": {
        "order_id": "ORD20240002",
        "user_id": "U10001",
        "product": "AirPods Pro 2",
        "price": 1899,
        "status": "待发货",
        "create_time": "2024-01-22 10:15:00",
        "logistics": None
    },
    "ORD20240003": {
        "order_id": "ORD20240003",
        "user_id": "U10002",
        "product": "MacBook Pro 14 M3 Pro",
        "price": 14999,
        "status": "已完成",
        "create_time": "2024-01-10 09:00:00",
        "logistics": {
            "company": "顺丰速运",
            "tracking_no": "SF0987654321",
            "status": "已签收",
            "current_location": "北京市朝阳区",
            "expected_delivery": "2024-01-12",
            "history": [
                {"time": "2024-01-10 10:00", "status": "已揽收", "location": "上海"},
                {"time": "2024-01-11 06:00", "status": "运输中", "location": "南京转运中心"},
                {"time": "2024-01-11 20:00", "status": "派送中", "location": "北京市朝阳区"},
                {"time": "2024-01-12 10:30", "status": "已签收", "location": "北京市朝阳区"},
            ]
        }
    },
}


class OrderAgent:
    """订单服务 Agent，处理订单查询和物流追踪"""

    SYSTEM_PROMPT = """你是一个专业的电商客服助手，专门负责处理订单相关的问题。

你的职责包括：
1. 查询订单状态和详情
2. 追踪物流信息
3. 解答订单相关疑问

订单信息：
{order_info}

回答要求：
- 如果用户提供了订单号，根据订单信息回答
- 如果用户没有提供订单号，引导用户提供
- 回答要准确、清晰
- 使用友好的语气

请用中文回复。"""

    def __init__(self):
        """初始化 OrderAgent"""
        self.llm = get_llm()

    def _get_order_info(self, message: str) -> str:
        """从消息中提取订单号并查询订单信息"""
        # 简单的订单号提取（查找 ORD 开头的字符串）
        import re
        order_pattern = r'ORD\d{8}'
        matches = re.findall(order_pattern, message.upper())
        
        if matches:
            order_id = matches[0]
            if order_id in MOCK_ORDERS:
                order = MOCK_ORDERS[order_id]
                info = f"""
订单号: {order['order_id']}
商品: {order['product']}
金额: ¥{order['price']}
状态: {order['status']}
下单时间: {order['create_time']}
"""
                if order['logistics']:
                    logistics = order['logistics']
                    info += f"""
物流公司: {logistics['company']}
运单号: {logistics['tracking_no']}
物流状态: {logistics['status']}
当前位置: {logistics['current_location']}
预计送达: {logistics['expected_delivery']}

物流轨迹:
"""
                    for record in logistics['history']:
                        info += f"  - {record['time']} | {record['status']} | {record['location']}\n"
                else:
                    info += "\n物流信息: 暂无（订单尚未发货）"
                
                return info
            else:
                return f"未找到订单号 {order_id} 的订单记录，请确认订单号是否正确。"
        
        # 没有找到订单号，返回可用的示例订单
        available_orders = ", ".join(MOCK_ORDERS.keys())
        return f"未检测到订单号。请提供您的订单号（示例订单号：{available_orders}）"

    def chat(self, user_input: str, chat_history: list | None = None) -> str:
        """
        处理用户输入并返回回复
        """
        if chat_history is None:
            chat_history = []

        # 获取订单信息
        order_info = self._get_order_info(user_input)

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
            "order_info": order_info,
            "input": user_input,
            "chat_history": formatted_history,
        })

        return response.content
