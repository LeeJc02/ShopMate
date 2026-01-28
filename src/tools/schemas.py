"""
Tool Schema 定义
定义客服系统可调用的外部工具，供正式后端实现
"""

from typing import TypedDict, Literal, Any
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Tool 调用请求"""
    tool_name: str = Field(..., description="工具名称")
    tool_args: dict[str, Any] = Field(default_factory=dict, description="工具参数")
    reason: str = Field(default="", description="调用原因说明")


class ToolResult(BaseModel):
    """Tool 执行结果"""
    tool_name: str = Field(..., description="工具名称")
    success: bool = Field(..., description="是否成功")
    data: dict[str, Any] | None = Field(default=None, description="返回数据")
    error: str | None = Field(default=None, description="错误信息")


# ============ Tool Schema 定义 ============
# 这些 Schema 定义了客服系统可以"请求调用"的工具
# 实际执行由正式后端完成

TOOL_SCHEMAS = {
    # ===== 订单相关工具 =====
    "query_order": {
        "name": "query_order",
        "description": "查询订单详情，包括订单状态、商品信息、支付信息等",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "订单号，如 ORD20240001",
                },
                "user_id": {
                    "type": "string",
                    "description": "用户ID（可选，用于验证订单归属）",
                },
            },
            "required": ["order_id"],
        },
        "returns": {
            "order_id": "订单号",
            "status": "订单状态(待支付/待发货/已发货/已完成/已取消)",
            "product_name": "商品名称",
            "price": "订单金额",
            "create_time": "下单时间",
            "pay_time": "支付时间（如有）",
        },
    },
    
    "query_logistics": {
        "name": "query_logistics",
        "description": "查询订单物流信息和配送轨迹",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "订单号",
                },
                "tracking_no": {
                    "type": "string",
                    "description": "运单号（可选，直接查询物流）",
                },
            },
            "required": ["order_id"],
        },
        "returns": {
            "logistics_company": "物流公司",
            "tracking_no": "运单号",
            "status": "物流状态",
            "current_location": "当前位置",
            "expected_delivery": "预计送达时间",
            "history": "物流轨迹列表",
        },
    },
    
    # ===== 用户相关工具 =====
    "query_user_orders": {
        "name": "query_user_orders",
        "description": "查询用户的订单列表",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "用户ID",
                },
                "status": {
                    "type": "string",
                    "description": "订单状态筛选（可选）",
                    "enum": ["all", "pending", "shipped", "completed", "cancelled"],
                },
                "limit": {
                    "type": "integer",
                    "description": "返回数量限制，默认10",
                },
            },
            "required": ["user_id"],
        },
        "returns": {
            "orders": "订单列表",
            "total": "总订单数",
        },
    },
    
    "query_user_info": {
        "name": "query_user_info",
        "description": "查询用户基本信息（会员等级、积分等）",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "用户ID",
                },
            },
            "required": ["user_id"],
        },
        "returns": {
            "user_id": "用户ID",
            "nickname": "昵称",
            "member_level": "会员等级",
            "points": "当前积分",
            "available_coupons": "可用优惠券数量",
        },
    },
    
    # ===== 商品相关工具 =====
    "query_product_stock": {
        "name": "query_product_stock",
        "description": "查询商品库存信息",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "商品ID",
                },
                "sku": {
                    "type": "string",
                    "description": "SKU（规格，如颜色+容量）",
                },
            },
            "required": ["product_id"],
        },
        "returns": {
            "product_id": "商品ID",
            "product_name": "商品名称",
            "in_stock": "是否有货",
            "stock_count": "库存数量",
            "expected_restock": "预计补货时间（如缺货）",
        },
    },
    
    "query_product_price": {
        "name": "query_product_price",
        "description": "查询商品实时价格（含促销）",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "商品ID",
                },
                "user_id": {
                    "type": "string",
                    "description": "用户ID（可选，用于计算会员价）",
                },
            },
            "required": ["product_id"],
        },
        "returns": {
            "original_price": "原价",
            "current_price": "当前售价",
            "member_price": "会员价（如有）",
            "promotions": "可用促销活动列表",
        },
    },
    
    # ===== 售后相关工具 =====
    "create_aftersales": {
        "name": "create_aftersales",
        "description": "创建售后申请（退货/换货/维修）",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "订单号",
                },
                "type": {
                    "type": "string",
                    "description": "售后类型",
                    "enum": ["refund", "exchange", "repair"],
                },
                "reason": {
                    "type": "string",
                    "description": "申请原因",
                },
            },
            "required": ["order_id", "type", "reason"],
        },
        "returns": {
            "aftersales_id": "售后单号",
            "status": "申请状态",
            "next_steps": "下一步操作说明",
        },
    },
    
    "query_aftersales": {
        "name": "query_aftersales",
        "description": "查询售后申请进度",
        "parameters": {
            "type": "object",
            "properties": {
                "aftersales_id": {
                    "type": "string",
                    "description": "售后单号",
                },
                "order_id": {
                    "type": "string",
                    "description": "订单号（二选一）",
                },
            },
            "required": [],
        },
        "returns": {
            "aftersales_id": "售后单号",
            "type": "售后类型",
            "status": "当前状态",
            "create_time": "申请时间",
            "timeline": "处理时间线",
        },
    },
    
    # ===== 系统工具 =====
    "transfer_to_human": {
        "name": "transfer_to_human",
        "description": "转接人工客服",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "转接原因",
                },
                "priority": {
                    "type": "string",
                    "description": "优先级",
                    "enum": ["normal", "urgent"],
                },
            },
            "required": ["reason"],
        },
        "returns": {
            "queue_position": "排队位置",
            "estimated_wait": "预计等待时间",
            "ticket_id": "服务单号",
        },
    },
}


def get_tool_schema_for_llm() -> list[dict]:
    """
    获取适用于 LLM function calling 的工具定义格式
    
    Returns:
        OpenAI function calling 格式的工具列表
    """
    tools = []
    for name, schema in TOOL_SCHEMAS.items():
        tools.append({
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["parameters"],
            },
        })
    return tools
