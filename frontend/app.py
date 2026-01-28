"""Streamlit 前端界面"""

import streamlit as st
import httpx

# 页面配置
st.set_page_config(
    page_title="智能电商客服",
    page_icon="🛒",
    layout="centered",
)

# API 配置
API_BASE_URL = "http://localhost:8000"


def init_session_state():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None


def send_message(user_input: str) -> dict:
    """
    发送消息到后端 API

    Args:
        user_input: 用户输入

    Returns:
        API 响应
    """
    # 构建请求数据
    request_data = {
        "message": user_input,
        "session_id": st.session_state.session_id,
        "chat_history": [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages
        ],
    }

    # 发送请求
    with httpx.Client(timeout=60.0) as client:
        response = client.post(f"{API_BASE_URL}/chat", json=request_data)
        response.raise_for_status()
        return response.json()


def main():
    """主函数"""
    init_session_state()

    # 页面标题
    st.title("🛒 智能电商客服")
    st.caption("基于 Multi-Agent 架构的智能客服系统")

    # 显示对话历史
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "agent_used" in msg:
                st.caption(f"🤖 由 {msg['agent_used']} 处理")

    # 用户输入
    if user_input := st.chat_input("请输入您的问题..."):
        # 添加用户消息到历史
        st.session_state.messages.append({"role": "user", "content": user_input})

        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(user_input)

        # 显示助手回复
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    response = send_message(user_input)

                    # 更新 session_id
                    st.session_state.session_id = response["session_id"]

                    # 显示回复
                    st.markdown(response["message"])
                    st.caption(f"🤖 由 {response['agent_used']} 处理")

                    # 保存到历史
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["message"],
                        "agent_used": response["agent_used"],
                    })

                except httpx.HTTPError as e:
                    st.error(f"请求失败: {e}")
                except Exception as e:
                    st.error(f"发生错误: {e}")

    # 侧边栏
    with st.sidebar:
        st.header("💡 使用提示")
        st.markdown("""
        你可以尝试问我：
        - 👋 "你好"
        - 🛍️ "有什么商品推荐？"
        - 📦 "我想查询订单"
        - 🔙 "我要退货"
        """)

        if st.button("🔄 清空对话"):
            st.session_state.messages = []
            st.session_state.session_id = None
            st.rerun()

        st.divider()
        st.caption("智能电商客服 v0.1.0")


if __name__ == "__main__":
    main()
