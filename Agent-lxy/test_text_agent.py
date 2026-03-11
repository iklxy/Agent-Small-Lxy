from agents.text_agent import text_agent

def test_query(query: str):
    print(f"\n❓ 用户提问: {query}")
    # 模拟输入状态
    inputs = {"messages": [("user", query)]}
    
    # 运行 Agent
    for s in text_agent.stream(inputs, stream_mode="values"):
        message = s["messages"][-1]
        if hasattr(message, "content") and message.content:
            print(f"AI 分身: {message.content}")

if __name__ == "__main__":
    # 测试你在 02_primary_school.md 中录入的内容
    test_query("李欣洋有哪些好朋友？")