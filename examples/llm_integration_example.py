"""
LLM 集成示例

演示如何使用 GPT 客户端和 prompt 模板进行工具调用
"""

import os
from toolgate.core import SymbolicState, StateType
from toolgate.contracts import ToolContract, Precondition, Postcondition
from toolgate.pipeline import ToolGatePipeline
from toolgate.llm import GPTClient, ReActPromptTemplate
from toolgate.core.tool_executor import ToolExecutor


# 定义工具合约
def create_movie_search_contract():
    """创建电影搜索工具的合约"""
    return ToolContract(
        tool_name="SearchMovieShowtimes",
        tool_description="根据城市和电影名称搜索电影场次信息",
        precondition=Precondition(
            constraints=["exists(city)", "exists(movie)"]
        ),
        postcondition=Postcondition(
            result_constraints=[
                'has_field("showtimes")',
                'has_field("cinemas")'
            ],
            state_update_rules={
                "showtimes": "result.showtimes",
                "cinemas": "result.cinemas"
            },
            semantic_constraints=[]
        ),
        input_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "movie": {"type": "string"}
            },
            "required": ["city", "movie"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "showtimes": {"type": "array"},
                "cinemas": {"type": "array"}
            }
        }
    )


# 模拟工具执行器
def mock_tool_executor(tool_name: str, params: dict, tool_config: dict = None):
    """模拟工具执行"""
    if tool_name == "SearchMovieShowtimes":
        city = params.get("city", "")
        movie = params.get("movie", "")
        return {
            "showtimes": [
                {"id": "st1", "time": "14:00", "cinema": f"{city}影城A", "movie": movie},
                {"id": "st2", "time": "18:00", "cinema": f"{city}影城B", "movie": movie}
            ],
            "cinemas": [f"{city}影城A", f"{city}影城B"]
        }
    else:
        return {"error": f"Unknown tool: {tool_name}"}


def main():
    """主函数"""
    # 1. 检查 OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        print("警告: 未设置 OPENAI_API_KEY 环境变量")
        print("请设置: export OPENAI_API_KEY='your-key'")
        return
    
    # 2. 创建工具合约
    contracts = {
        "SearchMovieShowtimes": create_movie_search_contract()
    }
    
    # 3. 创建 LLM 客户端
    llm_client = GPTClient(
        model="gpt-4o-mini",
        openai_key=openai_key
    )
    
    # 4. 创建 Prompt 模板
    prompt_template = ReActPromptTemplate()
    
    # 5. 创建工具执行器
    executor = ToolExecutor(tool_executor_func=mock_tool_executor)
    
    # 6. 创建管道
    pipeline = ToolGatePipeline(
        tool_contracts=contracts,
        llm_client=llm_client,
        prompt_template=prompt_template,
        tool_executor=executor,
        max_steps=5
    )
    
    # 7. 运行查询
    query = "我想看《疯狂动物城2》在上海的场次"
    initial_entities = {
        "city": ("上海", "City"),
        "movie": ("疯狂动物城2", "Movie")
    }
    
    print("=" * 60)
    print("查询:", query)
    print("=" * 60)
    print("\n开始执行...\n")
    
    result = pipeline.run(
        query=query,
        initial_entities=initial_entities
    )
    
    # 8. 输出结果
    print("\n" + "=" * 60)
    print("最终答案:")
    print("=" * 60)
    print(result["answer"])
    
    print("\n" + "=" * 60)
    print("最终状态:")
    print("=" * 60)
    import json
    print(json.dumps(result["final_state"], indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 60)
    print("执行轨迹:")
    print("=" * 60)
    for step in result["trace"]:
        print(f"\n步骤 {step['step']}:")
        print(f"  动作类型: {step.get('action_type', 'N/A')}")
        if step.get("tool_called"):
            print(f"  调用工具: {step['tool_called']}")
            print(f"  验证结果: {step.get('verification_result', {})}")
            print(f"  状态更新: {step.get('state_updated', False)}")
    
    print("\n" + "=" * 60)
    print("对话历史:")
    print("=" * 60)
    for i, msg in enumerate(result.get("conversation_history", [])):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:200]  # 截断显示
        print(f"\n[{i}] {role}:")
        print(content)


if __name__ == "__main__":
    main()

