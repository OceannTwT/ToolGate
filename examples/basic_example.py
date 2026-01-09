"""
基本使用示例

演示如何使用 ToolGate 进行工具调用
"""

from toolgate.core import SymbolicState, StateType
from toolgate.contracts import ToolContract, Precondition, Postcondition
from toolgate.pipeline import ToolGatePipeline
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


def create_ticket_booking_contract():
    """创建订票工具的合约"""
    return ToolContract(
        tool_name="BookMovieTicket",
        tool_description="预订电影票",
        precondition=Precondition(
            constraints=[
                "exists(showtimes)",
                "exists(selected_showtime)"
            ]
        ),
        postcondition=Postcondition(
            result_constraints=['has_field("order_id")'],
            state_update_rules={
                "order": "result.order_id",
                "ticket": "result.ticket_info"
            }
        ),
        input_schema={
            "type": "object",
            "properties": {
                "showtime_id": {"type": "string"},
                "seat_count": {"type": "integer"}
            },
            "required": ["showtime_id"]
        }
    )


# 模拟工具执行器
def mock_tool_executor(tool_name: str, params: dict, tool_config: dict = None):
    """模拟工具执行"""
    if tool_name == "SearchMovieShowtimes":
        return {
            "showtimes": [
                {"id": "st1", "time": "14:00", "cinema": "影城A"},
                {"id": "st2", "time": "18:00", "cinema": "影城B"}
            ],
            "cinemas": ["影城A", "影城B"]
        }
    elif tool_name == "BookMovieTicket":
        return {
            "order_id": "order_123",
            "ticket_info": {"seat": "A1", "price": 50}
        }
    else:
        return {"error": f"Unknown tool: {tool_name}"}


def main():
    """主函数"""
    # 1. 创建工具合约
    contracts = {
        "SearchMovieShowtimes": create_movie_search_contract(),
        "BookMovieTicket": create_ticket_booking_contract()
    }
    
    # 2. 创建工具执行器
    executor = ToolExecutor(tool_executor_func=mock_tool_executor)
    
    # 3. 创建管道
    pipeline = ToolGatePipeline(
        tool_contracts=contracts,
        tool_executor=executor,
        max_steps=5
    )
    
    # 4. 运行查询
    query = "我想看《疯狂动物城2》在上海的场次"
    initial_entities = {
        "city": ("上海", "City"),
        "movie": ("疯狂动物城2", "Movie")
    }
    
    result = pipeline.run(
        query=query,
        initial_entities=initial_entities
    )
    
    # 5. 输出结果
    print("=" * 50)
    print("查询:", query)
    print("=" * 50)
    print("\n最终答案:")
    print(result["answer"])
    print("\n最终状态:")
    print(result["final_state"])
    print("\n执行轨迹:")
    for step in result["trace"]:
        print(f"  步骤 {step['step']}: {step.get('action_type', 'N/A')}")
        if step.get("tool_called"):
            print(f"    调用工具: {step['tool_called']}")
            print(f"    验证结果: {step.get('verification_result', {})}")
            print(f"    状态更新: {step.get('state_updated', False)}")


if __name__ == "__main__":
    main()

