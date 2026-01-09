"""
与 Toolbench 集成的示例

演示如何从 Toolbench 的工具定义生成合约并执行
"""

import json
import os
from typing import Dict, Any

from toolgate.contracts import ContractGenerator
from toolgate.pipeline import ToolGatePipeline
from toolgate.core.tool_executor import ToolExecutor


def load_toolbench_tool(tool_path: str) -> Dict[str, Any]:
    """
    从 Toolbench 格式的工具 JSON 加载工具定义
    
    Args:
        tool_path: 工具 JSON 文件路径
    
    Returns:
        工具定义字典
    """
    with open(tool_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_toolbench_executor(tool_root_dir: str):
    """
    创建与 Toolbench 集成的工具执行器
    
    Args:
        tool_root_dir: Toolbench 工具根目录
    """
    def execute_tool(tool_name: str, params: dict, tool_config: dict = None):
        """
        执行工具（调用 Toolbench 的 API 服务）
        
        这里是一个示例实现，实际应该调用 Toolbench 的 server
        """
        # 实际实现应该：
        # 1. 根据 tool_name 找到对应的工具定义
        # 2. 调用 Toolbench 的 get_rapidapi_response 或类似函数
        # 3. 返回结果
        
        # 这里简化处理
        return {
            "status": "success",
            "data": params,
            "message": f"Tool {tool_name} executed via Toolbench"
        }
    
    return ToolExecutor(tool_executor_func=execute_tool)


def main():
    """主函数"""
    # 1. 加载 Toolbench 工具
    # 假设工具定义在 tool_root_dir 下
    tool_root_dir = "/path/to/toolbench/tools"  # 替换为实际路径
    
    # 2. 生成合约
    generator = ContractGenerator()
    tool_contracts = {}
    
    # 遍历工具目录，为每个工具生成合约
    # 这里简化处理，实际应该遍历所有工具
    # for category in os.listdir(tool_root_dir):
    #     category_path = os.path.join(tool_root_dir, category)
    #     if os.path.isdir(category_path):
    #         for tool_file in os.listdir(category_path):
    #             if tool_file.endswith('.json'):
    #                 tool_path = os.path.join(category_path, tool_file)
    #                 tool_json = load_toolbench_tool(tool_path)
    #                 contract = generator.generate_from_tool_json(tool_json)
    #                 tool_contracts[contract.tool_name] = contract
    
    # 3. 创建执行器
    executor = create_toolbench_executor(tool_root_dir)
    
    # 4. 创建管道
    pipeline = ToolGatePipeline(
        tool_contracts=tool_contracts,
        tool_executor=executor,
        max_steps=10
    )
    
    # 5. 运行查询
    query = "用户查询示例"
    result = pipeline.run(query=query)
    
    print("执行结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

