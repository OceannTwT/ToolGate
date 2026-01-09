"""
Toolbench 集成示例

演示如何使用 Toolbench 的工具执行器
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载 .env 文件
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_env_path = os.path.join(_project_root, ".env")

# 调试信息
_loaded_count = 0

try:
    from dotenv import load_dotenv
    if os.path.exists(_env_path):
        load_dotenv(dotenv_path=_env_path)
        _loaded_count = sum(1 for k in os.environ.keys() if k in ["OPENAI_API_KEY", "TOOLBENCH_API_KEY"])
        if _loaded_count > 0:
            print(f"✓ 使用 python-dotenv 加载了 .env 文件 ({_loaded_count} 个变量)")
except ImportError:
    # 如果没有安装 python-dotenv，手动读取 .env 文件
    if os.path.exists(_env_path):
        try:
            with open(_env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value
                            _loaded_count += 1
            if _loaded_count > 0:
                print(f"✓ 手动加载了 .env 文件 ({_loaded_count} 个变量)")
        except Exception as e:
            print(f"警告: 读取 .env 文件失败: {e}")
else:
    if not os.path.exists(_env_path):
        print(f"警告: .env 文件不存在: {_env_path}")

from toolgate.core.tool_executor import ToolExecutor
from toolgate.contracts import ToolContract, Precondition, Postcondition
from toolgate.pipeline import ToolGatePipeline
from toolgate.llm import GPTClient, ReActPromptTemplate


def create_toolbench_executor():
    """创建使用 Toolbench 接口的执行器"""
    
    # 检查 API Key（从环境变量或 .env 文件）
    toolbench_key = os.getenv("TOOLBENCH_API_KEY", "")
    if not toolbench_key:
        print("警告: 未设置 TOOLBENCH_API_KEY 环境变量")
        print("   请确保 .env 文件存在且包含 TOOLBENCH_API_KEY")
        print("   或者运行: export TOOLBENCH_API_KEY='your-key'")
    else:
        print(f"✓ 已加载 TOOLBENCH_API_KEY (长度: {len(toolbench_key)})")
    
    # 创建 Toolbench 执行器
    executor = ToolExecutor(
        use_toolbench=True,
        toolbench_config={
            "tools_root": "data.toolenv.tools",  # 或你的工具根目录
            "schema_root": "data/toolenv/response_examples",  # 或你的 schema 目录
            "rapidapi_key": toolbench_key,
            "toolbench_key": toolbench_key,
            "api_customization": False,
            "strip_method": "truncate"  # 或 "filter", "random"
        }
    )
    
    return executor


def example_toolbench_execution():
    """示例：使用 Toolbench 执行工具（带输入参数）"""
    
    executor = create_toolbench_executor()
    
    # 示例 1: 调用 Songkick concert API（需要 id_conc 参数）
    print("\n【示例 1】查询演唱会信息")
    print("-" * 70)
    print("工具: TheClique")
    print("API: Songkick concert")
    print("输入参数: id_conc = '40698227-lumineers-at-aware-super-theatre'")
    print("-" * 70)
    
    tool_config_1 = {
        "category": "Data",  # 工具类别
        "api_name": "Songkick concert",  # API 名称
        "tool_input": {
            "id_conc": "40698227-lumineers-at-aware-super-theatre"  # 必需参数
        }
    }
    
    result_1 = executor.execute(
        tool_name="TheClique",  # 工具名称
        params={},  # 参数（tool_input 已在 tool_config 中）
        tool_config=tool_config_1
    )
    
    print("\n执行结果:")
    if result_1.get('error'):
        print(f"  ❌ 错误: {result_1.get('error', '')}")
        print(f"  响应: {result_1.get('response', '')[:200]}...")
    else:
        print(f"  ✓ 成功")
        response = result_1.get('response', '')
        if len(response) > 500:
            print(f"  响应 (前500字符):\n  {response[:500]}...")
        else:
            print(f"  响应:\n  {response}")
    
    # 示例 2: 调用 Songkick artist API（需要 artist_id 参数）
    print("\n【示例 2】查询艺术家信息")
    print("-" * 70)
    print("工具: TheClique")
    print("API: Songkick artist")
    print("输入参数: artist_id = '520117-arctic-monkeys'")
    print("-" * 70)
    
    tool_config_2 = {
        "category": "Data",
        "api_name": "Songkick artist",
        "tool_input": {
            "artist_id": "520117-arctic-monkeys"  # 必需参数
        }
    }
    
    result_2 = executor.execute(
        tool_name="TheClique",
        params={},
        tool_config=tool_config_2
    )
    
    print("\n执行结果:")
    if result_2.get('error'):
        print(f"  ❌ 错误: {result_2.get('error', '')}")
        print(f"  响应: {result_2.get('response', '')[:200]}...")
    else:
        print(f"  ✓ 成功")
        response = result_2.get('response', '')
        if len(response) > 500:
            print(f"  响应 (前500字符):\n  {response[:500]}...")
        else:
            print(f"  响应:\n  {response}")
    
    # 示例 3: 使用 params 参数传递输入（另一种方式）
    print("\n【示例 3】使用 params 参数传递输入（替代方式）")
    print("-" * 70)
    print("工具: TheClique")
    print("API: Songkick concert")
    print("输入参数: 通过 params 传递 id_conc")
    print("-" * 70)
    
    tool_config_3 = {
        "category": "Data",
        "api_name": "Songkick concert",
        # tool_input 可以从 params 中获取（如果 tool_config 中没有指定）
    }
    
    result_3 = executor.execute(
        tool_name="TheClique",
        params={"id_conc": "40698227-lumineers-at-aware-super-theatre"},  # 通过 params 传递
        tool_config=tool_config_3
    )
    
    print("\n执行结果:")
    if result_3.get('error'):
        print(f"  ❌ 错误: {result_3.get('error', '')}")
        print(f"  响应: {result_3.get('response', '')[:200]}...")
    else:
        print(f"  ✓ 成功")
        response = result_3.get('response', '')
        if len(response) > 500:
            print(f"  响应 (前500字符):\n  {response[:500]}...")
        else:
            print(f"  响应:\n  {response}")
    
    return result_1, result_2, result_3


def example_pipeline_with_toolbench():
    """示例：在 Pipeline 中使用 Toolbench 执行器"""
    
    # 1. 创建 Toolbench 执行器
    executor = create_toolbench_executor()
    
    # 2. 创建工具合约（需要根据实际工具调整）
    contract = ToolContract(
        tool_name="example_tool",
        tool_description="示例工具",
        precondition=Precondition(constraints=[]),
        postcondition=Postcondition(
            result_constraints=['has_field("response")'],
            state_update_rules={"result": "result.response"}
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "required": []
        }
    )
    
    # 3. 创建 LLM 客户端
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        print("警告: 未设置 OPENAI_API_KEY")
        return
    
    llm_client = GPTClient(
        model="gpt-4o-mini",
        openai_key=openai_key
    )
    
    # 4. 创建 Pipeline
    pipeline = ToolGatePipeline(
        tool_contracts={"example_tool": contract},
        llm_client=llm_client,
        prompt_template=ReActPromptTemplate(),
        tool_executor=executor,
        max_steps=5
    )
    
    # 5. 运行（注意：需要在合约或工具配置中指定 category 和 api_name）
    # result = pipeline.run(query="用户查询")
    
    print("Pipeline 已创建，可以使用 Toolbench 工具")


if __name__ == "__main__":
    print("=" * 70)
    print("Toolbench 集成示例")
    print("=" * 70)
    
    # 示例 1: 直接执行工具（带输入参数）
    print("\n" + "=" * 70)
    print("1. 直接执行 Toolbench 工具（带输入参数）")
    print("=" * 70)
    try:
        results = example_toolbench_execution()
        print("\n✓ 所有示例执行完成")
    except Exception as e:
        print(f"\n✗ 执行错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n提示:")
        print("   - 请确保 StableToolBench 在正确路径")
        print("   - 请确保工具配置正确（category, tool_name, api_name）")
        print("   - 请确保输入参数格式正确")
    
    # 示例 2: 在 Pipeline 中使用
    print("\n" + "=" * 70)
    print("2. 在 Pipeline 中使用 Toolbench")
    print("=" * 70)
    try:
        example_pipeline_with_toolbench()
    except Exception as e:
        print(f"\n✗ 执行错误: {e}")
        import traceback
        traceback.print_exc()

