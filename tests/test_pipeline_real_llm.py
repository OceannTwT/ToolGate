"""
Pipeline 真实 LLM 测试

使用真实的 LLM API 进行测试（需要 OpenAI API Key）
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    # 加载项目根目录的 .env 文件
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(dotenv_path=env_path)
except ImportError:
    # 如果没有安装 python-dotenv，使用项目提供的工具
    try:
        from toolgate.utils.load_env import load_env_file
        load_env_file()
    except ImportError:
        # 如果都不可用，手动读取 .env 文件
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value

from toolgate.contracts import ToolContract, Precondition, Postcondition
from toolgate.pipeline import ToolGatePipeline
from toolgate.core.tool_executor import ToolExecutor
from toolgate.llm import GPTClient, ReActPromptTemplate
import json
import random


def load_toolbench_tools(tool_json_path, num_tools=10, tool_filter=None, random_seed=None):
    """
    从 Toolbench 的工具 JSON 文件中加载工具
    
    Args:
        tool_json_path: Toolbench 工具 JSON 文件路径
        num_tools: 要加载的工具数量（如果为 None，则加载所有工具）
        tool_filter: 可选的过滤器函数，用于筛选特定工具
                   函数签名: filter_func(tool_info) -> bool
        random_seed: 随机种子，用于可重复的随机选择（如果为 None，则使用系统随机）
    
    Returns:
        工具列表，每个工具包含 category_name, tool_name, api_name 等信息
    """
    try:
        with open(tool_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tools = []
        for item in data:
            if 'api_list' in item:
                for api in item['api_list']:
                    tool_info = {
                        'category_name': api.get('category_name', ''),
                        'tool_name': api.get('tool_name', ''),
                        'api_name': api.get('api_name', ''),
                        'api_description': api.get('api_description', ''),
                        'required_parameters': api.get('required_parameters', []),
                        'optional_parameters': api.get('optional_parameters', []),
                        'template_response': api.get('template_response', {})
                    }
                    # 如果提供了过滤器，应用它
                    if tool_filter is None or tool_filter(tool_info):
                        tools.append(tool_info)
        
        print(f"  找到 {len(tools)} 个可用工具")
        
        # 选择工具
        if num_tools is None:
            # 加载所有工具
            selected_tools = tools
            print(f"  加载所有 {len(selected_tools)} 个工具")
        elif len(tools) > num_tools:
            # 随机选择指定数量的工具
            if random_seed is not None:
                random.seed(random_seed)
            selected_tools = random.sample(tools, num_tools)
            print(f"  随机选择了 {len(selected_tools)} 个工具（从 {len(tools)} 个中）")
        else:
            # 工具数量不足，返回所有工具
            selected_tools = tools
            print(f"  工具数量不足，加载所有 {len(selected_tools)} 个工具")
        
        return selected_tools
    except Exception as e:
        print(f"警告: 无法加载 Toolbench 工具: {e}")
        import traceback
        traceback.print_exc()
        return []


def create_contract_from_toolbench_tool(tool_info):
    """
    从 Toolbench 工具信息创建工具合约
    
    Args:
        tool_info: 工具信息字典
    
    Returns:
        ToolContract 对象
    """
    tool_name = tool_info['tool_name']
    api_name = tool_info['api_name']
    api_description = tool_info.get('api_description', '')
    category = tool_info.get('category_name', '')
    
    # 构建更详细的工具描述，用于检索
    # 包括：类别、工具名、API名、描述、参数信息
    description_parts = []
    if category:
        description_parts.append(f"类别: {category}")
    if api_description:
        description_parts.append(api_description)
    else:
        # 如果没有描述，从工具名和API名推断
        description_parts.append(f"{tool_name} - {api_name}")
    
    # 添加参数信息，帮助理解工具功能
    required_params = tool_info.get('required_parameters', [])
    optional_params = tool_info.get('optional_parameters', [])
    if required_params:
        param_names = [p.get('name', '') for p in required_params if p.get('name')]
        if param_names:
            description_parts.append(f"需要参数: {', '.join(param_names)}")
    
    # 组合成完整描述
    description = ". ".join(description_parts)
    
    # 构建前置条件（基于必需参数）
    precondition_constraints = []
    required_params = tool_info.get('required_parameters', [])
    for param in required_params:
        param_name = param.get('name', '').lower()
        if param_name:
            precondition_constraints.append(f'exists({param_name})')
    
    # 构建后置条件（基于响应模板）
    result_constraints = []
    template_response = tool_info.get('template_response', {})
    if isinstance(template_response, dict):
        # 提取主要字段
        for key in list(template_response.keys())[:3]:  # 只取前3个字段
            result_constraints.append(f'has_field("{key}")')
    
    # 构建输入 schema
    properties = {}
    required = []
    
    for param in required_params:
        param_name = param.get('name', '')
        param_type = param.get('type', 'STRING').lower()
        if param_name:
            if 'string' in param_type or 'str' in param_type:
                properties[param_name] = {"type": "string"}
            elif 'int' in param_type:
                properties[param_name] = {"type": "integer"}
            elif 'bool' in param_type:
                properties[param_name] = {"type": "boolean"}
            else:
                properties[param_name] = {"type": "string"}
            required.append(param_name)
    
    for param in tool_info.get('optional_parameters', []):
        param_name = param.get('name', '')
        param_type = param.get('type', 'STRING').lower()
        if param_name:
            if 'string' in param_type or 'str' in param_type:
                properties[param_name] = {"type": "string"}
            elif 'int' in param_type:
                properties[param_name] = {"type": "integer"}
            elif 'bool' in param_type:
                properties[param_name] = {"type": "boolean"}
            else:
                properties[param_name] = {"type": "string"}
    
    input_schema = {
        "type": "object",
        "properties": properties,
        "required": required
    }
    
    # 创建合约
    contract = ToolContract(
        tool_name=f"{tool_name}_{api_name}",
        tool_description=description or f"{tool_name} - {api_name}",
        precondition=Precondition(constraints=precondition_constraints),
        postcondition=Postcondition(
            result_constraints=result_constraints if result_constraints else ['has_field("response")'],
            state_update_rules={}  # 可以根据需要添加状态更新规则
        ),
        input_schema=input_schema
    )
    
    return contract, tool_info


def create_real_test_setup():
    """创建使用真实 LLM 的测试设置"""
    
    # 检查 API Key（优先从 .env 文件加载）
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        raise ValueError(
            "需要设置 OPENAI_API_KEY 环境变量\n"
            "方法 1: 在 .env 文件中设置 OPENAI_API_KEY\n"
            "方法 2: 运行: export OPENAI_API_KEY='your-api-key'"
        )
    
    # 检查 Toolbench API Key
    toolbench_key = os.getenv("TOOLBENCH_API_KEY", "")
    if not toolbench_key:
        print("警告: 未设置 TOOLBENCH_API_KEY，将无法使用 Toolbench 工具")
    
    # 1. 从 Toolbench 加载工具
    toolbench_tool_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "..", "StableToolBench", "solvable_queries", "test_instruction", "G1_tool.json"
    )
    
    # 如果文件不存在，尝试其他路径
    if not os.path.exists(toolbench_tool_path):
        # 尝试其他可能的路径
        alt_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                       "StableToolBench", "solvable_queries", "test_instruction", "G1_tool.json"),
            "/home/oceann/code/StableToolBench/solvable_queries/test_instruction/G1_tool.json"
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                toolbench_tool_path = alt_path
                break
    
    print(f"正在从 Toolbench 加载工具: {toolbench_tool_path}")
    
    # 可以自定义工具选择策略：
    # 1. 指定数量：num_tools=4（随机选择4个）
    # 2. 加载所有：num_tools=None（加载所有工具）
    # 3. 使用过滤器：tool_filter=lambda t: t['category_name'] == 'Data'（只选择特定类别）
    # 4. 设置随机种子：random_seed=42（可重复的随机选择）
    
    # 示例：可以选择特定类别的工具
    # tool_filter = lambda t: t['category_name'] in ['Data', 'Social']
    tool_filter = None  # None 表示不过滤
    
    toolbench_tools = load_toolbench_tools(
        toolbench_tool_path, 
        num_tools=800,  # 可以改为 None 加载所有工具，或改为其他数字
        tool_filter=tool_filter,
        random_seed=None  # 可以设置随机种子，如 42，用于可重复的测试
    )
    
    if not toolbench_tools:
        print("警告: 无法加载 Toolbench 工具，使用默认工具")
        # 回退到默认工具
        toolbench_tools = []
    
    # 创建工具合约
    tool_contracts = {}
    tool_info_map = {}  # 存储工具信息，用于执行时使用
    
    for tool_info in toolbench_tools:
        contract, info = create_contract_from_toolbench_tool(tool_info)
        tool_key = contract.tool_name
        tool_contracts[tool_key] = contract
        tool_info_map[tool_key] = info
        print(f"  ✓ 加载工具: {info['tool_name']} - {info['api_name']} (类别: {info['category_name']})")
    
    if not tool_contracts:
        print("错误: 没有成功加载任何工具")
        raise ValueError("无法加载 Toolbench 工具")
    
    print(f"成功加载 {len(tool_contracts)} 个工具\n")
    
    # 2. 创建工具执行器（使用 Toolbench 接口）
    # 创建共享的 Toolbench 执行器
    toolbench_executor = ToolExecutor(
        use_toolbench=True,
        toolbench_config={
            "tools_root": "data.toolenv.tools",
            "schema_root": "data/toolenv/response_examples",
            "rapidapi_key": toolbench_key,
            "toolbench_key": toolbench_key,
            "api_customization": False,
            "strip_method": "truncate"
        }
    )
    
    def toolbench_execute(tool_name, params, tool_config=None):
        """
        使用 Toolbench 执行工具
        
        工具名称格式: {tool_name}_{api_name}
        需要从 tool_info_map 中获取工具信息
        """
        print(f"    [工具执行] {tool_name} with params: {params}")
        
        # 查找对应的工具信息
        tool_info = None
        for key, info in tool_info_map.items():
            if key == tool_name:
                tool_info = info
                break
        
        if not tool_info:
            return {"error": f"Unknown tool: {tool_name}"}
        
        # 构建 Toolbench 调用配置
        tool_config_dict = {
            "category": tool_info['category_name'],
            "api_name": tool_info['api_name'],
            "tool_input": params
        }
        
        # 使用 Toolbench 执行器
        result = toolbench_executor.execute(
            tool_name=tool_info['tool_name'],
            params=params,
            tool_config=tool_config_dict
        )
        
        # 处理 Toolbench 返回格式
        if result.get('error'):
            print(f"    [工具执行] 错误: {result['error']}")
            return {"error": result['error']}
        else:
            # 尝试解析响应
            response_str = result.get('response', '')
            try:
                if isinstance(response_str, str):
                    # 尝试解析 JSON
                    if response_str.strip().startswith('{') or response_str.strip().startswith('['):
                        response_dict = json.loads(response_str)
                    else:
                        # 如果不是 JSON，返回字符串
                        response_dict = {"response": response_str}
                else:
                    response_dict = response_str
                print(f"    [工具执行] 成功")
                return response_dict
            except json.JSONDecodeError:
                # 如果无法解析，返回原始响应
                print(f"    [工具执行] 成功（响应为字符串）")
                return {"response": response_str}
            except Exception as e:
                print(f"    [工具执行] 解析响应时出错: {e}")
                return {"response": response_str}
    
    executor = ToolExecutor(tool_executor_func=toolbench_execute)
    
    # 3. 创建真实的 LLM 客户端
    llm_client = GPTClient(
        model="gpt-4o-mini",  # 使用便宜的模型
        openai_key=openai_key
    )
    
    # 4. 创建 Prompt 模板
    prompt_template = ReActPromptTemplate()
    
    return {
        "tool_contracts": tool_contracts,
        "tool_executor": executor,
        "llm_client": llm_client,
        "prompt_template": prompt_template
    }


def test_pipeline_with_real_llm():
    """使用真实 LLM 测试 pipeline"""
    print("Pipeline 真实 LLM 测试")
    print("⚠️  注意：此测试会调用真实的 OpenAI API，会产生费用")
    print("使用的模型: gpt-4o-mini (相对便宜)\n")
    
    try:
        # 设置
        setup = create_real_test_setup()
        print("✓ 测试环境设置完成\n")
    except ValueError as e:
        print(f"✗ 设置失败: {e}")
        return None
    
    # 创建 pipeline
    pipeline = ToolGatePipeline(
        tool_contracts=setup["tool_contracts"],
        llm_client=setup["llm_client"],
        prompt_template=setup["prompt_template"],
        tool_executor=setup["tool_executor"],
        max_steps=5
    )
    
    # 输入（根据加载的工具调整查询）
    # 使用一个通用的查询，让 LLM 选择合适的工具
    query = "帮我想在网上买一个苹果手机"
    initial_entities = {}  # 让 LLM 从查询中提取实体
    
    print(f"【输入】")
    print(f"  查询: {query}")
    print(f"  初始实体: {initial_entities}\n")
    
    # 运行
    print("【执行过程】")
    print("正在调用 LLM 执行 pipeline...")
    
    try:
        result = pipeline.run(
            query=query,
            initial_entities=initial_entities
        )
        print("✓ Pipeline 执行完成\n")
    except Exception as e:
        print(f"  ✗ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 输出结果
    print("\n【输出结果】")
    
    print(f"\n最终答案:")
    print(f"{result['answer']}")
    
    print(f"\n最终状态:")
    import json
    state_summary = {}
    for key, value in result['final_state'].items():
        state_summary[key] = value.get('value', 'N/A')
    print(f"{json.dumps(state_summary, indent=2, ensure_ascii=False)}")
    
    print(f"\n执行轨迹 (共 {len(result['trace'])} 步):")
    for i, step in enumerate(result['trace']):
        step_num = step.get('step', i)
        print(f"\n步骤 {step_num}:")
        
        # 动作类型
        action_type = step.get('action_type', 'N/A')
        print(f"  动作类型: {action_type}")
        
        # 状态摘要
        state_summary = step.get('state_summary', 'N/A')
        if state_summary and state_summary != 'N/A':
            print(f"  当前状态: {state_summary[:200]}")
        
        # 动作内容（如果有）
        action_content = step.get('action_content')
        if action_content:
            print(f"  动作内容: {action_content[:300]}{'...' if len(action_content) > 300 else ''}")
        
        # 工具调用信息
        if step.get('tool_called'):
            tool_name = step['tool_called']
            print(f"  🔧 调用工具: {tool_name}")
            
            # 工具结果（如果有）
            tool_result = step.get('tool_result')
            if tool_result:
                import json
                # 检查是否有错误
                has_error = tool_result.get('error') or 'error' in str(tool_result).lower()
                if has_error:
                    print(f"  ⚠️  工具返回错误: {tool_result.get('error', 'Unknown error')}")
                else:
                    result_str = json.dumps(tool_result, indent=2, ensure_ascii=False)
                    if len(result_str) > 500:
                        print(f"  工具返回结果 (前500字符):")
                        print(f"  {result_str[:500]}...")
                    else:
                        print(f"  工具返回结果: {result_str}")
            
            # 验证结果
            verif = step.get('verification_result')
            if verif:
                if verif.get('success'):
                    print(f"  ✅ 后置条件验证: 通过")
                else:
                    error_msg = verif.get('error_message', 'N/A')
                    print(f"  ❌ 后置条件验证: 失败 - {error_msg}")
            
            # 状态更新
            if step.get('state_updated'):
                print(f"  ✅ 状态已更新")
            else:
                print(f"  ⚠️  状态未更新")
        else:
            # 如果是答案动作，显示答案内容
            if action_type == 'answer':
                print(f"  📝 生成最终答案")
    
    print(f"\n对话历史 (共 {len(result['conversation_history'])} 条):")
    for i, msg in enumerate(result['conversation_history']):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        
        # 检查特殊标记
        markers = []
        if "<start_call_tool>" in content:
            markers.append("🔧 CALL_TOOL")
        if "<start_tool_result>" in content:
            markers.append("📦 TOOL_RESULT")
        if "<end_call_tool>" in content:
            markers.append("✅ END_CALL")
        
        marker_str = f" {' | '.join(markers)}" if markers else ""
        
        print(f"\n[{i}] {role.upper()}{marker_str}")
        
        # 完整显示内容，保持格式
        # 如果内容很长，完整显示但添加提示
        if len(content) > 2000:
            print(f"(内容较长，显示前2000字符)")
            for line in content[:2000].split('\n'):
                print(f"{line}")
            print("... (内容较长，已截断) ...")
        else:
            # 完整显示，保持换行
            for line in content.split('\n'):
                print(f"{line}")
    
    # 统计信息
    print(f"\n【统计信息】")
    
    tool_calls = sum(1 for s in result['trace'] if s.get('tool_called'))
    successful_calls = sum(1 for s in result['trace'] 
                          if s.get('tool_called') and s.get('verification_result') and s.get('verification_result', {}).get('success'))
    failed_calls = sum(1 for s in result['trace'] 
                      if s.get('tool_called') and s.get('verification_result') and not s.get('verification_result', {}).get('success'))
    state_updates = sum(1 for s in result['trace'] if s.get('state_updated'))
    
    # 统计调用的工具
    called_tools = [s.get('tool_called') for s in result['trace'] if s.get('tool_called')]
    
    print(f"总执行步数: {len(result['trace'])}")
    print(f"工具调用次数: {tool_calls}")
    print(f"成功调用次数: {successful_calls}")
    print(f"失败调用次数: {failed_calls}")
    print(f"状态更新次数: {state_updates}")
    print(f"对话轮数: {len(result['conversation_history'])}")
    if called_tools:
        print(f"调用的工具: {', '.join(called_tools)}")
    
    # 验证
    print(f"\n【验证结果】")
    
    checks = {
        "答案不为空": bool(result.get("answer")),
        "包含最终状态": "final_state" in result,
        "包含执行轨迹": "trace" in result,
        "包含对话历史": "conversation_history" in result,
        "工具被调用": tool_calls > 0,
        "工具调用成功": successful_calls > 0,
        "有工具调用失败": failed_calls > 0,  # 新增：检查是否有失败的工具
        "状态被更新": state_updates > 0,
    }
    
    all_passed = True
    for check_name, check_result in checks.items():
        status = "✓" if check_result else "✗"
        print(f"{status} {check_name}")
        if not check_result:
            all_passed = False
    
    print()
    if all_passed:
        print("✓ 所有检查通过！")
    else:
        print("✗ 部分检查失败")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline 真实 LLM 测试")
    args = parser.parse_args()
    
    
    result = test_pipeline_with_real_llm()
    
    if result:
        # 保存结果
        import json
        output_file = "pipeline_real_llm_output.json"
        with open(output_file, "w", encoding="utf-8") as f:
            simplified = {
                "answer": result.get("answer", ""),
                "final_state": {
                    k: v.get("value") for k, v in result.get("final_state", {}).items()
                },
                "trace_summary": [
                    {
                        "step": s.get("step"),
                        "action_type": s.get("action_type"),
                        "tool_called": s.get("tool_called"),
                        "verification_success": (
                            s.get("verification_result", {}).get("success")
                            if s.get("verification_result") is not None
                            else None
                        ),
                        "state_updated": s.get("state_updated", False)
                    }
                    for s in result.get("trace", [])
                ],
                "conversation_turns": len(result.get("conversation_history", []))
            }
            json.dump(simplified, f, indent=2, ensure_ascii=False)
        
        print(f"\n测试结果已保存到: {output_file}")

