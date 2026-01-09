"""
环境变量加载工具

用于从 .env 文件加载环境变量
"""

import os
from pathlib import Path


def load_env_file(env_file: str = ".env"):
    """
    从 .env 文件加载环境变量
    
    Args:
        env_file: .env 文件路径（相对于项目根目录）
    """
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    env_path = project_root / env_file
    
    if not env_path.exists():
        print(f"警告: {env_path} 文件不存在")
        return
    
    # 读取 .env 文件
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith("#"):
                continue
            
            # 解析 KEY=VALUE
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                
                # 移除引号（如果有）
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                # 设置环境变量（如果尚未设置）
                if key and key not in os.environ:
                    os.environ[key] = value


# 自动加载（如果直接导入此模块）
if __name__ != "__main__":
    load_env_file()

