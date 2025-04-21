#!/usr/bin/env python3
"""
增强数据层测试报告运行程序
为"智送"城市货运智能调度系统生成高质量可视化测试报告
"""

import os
import sys
import importlib.util


def check_install_package(package_name):
    """检查并安装缺少的依赖包"""
    if importlib.util.find_spec(package_name) is None:
        print(f"检测到缺少依赖: {package_name}，正在尝试安装...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"{package_name} 安装成功!")
        except Exception as e:
            print(f"无法安装 {package_name}: {e}")
            print(f"提示: 请手动运行 'pip install {package_name}' 来安装此依赖")
            return False
    return True


# 检查并添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 检查必要的依赖
required_packages = ['numpy', 'pandas', 'matplotlib', 'seaborn', 'networkx']
optional_packages = ['plotly']

print("检查依赖...")
for package in required_packages:
    if not check_install_package(package):
        print(f"错误: 缺少必要的依赖 {package}，请安装后重试")
        sys.exit(1)

for package in optional_packages:
    check_install_package(package)  # 可选的，不强制退出

print("依赖检查完成!\n")

# 确保数据层包装器可用
try:
    from data_layer_wrapper import USING_MOCK

    if USING_MOCK:
        print("使用模拟数据层进行测试")
    else:
        print("使用真实数据层进行测试")
except ImportError:
    print("错误: 无法导入数据层包装器模块")
    sys.exit(1)

from enhanced_data_layer_report import EnhancedDataLayerReportGenerator


def main():
    """运行增强数据层测试报告生成器"""
    print("\n" + "=" * 50)
    print("开始运行智送系统数据层增强测试...")
    print("=" * 50 + "\n")

    # 创建测试实例并运行所有测试
    order_count = 20  # 测试订单数量
    vehicle_count = 8  # 测试车辆数量

    generator = EnhancedDataLayerReportGenerator(
        order_count=order_count,
        vehicle_count=vehicle_count
    )

    test_result = generator.run_tests_and_generate_report()

    print("\n" + "=" * 50)
    print("测试完成!")
    print("=" * 50)

    # 返回测试结果状态码
    return 0 if test_result else 1


if __name__ == "__main__":
    sys.exit(main())