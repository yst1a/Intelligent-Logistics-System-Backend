#!/usr/bin/env python3
"""
数据层可视化测试运行程序
"""

import os
import sys
import webbrowser
from test_data_layer_visual import DataLayerVisualTester

def main():
    """运行数据层可视化测试"""
    print("开始运行数据层可视化测试...")
    
    # 创建测试实例并运行所有测试
    tester = DataLayerVisualTester(order_count=15, vehicle_count=5)
    test_result = tester.run_all_tests()
    
    # 自动打开测试报告
    report_path = os.path.abspath("../test_results/test_report.html")
    
    # 检查报告是否生成
    if os.path.exists(report_path):
        print(f"\n测试报告已生成: {report_path}")
        print("尝试在浏览器中打开测试报告...")
        webbrowser.open('file://' + report_path)
    else:
        print(f"\n警告: 测试报告未生成")
    
    # 返回测试结果状态码
    return 0 if test_result else 1

if __name__ == "__main__":
    sys.exit(main())