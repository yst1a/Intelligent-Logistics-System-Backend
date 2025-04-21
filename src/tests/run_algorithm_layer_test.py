"""
运行算法层测试并生成报告
"""
import os
import sys
from algorithm_layer_test_report import AlgorithmLayerTestReporter

def main():
    print("开始运行算法层测试...")
    
    # 创建报告生成器并运行测试
    reporter = AlgorithmLayerTestReporter()
    reporter.run_tests()
    
    print(f"测试报告已生成在: {os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test_results', 'test_report.html'))}")
    print("测试完成!")


if __name__ == "__main__":
    main()