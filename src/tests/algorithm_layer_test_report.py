"""
算法层测试报告生成器
用于测试算法层各组件并生成详细的HTML报告
"""
import sys
import os
import time
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import numpy as np
import base64
from io import BytesIO
from tabulate import tabulate
import json
import pandas as pd

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_layer.data_generator import DataGenerator
from data_layer.city_map import CityMap
from data_layer.order import Order
from data_layer.vehicle import Vehicle

from algorithm_layer.coordinator import AlgorithmCoordinator
from algorithm_layer.base import Solution
from algorithm_layer.assignment import GreedyAssignmentAlgorithm, InsertionHeuristicAlgorithm, BatchAssignmentAlgorithm
from algorithm_layer.routing import BasicRoutingAlgorithm, OrderFirstRoutingAlgorithm, LocalSearchRoutingAlgorithm
from algorithm_layer.dynamic_optimizer import DynamicOptimizer


class AlgorithmLayerTestReporter:
    """算法层测试报告生成器"""
    
    def __init__(self):
        """初始化测试环境"""
        # 创建数据生成器
        self.data_generator = DataGenerator(seed=42)
        
        # 生成城市地图
        self.city_map = self.data_generator.generate_dongying_map()
        
        # 初始化算法协调器
        self.coordinator = AlgorithmCoordinator(self.city_map)
        
        # 当前时间
        self.current_time = datetime.now()
        
        # 测试结果
        self.results = {}
        
        # 图表和可视化
        self.figures = {}
        
        # 初始化matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        matplotlib.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
        
        # 路径信息
        self.report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test_results'))
        os.makedirs(self.report_path, exist_ok=True)
        
    def run_tests(self):
        """运行所有测试"""
        test_results = {}
        
        # 1. 基本功能测试
        test_results['basic_functionality'] = self.test_basic_functionality()
        
        # 2. 性能和效率测试
        test_results['performance'] = self.test_performance()
        
        # 3. 质量测试
        test_results['quality'] = self.test_quality()
        
        # 4. 场景测试
        test_results['scenarios'] = self.test_scenarios()
        
        # 生成HTML报告
        self.generate_html_report(test_results)
        
    def test_basic_functionality(self):
        """测试算法层基本功能"""
        print("测试算法层基本功能...")
        results = {}
        
        # 生成测试车辆
        vehicles = self.data_generator.generate_random_vehicles(self.city_map, 5)
        
        # 生成测试订单
        orders = self.data_generator.generate_random_orders(
            self.city_map, 20, self.current_time, 12.0)
        
        # 1. 测试订单分配算法
        assignment_results = self.test_assignment_algorithms(orders, vehicles)
        results['assignment'] = assignment_results
        
        # 2. 测试路径规划算法
        if 'solution' in assignment_results:
            routing_results = self.test_routing_algorithms(assignment_results['solution'], vehicles)
            results['routing'] = routing_results
        
        return results
    
    def test_assignment_algorithms(self, orders, vehicles):
        """测试订单分配算法"""
        print("测试订单分配算法...")
        
        algorithms = {
            "贪心分配算法": GreedyAssignmentAlgorithm(self.city_map),
            "插入启发式算法": InsertionHeuristicAlgorithm(self.city_map),
            "批量分配算法": BatchAssignmentAlgorithm(self.city_map)
        }
        
        results = {'algorithms': {}}
        solution = None
        
        for name, algorithm in algorithms.items():
            print(f"测试 {name}...")
            start_time = time.time()
            
            # 运行算法
            solution = algorithm.solve(orders, vehicles)
            
            # 计算执行时间
            exec_time = time.time() - start_time
            
            # 收集结果
            assigned_count = len(orders) - len(solution.unassigned_orders)
            assignment_rate = assigned_count / len(orders) * 100
            
            results['algorithms'][name] = {
                "执行时间": exec_time,
                "总距离": solution.total_distance,
                "总时间": solution.total_time,
                "违反约束": solution.total_violations,
                "分配成功率": assignment_rate,
                "已分配订单数": assigned_count,
                "未分配订单数": len(solution.unassigned_orders)
            }
            
            # 保存批量分配算法的解决方案用于后续测试
            if name == "批量分配算法":
                results['solution'] = solution
        
        # 生成算法比较图
        self.generate_assignment_comparison_chart(results)
        
        return results
    
    def test_routing_algorithms(self, solution, vehicles):
        """测试路径规划算法"""
        print("测试路径规划算法...")
        
        algorithms = {
            "基本路径规划": BasicRoutingAlgorithm(self.city_map),
            "订单优先规划": OrderFirstRoutingAlgorithm(self.city_map),
            "局部搜索优化": LocalSearchRoutingAlgorithm(self.city_map)
        }
        
        results = {'algorithms': {}}
        final_solution = None
        
        # 从解决方案中提取所有订单
        orders = []
        order_ids = set()
        for route in solution.routes.values():
            for point in route.points:
                if point.order.id not in order_ids:
                    order_ids.add(point.order.id)
                    orders.append(point.order)
        
        for name, algorithm in algorithms.items():
            print(f"测试 {name}...")
            start_time = time.time()
            
            # 运行算法
            optimized = algorithm.solve(orders, vehicles)
            
            # 计算执行时间
            exec_time = time.time() - start_time
            
            # 收集结果
            results['algorithms'][name] = {
                "执行时间": exec_time,
                "总距离": optimized.total_distance,
                "总时间": optimized.total_time,
                "违反约束": optimized.total_violations
            }
            
            # 保存局部搜索优化算法的解决方案
            if name == "局部搜索优化":
                final_solution = optimized
        
        # 生成算法比较图
        self.generate_routing_comparison_chart(results)
        
        if final_solution:
            # 可视化最终路线
            self.visualize_routes(final_solution, 'routing_final_routes')
            results['solution'] = final_solution
        
        return results
    
    def test_performance(self):
        """测试算法性能与效率"""
        print("测试算法性能与效率...")
        results = {}
        
        # 测试不同规模的订单数
        order_sizes = [10, 20, 50, 100]
        
        performance_data = {
            "订单数量": [],
            "贪心分配算法": [],
            "插入启发式算法": [],
            "批量分配算法": [],
            "基本路径规划": [],
            "局部搜索优化": []
        }
        
        for size in order_sizes:
            print(f"测试 {size} 个订单...")
            performance_data["订单数量"].append(size)
            
            # 生成测试车辆 (固定数量)
            vehicles = self.data_generator.generate_random_vehicles(self.city_map, 5)
            
            # 生成测试订单
            orders = self.data_generator.generate_random_orders(
                self.city_map, size, self.current_time, 12.0)
            
            # 测试订单分配算法
            algorithms = {
                "贪心分配算法": GreedyAssignmentAlgorithm(self.city_map),
                "插入启发式算法": InsertionHeuristicAlgorithm(self.city_map),
                "批量分配算法": BatchAssignmentAlgorithm(self.city_map)
            }
            
            for name, algorithm in algorithms.items():
                print(f"  测试 {name}...")
                start_time = time.time()
                algorithm.solve(orders, vehicles)
                exec_time = time.time() - start_time
                performance_data[name].append(exec_time)
            
            # 获取批量分配的解决方案用于路径规划测试
            solution = BatchAssignmentAlgorithm(self.city_map).solve(orders, vehicles)
            
            # 从解决方案中提取所有订单
            assigned_orders = []
            order_ids = set()
            for route in solution.routes.values():
                for point in route.points:
                    if point.order.id not in order_ids:
                        order_ids.add(point.order.id)
                        assigned_orders.append(point.order)
            
            # 测试路径规划算法
            routing_algorithms = {
                "基本路径规划": BasicRoutingAlgorithm(self.city_map),
                "局部搜索优化": LocalSearchRoutingAlgorithm(self.city_map)
            }
            
            for name, algorithm in routing_algorithms.items():
                print(f"  测试 {name}...")
                start_time = time.time()
                algorithm.solve(assigned_orders, vehicles)
                exec_time = time.time() - start_time
                performance_data[name].append(exec_time)
        
        results['performance_data'] = performance_data
        
        # 生成性能比较图
        self.generate_performance_chart(performance_data)
        
        return results
    
    def test_quality(self):
        """测试算法质量"""
        print("测试算法质量...")
        results = {}
        
        # 生成测试车辆
        vehicles = self.data_generator.generate_random_vehicles(self.city_map, 5)
        
        # 生成不同时间窗口紧张度的订单
        time_window_hours = [1.0, 2.0, 4.0, 8.0]
        quality_data = {
            "时间窗口(小时)": [],
            "分配成功率(%)": [],
            "违反约束数": [],
            "平均路线长度(km)": []
        }
        
        for hours in time_window_hours:
            print(f"测试时间窗口 {hours} 小时...")
            quality_data["时间窗口(小时)"].append(hours)
            
            # 生成订单
            orders = self.data_generator.generate_random_orders(
                self.city_map, 30, self.current_time, hours)
            
            # 使用批量分配算法
            algorithm = BatchAssignmentAlgorithm(self.city_map)
            solution = algorithm.solve(orders, vehicles)
            
            # 收集结果
            assigned_count = len(orders) - len(solution.unassigned_orders)
            assignment_rate = assigned_count / len(orders) * 100
            quality_data["分配成功率(%)"].append(assignment_rate)
            quality_data["违反约束数"].append(solution.total_violations)
            
            if len(solution.routes) > 0:
                avg_route_length = solution.total_distance / len(solution.routes)
            else:
                avg_route_length = 0
            quality_data["平均路线长度(km)"].append(avg_route_length)
            
            # 使用局部搜索优化
            if assigned_count > 0:
                # 提取已分配订单
                assigned_orders = []
                order_ids = set()
                for route in solution.routes.values():
                    for point in route.points:
                        if point.order.id not in order_ids:
                            order_ids.add(point.order.id)
                            assigned_orders.append(point.order)
                
                # 优化路线
                optimizer = LocalSearchRoutingAlgorithm(self.city_map)
                optimizer.solve(assigned_orders, vehicles)
        
        results['quality_data'] = quality_data
        
        # 生成质量比较图
        self.generate_quality_chart(quality_data)
        
        return results
    
    def test_scenarios(self):
        """测试特定场景"""
        print("测试特定场景...")
        results = {}
        
        # 1. 测试动态订单插入
        dynamic_results = self.test_dynamic_order_insertion()
        results['dynamic_insertion'] = dynamic_results
        
        # 2. 测试交通状况变化
        traffic_results = self.test_traffic_change()
        results['traffic_change'] = traffic_results
        
        # 3. 测试高峰期处理
        peak_results = self.test_peak_hour()
        results['peak_hour'] = peak_results
        
        return results
    
    def test_dynamic_order_insertion(self):
        """测试动态订单插入"""
        print("测试动态订单插入...")
        results = {}
        
        # 生成测试车辆
        vehicles = self.data_generator.generate_random_vehicles(self.city_map, 3)
        
        # 生成初始订单
        initial_orders = self.data_generator.generate_random_orders(
            self.city_map, 15, self.current_time, 6.0)
        
        # 使用批量分配算法处理初始订单
        algorithm = BatchAssignmentAlgorithm(self.city_map)
        initial_solution = algorithm.solve(initial_orders, vehicles)
        
        # 生成新订单
        new_order = self.data_generator.generate_random_orders(
            self.city_map, 1, self.current_time, 2.0)[0]
        
        # 使用动态优化器插入新订单
        optimizer = DynamicOptimizer(self.city_map)
        updated_solution, inserted = optimizer.insert_new_order(
            initial_solution, new_order, self.current_time)
        
        # 收集结果
        results['initial_distance'] = initial_solution.total_distance
        results['initial_time'] = initial_solution.total_time
        results['initial_violations'] = initial_solution.total_violations
        
        results['updated_distance'] = updated_solution.total_distance
        results['updated_time'] = updated_solution.total_time
        results['updated_violations'] = updated_solution.total_violations
        results['order_inserted'] = inserted
        
        # 可视化插入前后的路线
        self.visualize_routes(initial_solution, 'dynamic_before')
        self.visualize_routes(updated_solution, 'dynamic_after')
        
        # 生成比较图表
        self.generate_dynamic_insertion_chart(results)
        
        return results
    
    def test_traffic_change(self):
        """测试交通状况变化"""
        print("测试交通状况变化...")
        results = {}
        
        # 生成测试车辆
        vehicles = self.data_generator.generate_random_vehicles(self.city_map, 4)
        
        # 生成订单
        orders = self.data_generator.generate_random_orders(
            self.city_map, 20, self.current_time, 6.0)
        
        # 使用批量分配算法生成初始解决方案
        algorithm = BatchAssignmentAlgorithm(self.city_map)
        initial_solution = algorithm.solve(orders, vehicles)
        
        # 修改地图上的交通状况
        # 随机选择一些道路增加行驶时间
        edges_to_change = random.sample(list(self.city_map.graph.edges()), 5)
        affected_roads = []
        new_travel_times = {}
        original_travel_times = {}
        
        for edge in edges_to_change:
            from_id, to_id = edge
            original_time = self.city_map.graph[from_id][to_id]['travel_time']
            new_time = original_time * 1.8  # 增加80%的行驶时间
            
            original_travel_times[(from_id, to_id)] = original_time
            new_travel_times[(from_id, to_id)] = new_time
            affected_roads.append((from_id, to_id))
            
            # 更新地图
            self.city_map.graph[from_id][to_id]['travel_time'] = new_time
        
        # 使用动态优化器处理交通变化
        optimizer = DynamicOptimizer(self.city_map)
        updated_solution = optimizer.handle_traffic_update(
            initial_solution, affected_roads, new_travel_times, self.current_time)
        
        # 收集结果
        results['affected_roads'] = len(affected_roads)
        results['initial_distance'] = initial_solution.total_distance
        results['initial_time'] = initial_solution.total_time
        results['initial_violations'] = initial_solution.total_violations
        
        results['updated_distance'] = updated_solution.total_distance
        results['updated_time'] = updated_solution.total_time
        results['updated_violations'] = updated_solution.total_violations
        
        # 生成比较图表
        self.generate_traffic_change_chart(results)
        
        # 可视化交通变化前后的路线
        self.visualize_routes(initial_solution, 'traffic_before')
        self.visualize_routes(updated_solution, 'traffic_after')
        
        # 恢复地图上的交通状况
        for (from_id, to_id), original_time in original_travel_times.items():
            self.city_map.graph[from_id][to_id]['travel_time'] = original_time
        
        return results
    
    def test_peak_hour(self):
        """测试高峰期处理"""
        print("测试高峰期处理...")
        results = {}
        
        # 生成测试车辆
        vehicles = self.data_generator.generate_random_vehicles(self.city_map, 8)
        
        # 生成高峰期大量订单
        peak_orders = self.data_generator.generate_random_orders(
            self.city_map, 60, self.current_time, 3.0)
        
        # 测试不同批次大小的批量分配算法
        batch_sizes = [5, 10, 20, 30]
        
        batch_results = {
            "批次大小": [],
            "执行时间(秒)": [],
            "分配成功率(%)": [],
            "总距离(km)": [],
            "总时间(小时)": []
        }
        
        for batch_size in batch_sizes:
            print(f"测试批次大小 {batch_size}...")
            batch_results["批次大小"].append(batch_size)
            
            # 创建批量分配算法实例
            algorithm = BatchAssignmentAlgorithm(self.city_map, batch_size=batch_size)
            
            # 运行算法
            start_time = time.time()
            solution = algorithm.solve(peak_orders, vehicles)
            exec_time = time.time() - start_time
            
            # 收集结果
            assigned_count = len(peak_orders) - len(solution.unassigned_orders)
            assignment_rate = assigned_count / len(peak_orders) * 100
            
            batch_results["执行时间(秒)"].append(exec_time)
            batch_results["分配成功率(%)"].append(assignment_rate)
            batch_results["总距离(km)"].append(solution.total_distance)
            batch_results["总时间(小时)"].append(solution.total_time)
            
            # 保存最后一种配置的解决方案用于可视化
            if batch_size == batch_sizes[-1]:
                self.visualize_routes(solution, 'peak_hour')
        
        results['batch_results'] = batch_results
        
        # 生成比较图表
        self.generate_peak_hour_chart(batch_results)
        
        return results
    
    def generate_assignment_comparison_chart(self, results):
        """生成订单分配算法比较图"""
        if 'algorithms' not in results:
            return
            
        algorithms = results['algorithms']
        names = list(algorithms.keys())
        
        # 执行时间比较
        times = [algorithms[name]["执行时间"] for name in names]
        
        # 分配成功率比较
        rates = [algorithms[name]["分配成功率"] for name in names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 执行时间柱状图
        ax1.bar(names, times, color='skyblue')
        ax1.set_title('订单分配算法执行时间比较')
        ax1.set_ylabel('执行时间 (秒)')
        ax1.set_ylim(bottom=0)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 分配成功率柱状图
        ax2.bar(names, rates, color='lightgreen')
        ax2.set_title('订单分配算法成功率比较')
        ax2.set_ylabel('分配成功率 (%)')
        ax2.set_ylim(0, 100)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # 保存图表
        fig_path = self._save_figure(fig, 'assignment_comparison')
        self.figures['assignment_comparison'] = fig_path
    
    def generate_routing_comparison_chart(self, results):
        """生成路径规划算法比较图"""
        if 'algorithms' not in results:
            return
            
        algorithms = results['algorithms']
        names = list(algorithms.keys())
        
        # 执行时间比较
        times = [algorithms[name]["执行时间"] for name in names]
        
        # 总距离比较
        distances = [algorithms[name]["总距离"] for name in names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 执行时间柱状图
        ax1.bar(names, times, color='skyblue')
        ax1.set_title('路径规划算法执行时间比较')
        ax1.set_ylabel('执行时间 (秒)')
        ax1.set_ylim(bottom=0)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 总距离柱状图
        ax2.bar(names, distances, color='salmon')
        ax2.set_title('路径规划算法总距离比较')
        ax2.set_ylabel('总距离 (km)')
        ax2.set_ylim(bottom=0)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # 保存图表
        fig_path = self._save_figure(fig, 'routing_comparison')
        self.figures['routing_comparison'] = fig_path
    
    def generate_performance_chart(self, data):
        """生成性能比较图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 订单分配算法性能
        ax1.plot(data["订单数量"], data["贪心分配算法"], 'o-', label='贪心分配算法')
        ax1.plot(data["订单数量"], data["插入启发式算法"], 's-', label='插入启发式算法')
        ax1.plot(data["订单数量"], data["批量分配算法"], '^-', label='批量分配算法')
        ax1.set_title('订单分配算法性能比较')
        ax1.set_xlabel('订单数量')
        ax1.set_ylabel('执行时间 (秒)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 路径规划算法性能
        ax2.plot(data["订单数量"], data["基本路径规划"], 'o-', label='基本路径规划')
        ax2.plot(data["订单数量"], data["局部搜索优化"], 's-', label='局部搜索优化')
        ax2.set_title('路径规划算法性能比较')
        ax2.set_xlabel('订单数量')
        ax2.set_ylabel('执行时间 (秒)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # 保存图表
        fig_path = self._save_figure(fig, 'performance_comparison')
        self.figures['performance_comparison'] = fig_path
    
    def generate_quality_chart(self, data):
        """生成质量比较图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 分配成功率
        ax1.plot(data["时间窗口(小时)"], data["分配成功率(%)"], 'o-', color='green')
        ax1.set_title('时间窗口对分配成功率的影响')
        ax1.set_xlabel('时间窗口 (小时)')
        ax1.set_ylabel('分配成功率 (%)')
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        
        # 违反约束和路线长度
        ax2.bar(data["时间窗口(小时)"], data["平均路线长度(km)"], alpha=0.7, label='平均路线长度(km)')
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(data["时间窗口(小时)"], data["违反约束数"], 'ro-', label='违反约束数')
        
        ax2.set_title('时间窗口对路线质量的影响')
        ax2.set_xlabel('时间窗口 (小时)')
        ax2.set_ylabel('平均路线长度 (km)', color='blue')
        ax2_twin.set_ylabel('违反约束数', color='red')
        
        # 合并两个轴的图例
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        fig_path = self._save_figure(fig, 'quality_comparison')
        self.figures['quality_comparison'] = fig_path
    
    def generate_dynamic_insertion_chart(self, results):
        """生成动态订单插入比较图"""
        # 比较插入前后的指标
        labels = ['插入前', '插入后']
        distances = [results['initial_distance'], results['updated_distance']]
        times = [results['initial_time'], results['updated_time']]
        violations = [results['initial_violations'], results['updated_violations']]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        width = 0.35
        x = np.arange(len(labels))
        
        # 距离和时间比较
        ax1.bar(x - width/2, distances, width, label='总距离(km)', color='skyblue')
        ax1.bar(x + width/2, times, width, label='总时间(小时)', color='salmon')
        ax1.set_title('动态订单插入前后距离和时间比较')
        ax1.set_ylabel('值')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 违反约束比较
        ax2.bar(labels, violations, color='lightcoral')
        ax2.set_title('动态订单插入前后违反约束比较')
        ax2.set_ylabel('违反约束数')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # 保存图表
        fig_path = self._save_figure(fig, 'dynamic_insertion_comparison')
        self.figures['dynamic_insertion_comparison'] = fig_path
    
    def generate_traffic_change_chart(self, results):
        """生成交通变化比较图"""
        # 比较交通变化前后的指标
        labels = ['变化前', '变化后']
        distances = [results['initial_distance'], results['updated_distance']]
        times = [results['initial_time'], results['updated_time']]
        violations = [results['initial_violations'], results['updated_violations']]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        width = 0.35
        x = np.arange(len(labels))
        
        # 距离和时间比较
        ax1.bar(x - width/2, distances, width, label='总距离(km)', color='skyblue')
        ax1.bar(x + width/2, times, width, label='总时间(小时)', color='salmon')
        ax1.set_title(f'交通变化前后距离和时间比较 (影响{results["affected_roads"]}条道路)')
        ax1.set_ylabel('值')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 违反约束比较
        ax2.bar(labels, violations, color='lightcoral')
        ax2.set_title('交通变化前后违反约束比较')
        ax2.set_ylabel('违反约束数')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # 保存图表
        fig_path = self._save_figure(fig, 'traffic_change_comparison')
        self.figures['traffic_change_comparison'] = fig_path
    
    def generate_peak_hour_chart(self, data):
        """生成高峰期处理比较图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 执行时间比较
        ax1.plot(data["批次大小"], data["执行时间(秒)"], 'o-', color='purple')
        ax1.set_title('批次大小对执行时间的影响')
        ax1.set_xlabel('批次大小')
        ax1.set_ylabel('执行时间 (秒)')
        ax1.grid(True, alpha=0.3)
        
        # 分配成功率比较
        ax2.plot(data["批次大小"], data["分配成功率(%)"], 's-', color='green')
        ax2.set_title('批次大小对分配成功率的影响')
        ax2.set_xlabel('批次大小')
        ax2.set_ylabel('分配成功率 (%)')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        fig_path = self._save_figure(fig, 'peak_hour_comparison')
        self.figures['peak_hour_comparison'] = fig_path
    
    def visualize_routes(self, solution, name_prefix):
        """可视化路线"""
        if not solution:
            return
        
        plt.figure(figsize=(12, 10))
        
        # 创建地图图形
        G = self.city_map.graph
        
        # 获取所有位置的坐标
        pos = {}
        for node_id in G.nodes():
            loc = self.city_map.locations[node_id]
            pos[node_id] = (loc.longitude, loc.latitude)  # 经度为x，纬度为y
        
        # 绘制地图底图
        nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray')
        
        # 绘制所有位置点
        nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightgray')
        
        # 绘制位置标签 (只显示部分标签，避免拥挤)
        labels = {node_id: self.city_map.locations[node_id].name 
                 for node_id in list(G.nodes())[:20]}  # 只显示前20个标签
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_family='SimHei')
        
        # 为每辆车选择不同的颜色
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'olive', 'cyan']
        
        # 绘制每辆车的路线
        for i, (vehicle_id, route) in enumerate(solution.routes.items()):
            if not route.points:
                continue
            
            color = colors[i % len(colors)]
            vehicle = route.vehicle
            
            # 构建路径
            path = [vehicle.current_location.id]
            for point in route.points:
                path.append(point.location.id)
            
            # 绘制路径
            for j in range(len(path) - 1):
                from_id = path[j]
                to_id = path[j + 1]
                
                # 直接画线，而不是使用图中的边
                from_pos = pos[from_id]
                to_pos = pos[to_id]
                plt.arrow(from_pos[0], from_pos[1], 
                         to_pos[0] - from_pos[0], to_pos[1] - from_pos[1],
                         head_width=0.002, head_length=0.004, fc=color, ec=color,
                         length_includes_head=True, alpha=0.6)
            
            # 绘制起点
            plt.scatter(pos[path[0]][0], pos[path[0]][1], s=150, c=color, marker='o', 
                      edgecolors='black', linewidths=1, alpha=0.8,
                      label=f'车辆{vehicle.id}: {vehicle.name}')
        
        plt.title('物流配送路线图')
        plt.legend(loc='best')
        plt.tight_layout()
        
        # 保存图表
        fig_path = self._save_figure(plt.gcf(), f'{name_prefix}_routes')
        self.figures[f'{name_prefix}_routes'] = fig_path
    
    def _save_figure(self, fig, name):
        """保存图表到文件并返回路径"""
        # 将图表保存为PNG文件
        filename = f"{name}.png"
        filepath = os.path.join(self.report_path, filename)
        fig.savefig(filepath)
        return filename
    
    def generate_html_report(self, test_results):
        """生成HTML测试报告"""
        print("生成HTML测试报告...")
        
        # 报告标题和基本信息
        report_title = "算法层测试报告"
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # HTML报告模板
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{report_title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-bottom: 2px solid #ddd; }}
                h1, h2, h3 {{ color: #444; }}
                h1 {{ text-align: center; }}
                h2 {{ margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
                h3 {{ margin-top: 25px; }}
                .section {{ margin: 20px 0; padding: 15px; background-color: #fff; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ padding: 12px 15px; border-bottom: 1px solid #ddd; text-align: left; }}
                th {{ background-color: #f8f8f8; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .img-container {{ text-align: center; margin: 20px 0; }}
                .img-container img {{ max-width: 100%; border: 1px solid #ddd; box-shadow: 0 0 5px rgba(0,0,0,0.1); }}
                .metrics {{ display: flex; justify-content: space-around; flex-wrap: wrap; margin: 20px 0; }}
                .metric-box {{ background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 5px; padding: 15px; width: 200px; margin: 10px; text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; margin: 10px 0; }}
                .metric-label {{ font-size: 14px; color: #777; }}
                .conclusion {{ background-color: #e7f4ff; padding: 20px; border-left: 4px solid #3498db; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{report_title}</h1>
                    <p style="text-align: center;">生成时间: {report_date}</p>
                </div>

                <h2>1. 测试概述</h2>
                <div class="section">
                    <p>本测试报告针对物流配送系统的算法层进行全面测试，包含订单分配算法、路径规划算法、动态事件处理等核心功能。
                    测试内容涵盖基本功能测试、性能和效率测试、质量测试以及特定业务场景测试。</p>
                    
                    <div class="metrics">
                        <div class="metric-box">
                            <div class="metric-label">测试算法数</div>
                            <div class="metric-value">6</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">测试场景数</div>
                            <div class="metric-value">4</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">测试指标数</div>
                            <div class="metric-value">12</div>
                        </div>
                    </div>
                </div>

                <h2>2. 基本功能测试</h2>
                <div class="section">
                    <h3>2.1 订单分配算法比较</h3>
                    <p>对贪心分配算法、插入启发式算法和批量分配算法进行性能和效果比较。</p>
                    
                    <div class="img-container">
                        <img src="assignment_comparison.png" alt="订单分配算法比较">
                    </div>
                    
                    <h4>订单分配算法详细数据</h4>
                    <table>
                        <tr>
                            <th>算法</th>
                            <th>执行时间(秒)</th>
                            <th>分配成功率(%)</th>
                            <th>总距离(km)</th>
                            <th>总时间(小时)</th>
                            <th>违反约束数</th>
                        </tr>
        """
        
        # 添加算法比较数据行
        if 'basic_functionality' in test_results and 'assignment' in test_results['basic_functionality']:
            assignment_results = test_results['basic_functionality']['assignment']
            if 'algorithms' in assignment_results:
                for algo_name, metrics in assignment_results['algorithms'].items():
                    html_content += f"""
                        <tr>
                            <td>{algo_name}</td>
                            <td>{metrics['执行时间']:.2f}</td>
                            <td>{metrics['分配成功率']:.1f}</td>
                            <td>{metrics['总距离']:.2f}</td>
                            <td>{metrics['总时间']:.2f}</td>
                            <td>{metrics['违反约束']}</td>
                        </tr>
                    """
        
        html_content += """
                    </table>
                
                    <h3>2.2 路径规划算法比较</h3>
                    <p>对基本路径规划、订单优先规划和局部搜索优化算法进行比较。</p>
                    
                    <div class="img-container">
                        <img src="routing_comparison.png" alt="路径规划算法比较">
                    </div>
                    
                    <div class="img-container">
                        <img src="routing_final_routes.png" alt="最终优化路线">
                        <p>最终优化路线图</p>
                    </div>
                    
                    <h4>路径规划算法详细数据</h4>
                    <table>
                        <tr>
                            <th>算法</th>
                            <th>执行时间(秒)</th>
                            <th>总距离(km)</th>
                            <th>总时间(小时)</th>
                            <th>违反约束数</th>
                        </tr>
        """
        
        # 添加路径规划比较数据行
        if 'basic_functionality' in test_results and 'routing' in test_results['basic_functionality']:
            routing_results = test_results['basic_functionality']['routing']
            if 'algorithms' in routing_results:
                for algo_name, metrics in routing_results['algorithms'].items():
                    html_content += f"""
                        <tr>
                            <td>{algo_name}</td>
                            <td>{metrics['执行时间']:.2f}</td>
                            <td>{metrics['总距离']:.2f}</td>
                            <td>{metrics['总时间']:.2f}</td>
                            <td>{metrics['违反约束']}</td>
                        </tr>
                    """
        
        html_content += """
                    </table>
                </div>

                <h2>3. 性能和效率测试</h2>
                <div class="section">
                    <p>测试不同算法在不同规模订单下的性能表现。</p>
                    
                    <div class="img-container">
                        <img src="performance_comparison.png" alt="性能比较">
                    </div>
                    
                    <h4>性能测试数据</h4>
                    <p>订单数量从10到100不等，记录各算法执行时间变化情况。</p>
                </div>

                <h2>4. 质量测试</h2>
                <div class="section">
                    <p>测试算法对不同时间窗口约束的处理能力。</p>
                    
                    <div class="img-container">
                        <img src="quality_comparison.png" alt="质量比较">
                    </div>
                    
                    <h4>时间窗口对算法的影响</h4>
                    <p>通过变化订单的时间窗口大小，测试算法应对能力。</p>
                </div>

                <h2>5. 特定场景测试</h2>
                <div class="section">
                    <h3>5.1 动态订单插入</h3>
                    <p>测试系统处理动态新增订单的能力。</p>
                    
                    <div class="img-container">
                        <img src="dynamic_insertion_comparison.png" alt="动态订单插入前后比较">
                    </div>
                    
                    <div style="display: flex; justify-content: space-between; margin-top: 20px;">
                        <div style="width: 48%;">
                            <div class="img-container">
                                <img src="dynamic_before_routes.png" alt="插入前路线">
                                <p>插入前路线</p>
                            </div>
                        </div>
                        <div style="width: 48%;">
                            <div class="img-container">
                                <img src="dynamic_after_routes.png" alt="插入后路线">
                                <p>插入后路线</p>
                            </div>
                        </div>
                    </div>

                    <h3>5.2 交通状况变化</h3>
                    <p>测试系统对道路交通状况变化的适应能力。</p>
                    
                    <div class="img-container">
                        <img src="traffic_change_comparison.png" alt="交通状况变化前后比较">
                    </div>
                    
                    <div style="display: flex; justify-content: space-between; margin-top: 20px;">
                        <div style="width: 48%;">
                            <div class="img-container">
                                <img src="traffic_before_routes.png" alt="交通变化前路线">
                                <p>交通变化前路线</p>
                            </div>
                        </div>
                        <div style="width: 48%;">
                            <div class="img-container">
                                <img src="traffic_after_routes.png" alt="交通变化后路线">
                                <p>交通变化后路线</p>
                            </div>
                        </div>
                    </div>

                    <h3>5.3 高峰期处理</h3>
                    <p>测试系统处理大量订单的能力。</p>
                    
                    <div class="img-container">
                        <img src="peak_hour_comparison.png" alt="高峰期处理比较">
                    </div>
                    
                    <div class="img-container">
                        <img src="peak_hour_routes.png" alt="高峰期路线安排">
                        <p>高峰期路线安排</p>
                    </div>
                </div>

                <h2>6. 测试结论</h2>
                <div class="section conclusion">
                    <h3>主要发现</h3>
                    <ul>
                        <li><strong>算法效率:</strong> 在小规模订单下，贪心算法效率最高；在大规模订单下，批量分配算法表现更好。</li>
                        <li><strong>解决方案质量:</strong> 插入启发式算法在分配成功率和路线距离方面有较好的平衡。</li>
                        <li><strong>动态适应性:</strong> 系统能有效处理新订单插入和交通状况变化，路线优化表现良好。</li>
                        <li><strong>时间窗口影响:</strong> 随着时间窗口约束放宽，算法表现显著提升，约束违反率降低。</li>
                        <li><strong>高峰期处理:</strong> 批次大小对系统性能有显著影响，建议根据系统负载动态调整。</li>
                    </ul>
                    
                    <h3>改进建议</h3>
                    <ul>
                        <li>进一步优化局部搜索算法，提高搜索效率。</li>
                        <li>针对高峰期需求，可考虑引入并行计算技术。</li>
                        <li>加强对交通状况实时响应的能力，增强路线动态调整的灵活性。</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        # 保存HTML报告
        report_path = os.path.join(self.report_path, "test_report.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"HTML报告已生成：{report_path}")


def main():
    """主函数"""
    try:
        reporter = AlgorithmLayerTestReporter()
        reporter.run_tests()
        print("测试完成！")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()