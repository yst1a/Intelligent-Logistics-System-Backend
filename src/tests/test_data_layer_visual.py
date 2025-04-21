#!/usr/bin/env python3
"""
数据层可视化测试程序
"""

import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from matplotlib.lines import Line2D
from typing import List, Dict, Tuple, Optional

# 使用包装器模块导入必要的类
from data_layer_wrapper import (
    Location, Order, Vehicle, CityMap, DataLayer, USING_MOCK
)
# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Heiti SC']  
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题




class DataLayerVisualTester:
    """数据层可视化测试类"""
    
    def __init__(self, order_count: int = 15, vehicle_count: int = 5):
        """初始化测试器
        
        Args:
            order_count: 测试用订单数量
            vehicle_count: 测试用车辆数量
        """
        self.order_count = order_count
        self.vehicle_count = vehicle_count
        self.data_layer = DataLayer()
        self.test_results = []
        self.error_messages = []
        
        # 创建输出目录
        self.output_dir = "../test_results"
        os.makedirs(self.output_dir, exist_ok=True)

    def run_all_tests(self) -> bool:
        """运行所有测试并生成可视化报告
        
        Returns:
            测试是否全部通过
        """
        print("=== 开始数据层可视化测试 ===")
        
        # 1. 测试城市地图初始化和可视化
        map_test_result = self.test_city_map()
        self.test_results.append(("城市地图测试", map_test_result))
        
        # 2. 测试订单和车辆生成
        data_gen_test_result = self.test_data_generation()
        self.test_results.append(("数据生成测试", data_gen_test_result))
        
        # 3. 测试路径查询和距离计算
        routing_test_result = self.test_routing_and_distance()
        self.test_results.append(("路径和距离测试", routing_test_result))
        
        # 4. 测试订单分配
        assignment_test_result = self.test_order_assignment()
        self.test_results.append(("订单分配测试", assignment_test_result))
        
        # 5. 测试车辆移动和状态更新
        vehicle_test_result = self.test_vehicle_movement()
        self.test_results.append(("车辆移动测试", vehicle_test_result))
        
        # 6. 生成最终测试报告
        self.generate_test_report()
        
        # 检查是否所有测试都通过
        all_passed = all(result for _, result in self.test_results)
        status = "通过" if all_passed else "失败"
        print(f"=== 数据层测试完成: {status} ===")
        return all_passed

    def test_city_map(self) -> bool:
        """测试城市地图功能并可视化
        
        Returns:
            测试是否通过
        """
        print("测试城市地图...")
        try:
            # 初始化城市地图
            self.data_layer.initialize_city_map()
            city_map = self.data_layer.city_map
            
            # 验证基本属性
            if len(city_map.locations) != 15:
                self.error_messages.append(f"城市地图位置数量错误: {len(city_map.locations)} != 15")
                return False
                
            # 验证位置ID连续性
            location_ids = sorted(city_map.locations.keys())
            expected_ids = list(range(1, 16))
            if location_ids != expected_ids:
                self.error_messages.append(f"城市地图位置ID不连续: {location_ids}")
                return False
            
            # 验证图的连通性
            components = list(nx.weakly_connected_components(city_map.graph))
            if len(components) != 1:
                self.error_messages.append(f"城市地图不是连通的: {len(components)} 个连通分量")
                return False
            
            # 测试距离计算
            try:
                for i in range(1, 11):
                    for j in range(1, 11):
                        if i != j:
                            distance = city_map.get_distance(i, j)
                            if distance <= 0 and distance != float('inf'):
                                self.error_messages.append(f"距离计算错误: ({i},{j}) = {distance}")
                                return False
            except Exception as e:
                self.error_messages.append(f"测试距离计算时发生异常: {str(e)}")
                return False
            
            # 可视化城市地图
            self.visualize_city_map(city_map)
            return True
            
        except Exception as e:
            self.error_messages.append(f"城市地图测试异常: {str(e)}")
            return False

    def test_data_generation(self) -> bool:
        """测试数据生成功能并可视化
        
        Returns:
            测试是否通过
        """
        print("测试数据生成...")
        try:
            # 初始化测试数据
            self.data_layer.initialize_test_data(
                order_count=self.order_count,
                vehicle_count=self.vehicle_count
            )
            
            # 验证生成的订单和车辆
            if len(self.data_layer.orders) != self.order_count:
                self.error_messages.append(f"生成的订单数量错误: {len(self.data_layer.orders)} != {self.order_count}")
                return False
                
            if len(self.data_layer.vehicles) != self.vehicle_count:
                self.error_messages.append(f"生成的车辆数量错误: {len(self.data_layer.vehicles)} != {self.vehicle_count}")
                return False
            
            # 验证订单有效性
            for order in self.data_layer.orders.values():
                if order.pickup_location.id not in self.data_layer.city_map.locations:
                    self.error_messages.append(f"订单 {order.id} 取货位置无效: {order.pickup_location.id}")
                    return False
                if order.delivery_location.id not in self.data_layer.city_map.locations:
                    self.error_messages.append(f"订单 {order.id} 送货位置无效: {order.delivery_location.id}")
                    return False
                if order.latest_pickup_time <= order.earliest_pickup_time:
                    self.error_messages.append(f"订单 {order.id} 取货时间窗口无效")
                    return False
                if order.latest_delivery_time <= order.earliest_delivery_time:
                    self.error_messages.append(f"订单 {order.id} 送货时间窗口无效")
                    return False
            
            # 验证车辆有效性
            for vehicle in self.data_layer.vehicles.values():
                if vehicle.current_location.id not in self.data_layer.city_map.locations:
                    self.error_messages.append(f"车辆 {vehicle.id} 位置无效: {vehicle.current_location.id}")
                    return False
                if vehicle.max_volume <= 0 or vehicle.max_weight <= 0:
                    self.error_messages.append(f"车辆 {vehicle.id} 容量参数无效")
                    return False
            
            # 可视化订单
            self.visualize_orders(self.data_layer.orders.values())
            
            # 可视化车辆
            self.visualize_vehicles(self.data_layer.vehicles.values())
            
            return True
            
        except Exception as e:
            self.error_messages.append(f"数据生成测试异常: {str(e)}")
            return False

    def test_routing_and_distance(self) -> bool:
        """测试路径查询和距离计算功能并可视化
        
        Returns:
            测试是否通过
        """
        print("测试路径查询和距离计算...")
        try:
            city_map = self.data_layer.city_map
            
            # 测试随机路径对
            test_paths = []
            for _ in range(5):
                from_id = random.randint(1, 15)
                to_id = random.randint(1, 15)
                while to_id == from_id:
                    to_id = random.randint(1, 15)
                    
                try:
                    path = city_map.get_shortest_path(from_id, to_id)
                    distance = city_map.get_distance(from_id, to_id)
                    travel_time = city_map.get_travel_time(from_id, to_id)
                    
                    if not path:
                        self.error_messages.append(f"从 {from_id} 到 {to_id} 找不到路径")
                        return False
                        
                    if len(path) < 2:
                        self.error_messages.append(f"从 {from_id} 到 {to_id} 的路径长度小于2: {path}")
                        return False
                        
                    if distance <= 0:
                        self.error_messages.append(f"从 {from_id} 到 {to_id} 的距离无效: {distance}")
                        return False
                        
                    if travel_time <= 0:
                        self.error_messages.append(f"从 {from_id} 到 {to_id} 的行驶时间无效: {travel_time}")
                        return False
                    
                    test_paths.append((from_id, to_id, path, distance, travel_time))
                        
                except Exception as e:
                    self.error_messages.append(f"路径查询异常 ({from_id}->{to_id}): {str(e)}")
                    return False
            
            # 可视化路径
            self.visualize_paths(test_paths)
            
            return True
            
        except Exception as e:
            self.error_messages.append(f"路径和距离测试异常: {str(e)}")
            return False

    def test_order_assignment(self) -> bool:
        """测试订单分配功能并可视化
        
        Returns:
            测试是否通过
        """
        print("测试订单分配...")
        try:
            # 选择几个订单进行分配测试
            unassigned_orders = self.data_layer.get_unassigned_orders()
            vehicles = list(self.data_layer.vehicles.values())
            
            if not unassigned_orders:
                self.error_messages.append("没有未分配的订单可用于测试")
                return False
                
            if not vehicles:
                self.error_messages.append("没有车辆可用于测试")
                return False
            
            # 尝试分配3个订单或所有可用订单（取较小值）
            test_count = min(3, len(unassigned_orders))
            assignment_results = []
            
            for i in range(test_count):
                order = unassigned_orders[i]
                vehicle = vehicles[i % len(vehicles)]
                
                # 检查分配是否合理
                if vehicle.current_volume + order.volume > vehicle.max_volume:
                    # 跳过不合理的分配
                    continue
                    
                if vehicle.current_weight + order.weight > vehicle.max_weight:
                    # 跳过不合理的分配
                    continue
                
                # 执行分配
                result = self.data_layer.assign_order_to_vehicle(order.id, vehicle.id)
                
                if not result:
                    self.error_messages.append(f"订单 {order.id} 分配给车辆 {vehicle.id} 失败")
                    continue
                
                # 验证分配结果
                if order.assigned_vehicle_id != vehicle.id:
                    self.error_messages.append(f"订单 {order.id} 分配状态错误")
                    return False
                    
                if order not in vehicle.current_orders:
                    self.error_messages.append(f"订单 {order.id} 不在车辆 {vehicle.id} 的当前订单列表中")
                    return False
                
                assignment_results.append((order, vehicle))
            
            # 可能所有分配都不合理，但这不是错误
            if not assignment_results and test_count > 0:
                self.error_messages.append("没有成功的订单分配测试")
                return False
            
            # 可视化订单分配
            self.visualize_assignments(assignment_results)
            
            return True
            
        except Exception as e:
            self.error_messages.append(f"订单分配测试异常: {str(e)}")
            return False

    def test_vehicle_movement(self) -> bool:
        """测试车辆移动功能并可视化
        
        Returns:
            测试是否通过
        """
        print("测试车辆移动...")
        try:
            vehicles = list(self.data_layer.vehicles.values())
            if not vehicles:
                self.error_messages.append("没有车辆可用于测试")
                return False
            
            # 选择2个车辆进行移动测试
            test_count = min(2, len(vehicles))
            movement_results = []
            
            for i in range(test_count):
                vehicle = vehicles[i]
                original_location = vehicle.current_location
                
                # 随机选择一个新位置
                new_location_id = random.randint(1, 15)
                while new_location_id == original_location.id:
                    new_location_id = random.randint(1, 15)
                
                # 执行移动
                before_time = self.data_layer.current_time
                result = self.data_layer.update_vehicle_location(vehicle.id, new_location_id)
                
                if not result:
                    self.error_messages.append(f"车辆 {vehicle.id} 移动到位置 {new_location_id} 失败")
                    return False
                
                # 验证移动结果
                if vehicle.current_location.id != new_location_id:
                    self.error_messages.append(f"车辆 {vehicle.id} 位置更新后不匹配")
                    return False
                
                if vehicle.last_location_update_time != self.data_layer.current_time:
                    self.error_messages.append(f"车辆 {vehicle.id} 位置更新时间错误")
                    return False
                
                # 推进时间，模拟车辆行驶
                try:
                    travel_time = self.data_layer.city_map.get_travel_time(
                        original_location.id, new_location_id
                    )
                    self.data_layer.advance_time(travel_time * 60)  # 转换为分钟
                except Exception as e:
                    self.error_messages.append(f"计算行驶时间异常: {str(e)}")
                    return False
                
                movement_results.append((
                    vehicle, 
                    original_location, 
                    self.data_layer.city_map.locations[new_location_id],
                    before_time,
                    self.data_layer.current_time
                ))
            
            # 可视化车辆移动
            self.visualize_movements(movement_results)
            
            return True
            
        except Exception as e:
            self.error_messages.append(f"车辆移动测试异常: {str(e)}")
            return False

    def visualize_city_map(self, city_map: CityMap) -> None:
        """可视化城市地图
        
        Args:
            city_map: 城市地图对象
        """
        plt.figure(figsize=(12, 10))
        plt.title(f"{city_map.name}城市地图", fontsize=16)
        
        # 创建一个无向图来绘制（便于可视化）
        G = nx.Graph()
        
        # 添加节点
        for loc_id, location in city_map.locations.items():
            G.add_node(loc_id, 
                      pos=(location.longitude, location.latitude),
                      name=location.name)
            
        # 添加边
        for u, v, data in city_map.graph.edges(data=True):
            G.add_edge(u, v, weight=data['distance'])
        
        # 获取节点位置
        pos = nx.get_node_attributes(G, 'pos')
        
        # 绘制地图
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color='skyblue')
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7)
        
        # 添加节点标签
        labels = {node: f"{node}: {G.nodes[node]['name']}" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        # 添加边权标签
        edge_labels = {(u, v): f"{G[u][v]['weight']:.1f}km" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/city_map.png", dpi=300)
        plt.close()
        print(f"城市地图可视化已保存至 {self.output_dir}/city_map.png")

    def visualize_orders(self, orders: List[Order]) -> None:
        """可视化订单
        
        Args:
            orders: 订单列表
        """
        plt.figure(figsize=(12, 10))
        plt.title("订单分布图", fontsize=16)
        
        # 准备城市地图背景
        G = nx.Graph()
        city_map = self.data_layer.city_map
        
        # 添加节点
        for loc_id, location in city_map.locations.items():
            G.add_node(loc_id, pos=(location.longitude, location.latitude))
            
        # 添加边
        for u, v, data in city_map.graph.edges(data=True):
            G.add_edge(u, v)
        
        # 获取节点位置
        pos = nx.get_node_attributes(G, 'pos')
        
        # 绘制地图
        nx.draw_networkx_nodes(G, pos, node_size=200, node_color='lightgray', alpha=0.5)
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3)
        
        # 绘制订单（箭头从取货点指向送货点）
        for order in orders:
            start_pos = (order.pickup_location.longitude, order.pickup_location.latitude)
            end_pos = (order.delivery_location.longitude, order.delivery_location.latitude)
            
            plt.arrow(
                start_pos[0], start_pos[1],
                end_pos[0] - start_pos[0], end_pos[1] - start_pos[1],
                head_width=0.01, head_length=0.01, fc='red', ec='red',
                length_includes_head=True, alpha=0.7
            )
        
        # 添加图例
        plt.scatter([], [], color='red', label='订单（箭头方向：取货→送货）')
        plt.legend()
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/orders.png", dpi=300)
        plt.close()
        print(f"订单分布图已保存至 {self.output_dir}/orders.png")
        
        # 生成订单统计图
        self.visualize_order_stats(orders)

    def visualize_order_stats(self, orders: List[Order]) -> None:
        """可视化订单统计信息
        
        Args:
            orders: 订单列表
        """
        volumes = [order.volume for order in orders]
        weights = [order.weight for order in orders]
        
        # 创建一个2x2的子图布局
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("订单统计分析", fontsize=16)
        
        # 体积分布直方图
        axs[0, 0].hist(volumes, bins=10, alpha=0.7, color='blue')
        axs[0, 0].set_title("订单体积分布")
        axs[0, 0].set_xlabel("体积 (立方米)")
        axs[0, 0].set_ylabel("订单数量")
        
        # 重量分布直方图
        axs[0, 1].hist(weights, bins=10, alpha=0.7, color='green')
        axs[0, 1].set_title("订单重量分布")
        axs[0, 1].set_xlabel("重量 (千克)")
        axs[0, 1].set_ylabel("订单数量")
        
        # 体积与重量散点图
        axs[1, 0].scatter(volumes, weights, alpha=0.7, c='purple')
        axs[1, 0].set_title("订单体积与重量关系")
        axs[1, 0].set_xlabel("体积 (立方米)")
        axs[1, 0].set_ylabel("重量 (千克)")
        
        # 取货时间窗口分布图
        pickup_times = [
            (order.latest_pickup_time - order.earliest_pickup_time).total_seconds() / 3600 
            for order in orders
        ]
        axs[1, 1].hist(pickup_times, bins=10, alpha=0.7, color='orange')
        axs[1, 1].set_title("取货时间窗口分布")
        axs[1, 1].set_xlabel("时间窗口宽度 (小时)")
        axs[1, 1].set_ylabel("订单数量")
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/order_stats.png", dpi=300)
        plt.close()
        print(f"订单统计分析已保存至 {self.output_dir}/order_stats.png")

    def visualize_vehicles(self, vehicles: List[Vehicle]) -> None:
        """可视化车辆
        
        Args:
            vehicles: 车辆列表
        """
        plt.figure(figsize=(12, 10))
        plt.title("车辆分布图", fontsize=16)
        
        # 准备城市地图背景
        G = nx.Graph()
        city_map = self.data_layer.city_map
        
        # 添加节点
        for loc_id, location in city_map.locations.items():
            G.add_node(loc_id, pos=(location.longitude, location.latitude))
            
        # 添加边
        for u, v, data in city_map.graph.edges(data=True):
            G.add_edge(u, v)
        
        # 获取节点位置
        pos = nx.get_node_attributes(G, 'pos')
        
        # 绘制地图
        nx.draw_networkx_nodes(G, pos, node_size=200, node_color='lightgray', alpha=0.5)
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3)
        
        # 绘制车辆位置
        vehicle_types = {}  # 用于跟踪不同类型的车辆
        
        for vehicle in vehicles:
            vehicle_type = "小型货车" if "小型" in vehicle.name else (
                "中型货车" if "中型" in vehicle.name else "大型货车"
            )
            
            vehicle_types[vehicle_type] = vehicle_types.get(vehicle_type, 0) + 1
            
            color = 'blue' if "小型" in vehicle.name else (
                'green' if "中型" in vehicle.name else 'red'
            )
            
            v_pos = (vehicle.current_location.longitude, vehicle.current_location.latitude)
            plt.scatter(v_pos[0], v_pos[1], s=100, color=color, marker='s', alpha=0.8)
            
            # 添加车辆ID标签
            plt.text(v_pos[0], v_pos[1], str(vehicle.id), fontsize=8, 
                    ha='center', va='center', color='white')
        
        # 添加图例
        legend_elements = [
            mpatches.Patch(color='blue', label='小型货车'),
            mpatches.Patch(color='green', label='中型货车'),
            mpatches.Patch(color='red', label='大型货车')
        ]
        plt.legend(handles=legend_elements)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/vehicles.png", dpi=300)
        plt.close()
        print(f"车辆分布图已保存至 {self.output_dir}/vehicles.png")
        
        # 生成车辆统计图
        self.visualize_vehicle_stats(vehicles)

    def visualize_vehicle_stats(self, vehicles: List[Vehicle]) -> None:
        """可视化车辆统计信息
        
        Args:
            vehicles: 车辆列表
        """
        # 创建一个2x2的子图布局
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("车辆统计分析", fontsize=16)
        
        # 按类型统计车辆数量
        vehicle_types = {}
        for vehicle in vehicles:
            vehicle_type = "小型货车" if "小型" in vehicle.name else (
                "中型货车" if "中型" in vehicle.name else "大型货车"
            )
            vehicle_types[vehicle_type] = vehicle_types.get(vehicle_type, 0) + 1
        
        # 车辆类型分布饼图
        axs[0, 0].pie(
            vehicle_types.values(), 
            labels=vehicle_types.keys(),
            autopct='%1.1f%%',
            colors=['blue', 'green', 'red']
        )
        axs[0, 0].set_title("车辆类型分布")
        
        # 最大载重分布直方图
        weights = [vehicle.max_weight for vehicle in vehicles]
        axs[0, 1].hist(weights, bins=5, alpha=0.7, color='orange')
        axs[0, 1].set_title("车辆最大载重分布")
        axs[0, 1].set_xlabel("最大载重 (千克)")
        axs[0, 1].set_ylabel("车辆数量")
        
        # 最大容积分布直方图
        volumes = [vehicle.max_volume for vehicle in vehicles]
        axs[1, 0].hist(volumes, bins=5, alpha=0.7, color='purple')
        axs[1, 0].set_title("车辆最大容积分布")
        axs[1, 0].set_xlabel("最大容积 (立方米)")
        axs[1, 0].set_ylabel("车辆数量")
        
        # 车辆位置分布
        location_counts = {}
        for vehicle in vehicles:
            loc_name = vehicle.current_location.name
            location_counts[loc_name] = location_counts.get(loc_name, 0) + 1
        
        # 只显示前8个位置，避免图表过于拥挤
        top_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        axs[1, 1].bar(
            [loc[0] for loc in top_locations],
            [loc[1] for loc in top_locations],
            alpha=0.7,
            color='teal'
        )
        axs[1, 1].set_title("车辆位置分布 (前8位)")
        axs[1, 1].set_xlabel("位置")
        axs[1, 1].set_ylabel("车辆数量")
        axs[1, 1].tick_params(axis='x', rotation=45)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/vehicle_stats.png", dpi=300)
        plt.close()
        print(f"车辆统计分析已保存至 {self.output_dir}/vehicle_stats.png")

    def visualize_paths(self, path_data: List[Tuple[int, int, List[int], float, float]]) -> None:
        """可视化路径
        
        Args:
            path_data: 路径数据列表，每项包含 (起点ID, 终点ID, 路径, 距离, 时间)
        """
        plt.figure(figsize=(12, 10))
        plt.title("最短路径可视化", fontsize=16)
        
        # 准备城市地图背景
        G = nx.Graph()
        city_map = self.data_layer.city_map
        
        # 添加节点
        for loc_id, location in city_map.locations.items():
            G.add_node(loc_id, pos=(location.longitude, location.latitude))
            
        # 添加边
        for u, v, data in city_map.graph.edges(data=True):
            G.add_edge(u, v)
        
        # 获取节点位置
        pos = nx.get_node_attributes(G, 'pos')
        
        # 绘制地图
        nx.draw_networkx_nodes(G, pos, node_size=200, node_color='lightgray', alpha=0.5)
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3)
        
        # 绘制路径
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        for i, (from_id, to_id, path, distance, time) in enumerate(path_data):
            color = colors[i % len(colors)]
            
            # 绘制路径的边
            for j in range(len(path) - 1):
                u, v = path[j], path[j + 1]
                edge_pos = [pos[u], pos[v]]
                x_coords, y_coords = zip(*edge_pos)
                plt.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.8)
            
            # 标记起点和终点
            start_pos = pos[from_id]
            end_pos = pos[to_id]
            plt.scatter(start_pos[0], start_pos[1], s=100, color=color, marker='o')
            plt.scatter(end_pos[0], end_pos[1], s=100, color=color, marker='x')
            
            # 添加路径信息标签
            mid_pos = pos[path[len(path) // 2]]
            plt.text(
                mid_pos[0], mid_pos[1] + 0.01, 
                f"路径{i+1}: {from_id}->{to_id}\n距离: {distance:.1f}km\n时间: {time:.1f}h",
                fontsize=8, ha='center', va='bottom', color=color,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3')
            )
        
        # 添加图例
        legend_elements = [
            Line2D([0], [0], color=color, lw=2, label=f"路径{i+1}")
            for i, color in enumerate(colors[:len(path_data)])
        ]
        plt.legend(handles=legend_elements)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/paths.png", dpi=300)
        plt.close()
        print(f"路径可视化已保存至 {self.output_dir}/paths.png")

    def visualize_assignments(self, assignments: List[Tuple[Order, Vehicle]]) -> None:
        """可视化订单分配
        
        Args:
            assignments: 订单分配结果列表
        """
        plt.figure(figsize=(12, 10))
        plt.title("订单分配可视化", fontsize=16)
        
        # 准备城市地图背景
        G = nx.Graph()
        city_map = self.data_layer.city_map
        
        # 添加节点
        for loc_id, location in city_map.locations.items():
            G.add_node(loc_id, pos=(location.longitude, location.latitude))
            
        # 添加边
        for u, v, data in city_map.graph.edges(data=True):
            G.add_edge(u, v)
        
        # 获取节点位置
        pos = nx.get_node_attributes(G, 'pos')
        
        # 绘制地图
        nx.draw_networkx_nodes(G, pos, node_size=200, node_color='lightgray', alpha=0.5)
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3)
        
        # 绘制分配关系
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        for i, (order, vehicle) in enumerate(assignments):
            color = colors[i % len(colors)]
            
            # 绘制车辆位置
            v_pos = (vehicle.current_location.longitude, vehicle.current_location.latitude)
            plt.scatter(v_pos[0], v_pos[1], s=150, color=color, marker='s', alpha=0.8)
            plt.text(v_pos[0], v_pos[1], f"V{vehicle.id}", fontsize=8, ha='center', va='center', color='white')
            
            # 绘制订单取货点和送货点
            pickup_pos = (order.pickup_location.longitude, order.pickup_location.latitude)
            delivery_pos = (order.delivery_location.longitude, order.delivery_location.latitude)
            
            plt.scatter(pickup_pos[0], pickup_pos[1], s=100, color=color, marker='^', alpha=0.8)
            plt.scatter(delivery_pos[0], delivery_pos[1], s=100, color=color, marker='v', alpha=0.8)
            
            # 绘制车辆到取货点的连线（虚线）
            plt.plot(
                [v_pos[0], pickup_pos[0]], 
                [v_pos[1], pickup_pos[1]], 
                color=color, linestyle='--', linewidth=1.5, alpha=0.6
            )
            
            # 绘制取货点到送货点的连线（实线）
            plt.plot(
                [pickup_pos[0], delivery_pos[0]], 
                [pickup_pos[1], delivery_pos[1]], 
                color=color, linestyle='-', linewidth=1.5, alpha=0.6
            )
            
            # 添加订单信息标签
            plt.text(
                pickup_pos[0], pickup_pos[1] + 0.01, 
                f"O{order.id}取货点\n{order.volume:.1f}m³, {order.weight:.1f}kg",
                fontsize=7, ha='center', va='bottom', color=color,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
            )
            
            plt.text(
                delivery_pos[0], delivery_pos[1] - 0.01, 
                f"O{order.id}送货点",
                fontsize=7, ha='center', va='top', color=color,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
            )
        
        # 添加图例
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[i % len(colors)], 
                  markersize=10, label=f"车辆{assignments[i][1].id}")
            for i in range(len(assignments))
        ]
        legend_elements.append(Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
                             markersize=8, label='取货点'))
        legend_elements.append(Line2D([0], [0], marker='v', color='w', markerfacecolor='gray', 
                             markersize=8, label='送货点'))
        legend_elements.append(Line2D([0], [0], linestyle='--', color='gray', label='车辆→取货'))
        legend_elements.append(Line2D([0], [0], linestyle='-', color='gray', label='取货→送货'))
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/assignments.png", dpi=300)
        plt.close()
        print(f"订单分配可视化已保存至 {self.output_dir}/assignments.png")

    def visualize_movements(self, 
                           movements: List[Tuple[Vehicle, Location, Location, datetime, datetime]]) -> None:
        """可视化车辆移动
        
        Args:
            movements: 车辆移动数据列表，每项包含 (车辆, 原位置, 新位置, 开始时间, 结束时间)
        """
        plt.figure(figsize=(12, 10))
        plt.title("车辆移动可视化", fontsize=16)
        
        # 准备城市地图背景
        G = nx.Graph()
        city_map = self.data_layer.city_map
        
        # 添加节点
        for loc_id, location in city_map.locations.items():
            G.add_node(loc_id, pos=(location.longitude, location.latitude))
            
        # 添加边
        for u, v, data in city_map.graph.edges(data=True):
            G.add_edge(u, v)
        
        # 获取节点位置
        pos = nx.get_node_attributes(G, 'pos')
        
        # 绘制地图
        nx.draw_networkx_nodes(G, pos, node_size=200, node_color='lightgray', alpha=0.5)
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3)
        
        # 绘制车辆移动
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        for i, (vehicle, orig_loc, new_loc, start_time, end_time) in enumerate(movements):
            color = colors[i % len(colors)]
            
            # 原位置和新位置
            orig_pos = (orig_loc.longitude, orig_loc.latitude)
            new_pos = (new_loc.longitude, new_loc.latitude)
            
            # 绘制原位置（空心标记）
            plt.scatter(orig_pos[0], orig_pos[1], s=120, facecolors='white', 
                       edgecolors=color, linewidth=2, marker='o', alpha=0.8)
            
            # 绘制新位置（实心标记）
            plt.scatter(new_pos[0], new_pos[1], s=120, color=color, marker='o', alpha=0.8)
            
            # 绘制移动路径和箭头
            try:
                # 获取最短路径
                path = city_map.get_shortest_path(orig_loc.id, new_loc.id)
                
                # 如果能找到路径，绘制完整路径
                if path and len(path) > 1:
                    path_coords = []
                    for node_id in path:
                        node_pos = (city_map.locations[node_id].longitude, 
                                   city_map.locations[node_id].latitude)
                        path_coords.append(node_pos)
                    
                    # 绘制路径线
                    x_coords, y_coords = zip(*path_coords)
                    plt.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.7)
                    
                    # 在路径中间添加箭头
                    mid_idx = len(path_coords) // 2
                    if mid_idx > 0:
                        dx = path_coords[mid_idx][0] - path_coords[mid_idx-1][0]
                        dy = path_coords[mid_idx][1] - path_coords[mid_idx-1][1]
                        plt.arrow(
                            path_coords[mid_idx-1][0], path_coords[mid_idx-1][1],
                            dx, dy, head_width=0.01, head_length=0.01, 
                            fc=color, ec=color, alpha=0.8
                        )
            except Exception as e:
                # 如果找不到路径，只绘制直线
                plt.plot([orig_pos[0], new_pos[0]], [orig_pos[1], new_pos[1]], 
                        color=color, linestyle='--', linewidth=2, alpha=0.7)
            
            # 添加车辆标签
            plt.text(orig_pos[0], orig_pos[1] + 0.01, 
                    f"V{vehicle.id}原位置", fontsize=8, ha='center', va='bottom', color=color)
            plt.text(new_pos[0], new_pos[1] + 0.01, 
                    f"V{vehicle.id}新位置", fontsize=8, ha='center', va='bottom', color=color)
            
            # 计算移动时间
            time_diff = (end_time - start_time).total_seconds() / 60  # 分钟
            distance = orig_loc.distance_to(new_loc)
            
            # 添加移动信息标签
            mid_x = (orig_pos[0] + new_pos[0]) / 2
            mid_y = (orig_pos[1] + new_pos[1]) / 2
            plt.text(
                mid_x, mid_y + 0.02, 
                f"V{vehicle.id}: {orig_loc.name} → {new_loc.name}\n"
                f"距离: {distance:.1f}km, 时间: {time_diff:.1f}分钟",
                fontsize=8, ha='center', va='bottom', color=color,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3')
            )
        
        # 添加图例
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i % len(colors)], 
                  markersize=10, label=f"车辆{movements[i][0].id}")
            for i in range(len(movements))
        ]
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
                             markeredgecolor='gray', markersize=8, label='原位置'))
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                             markeredgecolor='gray', markersize=8, label='新位置'))
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/movements.png", dpi=300)
        plt.close()
        print(f"车辆移动可视化已保存至 {self.output_dir}/movements.png")

    def generate_test_report(self) -> None:
        """生成最终测试报告"""
        report_path = f"{self.output_dir}/test_report.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
            <title>数据层测试报告</title>
            <style>
                body {{ font-family: "Microsoft YaHei", "微软雅黑", Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; }}
                .result {{ margin: 10px 0; padding: 10px; border-radius: 5px; }}
                .pass {{ background-color: #dff0d8; color: #3c763d; }}
                .fail {{ background-color: #f2dede; color: #a94442; }}
                .image {{ margin: 20px 0; text-align: center; }}
                img {{ max-width: 100%; border: 1px solid #ddd; box-shadow: 0 0 5px rgba(0,0,0,0.2); }}
                .error {{ background-color: #fcf8e3; padding: 10px; border-left: 4px solid #f0ad4e; }}
            </style>
        </head>
        <body>
            <h1>数据层测试报告</h1>
            <p>测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>测试结果摘要</h2>
        """
        
        # 添加测试结果摘要
        all_passed = True
        for test_name, result in self.test_results:
            status = "通过" if result else "失败"
            status_class = "pass" if result else "fail"
            html_content += f"""
            <div class="result {status_class}">
                <strong>{test_name}:</strong> {status}
            </div>
            """
            if not result:
                all_passed = False
        
        # 添加总体测试结果
        overall_status = "通过" if all_passed else "失败"
        overall_class = "pass" if all_passed else "fail"
        html_content += f"""
        <div class="result {overall_class}">
            <strong>总体测试结果:</strong> {overall_status}
        </div>
        """
        
        # 添加错误信息（如果有）
        if self.error_messages:
            html_content += "<h2>错误信息</h2>"
            for error in self.error_messages:
                html_content += f'<div class="error">{error}</div>'
        
        # 添加可视化图像
        html_content += "<h2>测试可视化</h2>"
        
        images = [
            ("城市地图", "city_map.png", "显示城市地图布局、位置和道路网络"),
            ("订单分布", "orders.png", "显示所有订单的取货和送货位置"),
            ("订单统计", "order_stats.png", "订单体积、重量和时间窗口的统计分析"),
            ("车辆分布", "vehicles.png", "显示车辆在地图上的分布"),
            ("车辆统计", "vehicle_stats.png", "车辆类型、容量和位置的统计分析"),
            ("路径测试", "paths.png", "最短路径查询和距离计算的可视化"),
            ("订单分配", "assignments.png", "订单分配给车辆的可视化"),
            ("车辆移动", "movements.png", "车辆在路网中移动的可视化")
        ]
        
        for title, filename, description in images:
            if os.path.exists(f"{self.output_dir}/{filename}"):
                html_content += f"""
                <div class="image">
                    <h3>{title}</h3>
                    <p>{description}</p>
                    <img src="{filename}" alt="{title}">
                </div>
                """
        
        # 结束HTML文档
        html_content += """
        </body>
        </html>
        """
        
        # 写入报告文件，确保使用UTF-8编码
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"测试报告已生成: {report_path}")


if __name__ == "__main__":
    # 创建并运行测试
    tester = DataLayerVisualTester(order_count=15, vehicle_count=5)
    tester.run_all_tests()