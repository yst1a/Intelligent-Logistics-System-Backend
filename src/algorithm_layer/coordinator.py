"""
算法协调器模块
作为算法层的主入口，协调各算法组件的工作
"""
from typing import List, Dict, Optional, Tuple
import time
from datetime import datetime
import logging

from data_layer.order import Order
from data_layer.vehicle import Vehicle
from data_layer.city_map import CityMap
from data_layer.data_layer import DataLayer

from algorithm_layer.base import Solution, Route, RoutePoint
from algorithm_layer.assignment import GreedyAssignmentAlgorithm, InsertionHeuristicAlgorithm, BatchAssignmentAlgorithm
from algorithm_layer.routing import BasicRoutingAlgorithm, OrderFirstRoutingAlgorithm, LocalSearchRoutingAlgorithm
from algorithm_layer.evaluation import RouteEvaluator, CostCalculator, FeasibilityChecker


class AlgorithmCoordinator:
    """算法协调器，作为算法层的主入口，协调各算法组件的工作"""
    
    def __init__(self, city_map: CityMap):
        """初始化算法协调器
        
        Args:
            city_map: 城市地图对象
        """
        self.city_map = city_map
        
        # 初始化各个算法组件
        self.route_evaluator = RouteEvaluator(city_map)
        self.cost_calculator = CostCalculator(city_map)
        self.feasibility_checker = FeasibilityChecker(city_map)
        
        # 订单分配算法
        self.greedy_assignment = GreedyAssignmentAlgorithm(city_map)
        self.insertion_assignment = InsertionHeuristicAlgorithm(city_map)
        self.batch_assignment = BatchAssignmentAlgorithm(city_map)
        
        # 路径规划算法
        self.basic_routing = BasicRoutingAlgorithm(city_map)
        self.order_first_routing = OrderFirstRoutingAlgorithm(city_map)
        self.local_search_routing = LocalSearchRoutingAlgorithm(city_map)
        
        # 当前解决方案
        self.current_solution: Optional[Solution] = None
        
        # 日志记录器
        self.logger = logging.getLogger(__name__)
    
    def process_new_orders(self, new_orders: List[Order], available_vehicles: List[Vehicle], 
                         current_time: datetime, method: str = 'insertion') -> Solution:
        """处理新订单
        
        Args:
            new_orders: 新订单列表
            available_vehicles: 可用车辆列表
            current_time: 当前时间
            method: 使用的分配算法，可选 'greedy', 'insertion', 'batch'
            
        Returns:
            更新后的解决方案
        """
        self.logger.info(f"处理 {len(new_orders)} 个新订单，使用 {method} 算法")
        
        # 根据指定方法选择算法
        if method == 'greedy':
            assignment_algo = self.greedy_assignment
        elif method == 'batch':
            assignment_algo = self.batch_assignment
        else:  # default to insertion
            assignment_algo = self.insertion_assignment
        
        # 如果当前没有解决方案，创建新的解决方案
        if not self.current_solution:
            self.current_solution = assignment_algo.solve(new_orders, available_vehicles)
            return self.current_solution
        
        # 否则，将新订单添加到当前解决方案
        temp_solution = self.current_solution.copy()
        
        # 将已分配订单转换为Order对象列表
        assigned_orders = []
        for route in temp_solution.routes.values():
            # 获取路线中的所有不重复订单
            order_ids = set()
            for point in route.points:
                if point.order.id not in order_ids:
                    order_ids.add(point.order.id)
                    assigned_orders.append(point.order)
        
        # 合并已分配订单和新订单
        all_orders = assigned_orders + new_orders
        
        # 使用选择的算法重新求解
        new_solution = assignment_algo.solve(all_orders, available_vehicles)
        
        # 更新当前解决方案
        self.current_solution = new_solution
        return new_solution
    
    def optimize_routes(self, solution: Solution, current_time: datetime, 
                      method: str = 'local_search') -> Solution:
        """优化路线
        
        Args:
            solution: 要优化的解决方案
            current_time: 当前时间
            method: 使用的路径规划算法，可选 'basic', 'order_first', 'local_search'
            
        Returns:
            优化后的解决方案
        """
        self.logger.info(f"优化路线，使用 {method} 算法")
        
        # 从解决方案中提取所有订单和车辆
        orders = []
        vehicles = []
        
        # 获取所有订单
        order_ids = set()
        for route in solution.routes.values():
            vehicle = route.vehicle
            vehicles.append(vehicle)
            
            for point in route.points:
                if point.order.id not in order_ids:
                    order_ids.add(point.order.id)
                    orders.append(point.order)
        
        # 选择算法并优化
        if method == 'basic':
            routing_algo = self.basic_routing
        elif method == 'order_first':
            routing_algo = self.order_first_routing
        else:  # default to local_search
            routing_algo = self.local_search_routing
        
        # 使用选择的算法优化路线
        optimized_solution = routing_algo.solve(orders, vehicles)
        
        # 添加未分配订单
        for order_id in solution.unassigned_orders:
            optimized_solution.add_unassigned_order(order_id)
        
        return optimized_solution
    
    def handle_dynamic_event(self, event_type: str, event_data: Dict, current_time: datetime) -> Solution:
        """处理动态事件
        
        Args:
            event_type: 事件类型，如 'new_order', 'traffic_update', 'vehicle_breakdown'
            event_data: 事件数据，包含事件的具体信息
            current_time: 当前时间
            
        Returns:
            更新后的解决方案
        """
        self.logger.info(f"处理动态事件: {event_type}")
        
        if event_type == 'new_order':
            # 处理新订单
            new_order = event_data.get('order')
            if new_order:
                return self.process_new_orders([new_order], list(event_data.get('vehicles', [])), current_time)
        
        elif event_type == 'traffic_update':
            # 处理交通更新
            # 这里需要根据交通更新重新评估和优化路线
            # 暂时简单实现，后续可以扩展
            if self.current_solution:
                return self.optimize_routes(self.current_solution, current_time)
        
        elif event_type == 'vehicle_breakdown':
            # 处理车辆故障
            # 需要重新分配受影响车辆的订单
            # 暂时简单实现，后续可以扩展
            if self.current_solution and 'vehicle_id' in event_data:
                vehicle_id = event_data['vehicle_id']
                affected_orders = []
                
                # 收集受影响车辆的订单
                if vehicle_id in self.current_solution.routes:
                    route = self.current_solution.routes[vehicle_id]
                    for point in route.points:
                        if point.is_pickup and point.order not in affected_orders:
                            affected_orders.append(point.order)
                
                # 移除受影响车辆的路线
                new_solution = self.current_solution.copy()
                if vehicle_id in new_solution.routes:
                    del new_solution.routes[vehicle_id]
                
                # 获取可用车辆
                available_vehicles = [v for v in event_data.get('vehicles', []) 
                                     if v.id != vehicle_id]
                
                # 重新分配受影响的订单
                return self.process_new_orders(affected_orders, available_vehicles, current_time)
        
        # 对于其他事件或没有有效事件数据，返回当前解决方案
        return self.current_solution if self.current_solution else Solution()
    
    def calculate_price(self, order: Order, distance: float = None, 
                      urgency_factor: float = 1.0) -> float:
        """计算订单价格
        
        Args:
            order: 订单对象
            distance: 行驶距离，如果为None则使用直线距离
            urgency_factor: 紧急程度因子，越高价格越高
            
        Returns:
            订单价格
        """
        # 如果未指定距离，计算直线距离
        if distance is None:
            distance = order.pickup_location.distance_to(order.delivery_location)
        
        # 基础价格计算
        base_fee = 10.0  # 起步价
        distance_fee = distance * 2.0  # 每公里2元
        
        # 根据货物体积和重量计算附加费
        volume_fee = order.volume * 5.0  # 每立方米5元
        weight_fee = order.weight * 0.1  # 每公斤0.1元
        
        # 时间窗口紧急程度计算
        time_window = (order.latest_delivery_time - order.earliest_pickup_time).total_seconds() / 3600.0
        time_factor = max(1.0, 5.0 / max(1.0, time_window))  # 时间窗口越紧，因子越大
        
        # 最终价格
        price = (base_fee + distance_fee + volume_fee + weight_fee) * time_factor * urgency_factor
        
        return round(price, 2)  # 四舍五入到分
    
    def get_current_solution(self) -> Optional[Solution]:
        """获取当前解决方案"""
        return self.current_solution