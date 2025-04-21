"""
动态优化器模块
负责处理新订单、交通变化等动态事件，调整和优化路线
"""
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import copy

from data_layer.order import Order
from data_layer.vehicle import Vehicle
from data_layer.city_map import CityMap

from algorithm_layer.base import Route, Solution, RoutePoint
from algorithm_layer.evaluation import RouteEvaluator, CostCalculator


class DynamicOptimizer:
    """动态优化器，处理实时事件并调整路线"""
    
    def __init__(self, city_map: CityMap):
        """
        Args:
            city_map: 城市地图对象
        """
        self.city_map = city_map
        self.route_evaluator = RouteEvaluator(city_map)
        self.cost_calculator = CostCalculator(city_map)
    
    def insert_new_order(self, solution: Solution, order: Order, 
                        current_time: datetime) -> Tuple[Solution, bool]:
        """尝试将新订单插入当前解决方案
        
        Args:
            solution: 当前解决方案
            order: 新订单
            current_time: 当前时间
            
        Returns:
            (更新后的解决方案, 是否成功插入)
        """
        best_solution = solution.copy()
        min_extra_cost = float('inf')
        inserted = False
        
        # 尝试将订单插入每辆车的路线
        for vehicle_id, route in solution.routes.items():
            # 检查车辆是否能容纳该订单
            if not route.vehicle.can_accommodate(order):
                continue
            
            # 计算添加订单前的成本
            original_cost = self.cost_calculator.calculate_route_cost(route, current_time)
            
            # 尝试所有可能的插入位置
            best_route = None
            best_pickup_pos = -1
            best_delivery_pos = -1
            best_route_cost = float('inf')
            
            for pickup_pos in range(len(route.points) + 1):
                # 送货位置必须在取货位置之后
                for delivery_pos in range(pickup_pos + 1, len(route.points) + 2):
                    # 创建临时路线进行评估
                    temp_route = copy.deepcopy(route)
                    
                    # 插入取货和送货点
                    pickup_point = RoutePoint(order.pickup_location, RoutePoint.ACTION_PICKUP, order)
                    delivery_point = RoutePoint(order.delivery_location, RoutePoint.ACTION_DELIVERY, order)
                    
                    temp_route.insert_point(pickup_pos, pickup_point)
                    temp_route.insert_point(delivery_pos, delivery_point)
                    
                    # 评估新路线
                    self.route_evaluator.evaluate_route(temp_route, current_time)
                    new_cost = self.cost_calculator.calculate_route_cost(temp_route, current_time)
                    
                    if new_cost < best_route_cost:
                        best_route = temp_route
                        best_pickup_pos = pickup_pos
                        best_delivery_pos = delivery_pos
                        best_route_cost = new_cost
            
            if best_route is not None:
                # 计算额外成本
                extra_cost = best_route_cost - original_cost
                
                if extra_cost < min_extra_cost:
                    min_extra_cost = extra_cost
                    best_solution = solution.copy()
                    best_solution.routes[vehicle_id] = best_route
                    inserted = True
        
        if not inserted:
            # 如果无法插入，标记为未分配订单
            best_solution = solution.copy()
            best_solution.add_unassigned_order(order.id)
        
        return best_solution, inserted
    
    def handle_traffic_update(self, solution: Solution, affected_roads: List[Tuple[int, int]], 
                           new_travel_times: Dict[Tuple[int, int], float],
                           current_time: datetime) -> Solution:
        """处理交通状况更新
        
        Args:
            solution: 当前解决方案
            affected_roads: 受影响的道路列表，每项为(起点ID, 终点ID)
            new_travel_times: 新的行驶时间，键为(起点ID, 终点ID)，值为新行驶时间(小时)
            current_time: 当前时间
            
        Returns:
            更新后的解决方案
        """
        # 更新城市地图中的路径行驶时间
        for (from_id, to_id), travel_time in new_travel_times.items():
            # 假设城市地图有这样的更新方法
            # 实际实现可能需要修改
            for edge in self.city_map.graph.edges:
                if edge[0] == from_id and edge[1] == to_id:
                    self.city_map.graph[from_id][to_id]['travel_time'] = travel_time
        
        # 重新评估所有路线
        updated_solution = solution.copy()
        for vehicle_id, route in updated_solution.routes.items():
            self.route_evaluator.evaluate_route(route, current_time)
        
        # 更新解决方案的总体指标
        updated_solution.update_metrics()
        
        return updated_solution
    
    def reoptimize_global(self, solution: Solution, orders: List[Order], 
                        vehicles: List[Vehicle], current_time: datetime) -> Solution:
        """全局重新优化
        
        在特定条件下（如积累了足够多的变化），进行全局重新优化
        
        Args:
            solution: 当前解决方案
            orders: 所有订单列表
            vehicles: 所有车辆列表
            current_time: 当前时间
            
        Returns:
            优化后的解决方案
        """
        # 这里可以调用协调器中的优化方法，或者实现自己的全局优化逻辑
        # 暂时简单返回当前解决方案
        return solution