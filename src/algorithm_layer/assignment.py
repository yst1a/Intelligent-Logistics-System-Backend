"""
订单分配算法模块
负责将订单分配给合适的车辆
"""
from typing import List, Dict, Tuple, Set, Optional
import random
from datetime import datetime

from data_layer.order import Order
from data_layer.vehicle import Vehicle
from data_layer.city_map import CityMap

from algorithm_layer.base import Route, RoutePoint, Solution, Algorithm
from algorithm_layer.evaluation import RouteEvaluator, CostCalculator, FeasibilityChecker


class GreedyAssignmentAlgorithm(Algorithm):
    """贪心分配算法
    
    使用贪心策略将订单分配给最近或最合适的车辆。
    适合低负载、简单场景的快速分配。
    """
    
    def __init__(self, city_map: CityMap):
        """
        Args:
            city_map: 城市地图对象
        """
        super().__init__(city_map)
        self.route_evaluator = RouteEvaluator(city_map)
        self.cost_calculator = CostCalculator(city_map)
    
    def solve(self, orders: List[Order], vehicles: List[Vehicle]) -> Solution:
        """使用贪心策略将订单分配给车辆
        
        Args:
            orders: 订单列表
            vehicles: 车辆列表
            
        Returns:
            分配解决方案
        """
        solution = Solution()
        current_time = datetime.now()
        
        # 初始化每个车辆的空路线
        for vehicle in vehicles:
            solution.add_route(Route(vehicle))
        
        # 按创建时间排序订单
        sorted_orders = sorted(orders, key=lambda o: o.creation_time)
        
        # 逐个处理订单
        for order in sorted_orders:
            best_vehicle_id = None
            min_extra_cost = float('inf')
            
            # 找到插入成本最小的车辆
            for vehicle in vehicles:
                # 检查车辆是否能容纳该订单
                if not vehicle.can_accommodate(order):
                    continue
                
                # 获取车辆当前路线
                route = solution.routes[vehicle.id]
                
                # 计算添加订单前的成本
                original_cost = self.cost_calculator.calculate_route_cost(route, current_time)
                
                # 添加订单到路线末尾
                pickup_point = RoutePoint(order.pickup_location, RoutePoint.ACTION_PICKUP, order)
                delivery_point = RoutePoint(order.delivery_location, RoutePoint.ACTION_DELIVERY, order)
                route.add_point(pickup_point)
                route.add_point(delivery_point)
                
                # 计算添加订单后的成本
                self.route_evaluator.evaluate_route(route, current_time)
                new_cost = self.cost_calculator.calculate_route_cost(route, current_time)
                
                # 计算额外成本
                extra_cost = new_cost - original_cost
                
                # 如果这是最佳选择，更新最佳车辆
                if extra_cost < min_extra_cost:
                    min_extra_cost = extra_cost
                    best_vehicle_id = vehicle.id
                
                # 从路线中移除订单点，恢复原状
                route.points.pop()
                route.points.pop()
                self.route_evaluator.evaluate_route(route, current_time)
            
            # 将订单分配给最佳车辆，或标记为未分配
            if best_vehicle_id is not None:
                best_vehicle = next(v for v in vehicles if v.id == best_vehicle_id)
                best_route = solution.routes[best_vehicle_id]
                
                # 将订单添加到最佳车辆的路线
                pickup_point = RoutePoint(order.pickup_location, RoutePoint.ACTION_PICKUP, order)
                delivery_point = RoutePoint(order.delivery_location, RoutePoint.ACTION_DELIVERY, order)
                best_route.add_point(pickup_point)
                best_route.add_point(delivery_point)
                
                # 更新路线评价指标
                self.route_evaluator.evaluate_route(best_route, current_time)
                
                # 更新车辆状态
                best_vehicle.add_order(order)
            else:
                # 标记为未分配订单
                solution.add_unassigned_order(order.id)
        
        # 更新解决方案的总体指标
        solution.update_metrics()
        return solution


class InsertionHeuristicAlgorithm(Algorithm):
    """插入启发式算法
    
    使用插入启发式策略优化订单分配和路径规划。
    对每个订单，尝试在每辆车的当前路线中找到最佳插入位置。
    """
    
    def __init__(self, city_map: CityMap):
        """
        Args:
            city_map: 城市地图对象
        """
        super().__init__(city_map)
        self.route_evaluator = RouteEvaluator(city_map)
        self.cost_calculator = CostCalculator(city_map)
        self.feasibility_checker = FeasibilityChecker(city_map)
    
    def solve(self, orders: List[Order], vehicles: List[Vehicle]) -> Solution:
        """使用插入启发式策略解决问题
        
        Args:
            orders: 订单列表
            vehicles: 车辆列表
            
        Returns:
            求解得到的方案
        """
        solution = Solution()
        current_time = datetime.now()
        
        # 初始化每个车辆的空路线
        for vehicle in vehicles:
            solution.add_route(Route(vehicle))
        
        # 按创建时间排序订单
        sorted_orders = sorted(orders, key=lambda o: o.creation_time)
        
        # 逐个处理订单
        for order in sorted_orders:
            best_vehicle_id = None
            best_pickup_pos = -1
            best_delivery_pos = -1
            min_extra_cost = float('inf')
            
            # 为每辆车找到最佳插入位置
            for vehicle in vehicles:
                # 检查车辆是否能容纳该订单
                if not vehicle.can_accommodate(order):
                    continue
                
                # 获取车辆当前路线
                route = solution.routes[vehicle.id]
                
                # 计算添加订单前的成本
                original_cost = self.cost_calculator.calculate_route_cost(route, current_time)
                
                # 尝试所有可能的插入位置
                for pickup_pos in range(len(route.points) + 1):
                    # 送货位置必须在取货位置之后
                    for delivery_pos in range(pickup_pos + 1, len(route.points) + 2):
                        # 创建临时路线进行评估
                        temp_route = route.copy()
                        
                        # 插入取货和送货点
                        pickup_point = RoutePoint(order.pickup_location, RoutePoint.ACTION_PICKUP, order)
                        delivery_point = RoutePoint(order.delivery_location, RoutePoint.ACTION_DELIVERY, order)
                        
                        temp_route.insert_point(pickup_pos, pickup_point)
                        temp_route.insert_point(delivery_pos, delivery_point)
                        
                        # 检查路线可行性
                        is_feasible, _ = self.feasibility_checker.is_route_feasible(temp_route, current_time)
                        if not is_feasible:
                            continue
                        
                        # 评估新路线成本
                        self.route_evaluator.evaluate_route(temp_route, current_time)
                        new_cost = self.cost_calculator.calculate_route_cost(temp_route, current_time)
                        
                        # 计算额外成本
                        extra_cost = new_cost - original_cost
                        
                        # 如果这是最佳选择，更新最佳插入位置
                        if extra_cost < min_extra_cost:
                            min_extra_cost = extra_cost
                            best_vehicle_id = vehicle.id
                            best_pickup_pos = pickup_pos
                            best_delivery_pos = delivery_pos
            
            # 将订单分配给最佳车辆的最佳位置，或标记为未分配
            if best_vehicle_id is not None:
                best_vehicle = next(v for v in vehicles if v.id == best_vehicle_id)
                best_route = solution.routes[best_vehicle_id]
                
                # 插入订单到最佳位置
                pickup_point = RoutePoint(order.pickup_location, RoutePoint.ACTION_PICKUP, order)
                delivery_point = RoutePoint(order.delivery_location, RoutePoint.ACTION_DELIVERY, order)
                
                # 注意先插入后面的点，避免索引变化
                if best_delivery_pos > best_pickup_pos:
                    best_route.insert_point(best_delivery_pos, delivery_point)
                    best_route.insert_point(best_pickup_pos, pickup_point)
                else:
                    best_route.insert_point(best_pickup_pos, pickup_point)
                    best_route.insert_point(best_delivery_pos, delivery_point)
                
                # 更新路线评价指标
                self.route_evaluator.evaluate_route(best_route, current_time)
                
                # 更新车辆状态
                best_vehicle.add_order(order)
            else:
                # 标记为未分配订单
                solution.add_unassigned_order(order.id)
        
        # 更新解决方案的总体指标
        solution.update_metrics()
        return solution


class BatchAssignmentAlgorithm(Algorithm):
    """批量分配算法
    
    收集一批订单后一起优化分配，适合高峰期使用。
    使用贪婪构造+局部搜索优化策略。
    """
    
    def __init__(self, city_map: CityMap, batch_size: int = 10):
        """
        Args:
            city_map: 城市地图对象
            batch_size: 每批处理的订单数，默认为10
        """
        super().__init__(city_map)
        self.route_evaluator = RouteEvaluator(city_map)
        self.cost_calculator = CostCalculator(city_map)
        self.feasibility_checker = FeasibilityChecker(city_map)
        self.batch_size = batch_size
        
        # 使用插入启发式算法处理单个批次
        self.insertion_algorithm = InsertionHeuristicAlgorithm(city_map)
    
    def solve(self, orders: List[Order], vehicles: List[Vehicle]) -> Solution:
        """使用批量处理策略解决问题
        
        Args:
            orders: 订单列表
            vehicles: 车辆列表
            
        Returns:
            求解得到的方案
        """
        solution = Solution()
        current_time = datetime.now()
        
        # 初始化每个车辆的空路线
        for vehicle in vehicles:
            solution.add_route(Route(vehicle))
        
        # 按创建时间排序订单
        sorted_orders = sorted(orders, key=lambda o: o.creation_time)
        
        # 按批次处理订单
        for i in range(0, len(sorted_orders), self.batch_size):
            # 获取当前批次的订单
            batch_orders = sorted_orders[i:i+self.batch_size]
            
            # 处理这一批订单
            self._process_batch(batch_orders, vehicles, solution, current_time)
        
        # 更新解决方案的总体指标
        solution.update_metrics()
        return solution
    
    def _process_batch(self, batch_orders: List[Order], vehicles: List[Vehicle], 
                      solution: Solution, current_time: datetime) -> None:
        """处理一批订单
        
        Args:
            batch_orders: 当前批次的订单列表
            vehicles: 车辆列表
            solution: 当前解决方案（会被修改）
            current_time: 当前时间
        """
        # 对于大批次，尝试不同的订单排序
        if len(batch_orders) > 5:
            # 创建多个排序方案
            ordering_options = [
                # 按创建时间排序
                sorted(batch_orders, key=lambda o: o.creation_time),
                # 按最晚取货时间排序
                sorted(batch_orders, key=lambda o: o.latest_pickup_time),
                # 按货物体积降序排序
                sorted(batch_orders, key=lambda o: o.volume, reverse=True),
                # 按取货点与送货点距离排序
                sorted(batch_orders, key=lambda o: o.pickup_location.distance_to(o.delivery_location))
            ]
            
            best_solution = None
            best_cost = float('inf')
            
            # 尝试每种排序方案
            for ordered_batch in ordering_options:
                # 创建临时解决方案的深拷贝
                temp_solution = solution.copy()
                
                # 使用插入启发式算法处理这批订单
                self._assign_batch_with_insertion(ordered_batch, vehicles, temp_solution, current_time)
                
                # 评估该解决方案
                self.route_evaluator.evaluate_solution(temp_solution, current_time)
                cost = self.cost_calculator.calculate_solution_cost(temp_solution, current_time)
                
                # 如果这是最佳解决方案，保存它
                if cost < best_cost:
                    best_cost = cost
                    best_solution = temp_solution
            
            # 使用最佳解决方案
            if best_solution:
                # 将最佳解决方案的路线复制到原始解决方案
                for vehicle_id, route in best_solution.routes.items():
                    solution.routes[vehicle_id] = route
                
                # 更新未分配订单
                solution.unassigned_orders = best_solution.unassigned_orders.copy()
        else:
            # 对于小批次，直接使用插入启发式
            self._assign_batch_with_insertion(batch_orders, vehicles, solution, current_time)
    
    def _assign_batch_with_insertion(self, batch_orders: List[Order], vehicles: List[Vehicle],
                                   solution: Solution, current_time: datetime) -> None:
        """使用插入启发式算法分配一批订单
        
        Args:
            batch_orders: 订单批次
            vehicles: 车辆列表
            solution: 当前解决方案（会被修改）
            current_time: 当前时间
        """
        for order in batch_orders:
            best_vehicle_id = None
            best_pickup_pos = -1
            best_delivery_pos = -1
            min_extra_cost = float('inf')
            
            # 为每辆车找到最佳插入位置
            for vehicle in vehicles:
                # 检查车辆是否能容纳该订单
                if not vehicle.can_accommodate(order):
                    continue
                
                # 获取车辆当前路线
                route = solution.routes[vehicle.id]
                
                # 计算添加订单前的成本
                original_cost = self.cost_calculator.calculate_route_cost(route, current_time)
                
                # 尝试所有可能的插入位置
                for pickup_pos in range(len(route.points) + 1):
                    # 送货位置必须在取货位置之后
                    for delivery_pos in range(pickup_pos + 1, len(route.points) + 2):
                        # 创建临时路线进行评估
                        temp_route = route.copy()
                        
                        # 插入取货和送货点
                        pickup_point = RoutePoint(order.pickup_location, RoutePoint.ACTION_PICKUP, order)
                        delivery_point = RoutePoint(order.delivery_location, RoutePoint.ACTION_DELIVERY, order)
                        
                        temp_route.insert_point(pickup_pos, pickup_point)
                        temp_route.insert_point(delivery_pos, delivery_point)
                        
                        # 检查路线可行性
                        is_feasible, _ = self.feasibility_checker.is_route_feasible(temp_route, current_time)
                        if not is_feasible:
                            continue
                        
                        # 评估新路线成本
                        self.route_evaluator.evaluate_route(temp_route, current_time)
                        new_cost = self.cost_calculator.calculate_route_cost(temp_route, current_time)
                        
                        # 计算额外成本
                        extra_cost = new_cost - original_cost
                        
                        # 如果这是最佳选择，更新最佳插入位置
                        if extra_cost < min_extra_cost:
                            min_extra_cost = extra_cost
                            best_vehicle_id = vehicle.id
                            best_pickup_pos = pickup_pos
                            best_delivery_pos = delivery_pos
            
            # 将订单分配给最佳车辆的最佳位置，或标记为未分配
            if best_vehicle_id is not None:
                best_vehicle = next(v for v in vehicles if v.id == best_vehicle_id)
                best_route = solution.routes[best_vehicle_id]
                
                # 插入订单到最佳位置
                pickup_point = RoutePoint(order.pickup_location, RoutePoint.ACTION_PICKUP, order)
                delivery_point = RoutePoint(order.delivery_location, RoutePoint.ACTION_DELIVERY, order)
                
                # 注意先插入后面的点，避免索引变化
                if best_delivery_pos > best_pickup_pos:
                    best_route.insert_point(best_delivery_pos, delivery_point)
                    best_route.insert_point(best_pickup_pos, pickup_point)
                else:
                    best_route.insert_point(best_pickup_pos, pickup_point)
                    best_route.insert_point(best_delivery_pos, delivery_point)
                
                # 更新路线评价指标
                self.route_evaluator.evaluate_route(best_route, current_time)
                
                # 更新车辆状态
                best_vehicle.add_order(order)
                
                # 从未分配列表中移除
                solution.remove_unassigned_order(order.id)
            else:
                # 标记为未分配订单
                solution.add_unassigned_order(order.id)