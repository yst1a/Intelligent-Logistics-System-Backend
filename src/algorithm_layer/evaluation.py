"""
评价函数模块
负责评估路线和解决方案的质量
"""
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime, timedelta

from data_layer.order import Order
from data_layer.vehicle import Vehicle
from data_layer.location import Location
from data_layer.city_map import CityMap

from algorithm_layer.base import Route, RoutePoint, Solution


class RouteEvaluator:
    """路线评价器，计算路线的各种指标"""
    
    def __init__(self, city_map: CityMap):
        """
        Args:
            city_map: 城市地图对象
        """
        self.city_map = city_map
    
    def evaluate_route(self, route: Route, current_time: datetime) -> None:
        """评估路线的各项指标
        
        Args:
            route: 要评估的路线
            current_time: 当前时间
        """
        if len(route.points) == 0:
            route.total_distance = 0.0
            route.total_time = 0.0
            route.violations = 0
            return
        
        # 重新计算路线指标
        total_distance = 0.0
        total_time = 0.0
        violations = 0
        
        # 跟踪车辆当前位置、时间和载荷
        current_location = route.vehicle.current_location
        current_vehicle_time = current_time
        current_volume = route.vehicle.current_volume
        current_weight = route.vehicle.current_weight
        
        # 跟踪已取货但未送达的订单
        picked_not_delivered = set()
        
        # 计算每个路线点的到达时间和相关指标
        for i, point in enumerate(route.points):
            # 计算到下一个点的距离和时间
            try:
                distance = self.city_map.get_distance(current_location.id, point.location.id)
                travel_time = distance / route.vehicle.average_speed  # 小时
            except ValueError:
                # 如果没有路径，使用直线距离作为估计
                distance = current_location.distance_to(point.location)
                travel_time = distance / route.vehicle.average_speed
                violations += 1  # 无路径视为一次违反约束
            
            # 更新累计距离和时间
            total_distance += distance
            total_time += travel_time
            
            # 计算到达时间
            arrival_time = current_vehicle_time + timedelta(hours=travel_time)
            point.arrival_time = arrival_time
            
            # 检查时间窗约束
            if point.is_pickup:
                if arrival_time < point.order.earliest_pickup_time:
                    # 早到，需要等待
                    wait_time = (point.order.earliest_pickup_time - arrival_time).total_seconds() / 3600.0
                    total_time += wait_time
                    arrival_time = point.order.earliest_pickup_time
                    point.arrival_time = arrival_time
                elif arrival_time > point.order.latest_pickup_time:
                    # 晚到，违反约束
                    violations += 1
                
                # 取货后更新载荷
                current_volume += point.order.volume
                current_weight += point.order.weight
                picked_not_delivered.add(point.order.id)
                
                # 检查车辆容量约束
                if current_volume > route.vehicle.max_volume:
                    violations += 1
                if current_weight > route.vehicle.max_weight:
                    violations += 1
            else:  # 送货点
                if arrival_time < point.order.earliest_delivery_time:
                    # 早到，需要等待
                    wait_time = (point.order.earliest_delivery_time - arrival_time).total_seconds() / 3600.0
                    total_time += wait_time
                    arrival_time = point.order.earliest_delivery_time
                    point.arrival_time = arrival_time
                elif arrival_time > point.order.latest_delivery_time:
                    # 晚到，违反约束
                    violations += 1
                
                # 送货后更新载荷
                current_volume -= point.order.volume
                current_weight -= point.order.weight
                
                # 检查送货先后顺序约束
                if point.order.id not in picked_not_delivered:
                    violations += 1
                else:
                    picked_not_delivered.remove(point.order.id)
            
            # 更新当前位置和时间
            current_location = point.location
            current_vehicle_time = arrival_time
        
        # 更新路线指标
        route.total_distance = total_distance
        route.total_time = total_time
        route.violations = violations
    
    def evaluate_solution(self, solution: Solution, current_time: datetime) -> None:
        """评估整体解决方案的各项指标
        
        Args:
            solution: 要评估的解决方案
            current_time: 当前时间
        """
        # 评估每条路线
        for route in solution.routes.values():
            self.evaluate_route(route, current_time)
        
        # 更新解决方案的总体指标
        solution.update_metrics()


class CostCalculator:
    """成本计算器，计算路线和解决方案的成本"""
    
    def __init__(self, city_map: CityMap, 
                distance_cost_per_km: float = 2.0,
                time_cost_per_hour: float = 50.0,
                violation_penalty: float = 1000.0,
                unassigned_penalty: float = 500.0):
        """
        Args:
            city_map: 城市地图对象
            distance_cost_per_km: 每公里距离成本
            time_cost_per_hour: 每小时时间成本
            violation_penalty: 违反约束的惩罚成本
            unassigned_penalty: 未分配订单的惩罚成本
        """
        self.city_map = city_map
        self.distance_cost_per_km = distance_cost_per_km
        self.time_cost_per_hour = time_cost_per_hour
        self.violation_penalty = violation_penalty
        self.unassigned_penalty = unassigned_penalty
        self.route_evaluator = RouteEvaluator(city_map)
    
    def calculate_route_cost(self, route: Route, current_time: datetime) -> float:
        """计算路线的成本
        
        Args:
            route: 要计算成本的路线
            current_time: 当前时间
            
        Returns:
            路线总成本
        """
        # 确保路线已评估
        self.route_evaluator.evaluate_route(route, current_time)
        
        # 计算成本
        distance_cost = route.total_distance * self.distance_cost_per_km
        time_cost = route.total_time * self.time_cost_per_hour
        violation_cost = route.violations * self.violation_penalty
        
        total_cost = distance_cost + time_cost + violation_cost
        return total_cost
    
    def calculate_solution_cost(self, solution: Solution, current_time: datetime) -> float:
        """计算解决方案的总成本
        
        Args:
            solution: 要计算成本的解决方案
            current_time: 当前时间
            
        Returns:
            解决方案总成本
        """
        # 确保解决方案已评估
        self.route_evaluator.evaluate_solution(solution, current_time)
        
        # 计算成本
        distance_cost = solution.total_distance * self.distance_cost_per_km
        time_cost = solution.total_time * self.time_cost_per_hour
        violation_cost = solution.total_violations * self.violation_penalty
        unassigned_cost = len(solution.unassigned_orders) * self.unassigned_penalty
        
        total_cost = distance_cost + time_cost + violation_cost + unassigned_cost
        return total_cost


class FeasibilityChecker:
    """可行性检查器，检查路线是否满足各种约束"""
    
    def __init__(self, city_map: CityMap):
        """
        Args:
            city_map: 城市地图对象
        """
        self.city_map = city_map
    
    def is_route_feasible(self, route: Route, current_time: datetime) -> Tuple[bool, List[str]]:
        """检查路线是否可行
        
        Args:
            route: 要检查的路线
            current_time: 当前时间
            
        Returns:
            (是否可行, 违反约束的说明列表)
        """
        if len(route.points) == 0:
            return True, []
        
        violations = []
        
        # 跟踪车辆当前位置、时间和载荷
        current_location = route.vehicle.current_location
        current_vehicle_time = current_time
        current_volume = route.vehicle.current_volume
        current_weight = route.vehicle.current_weight
        
        # 跟踪已取货但未送达的订单
        picked_not_delivered = set()
        
        # 检查每个路线点
        for i, point in enumerate(route.points):
            # 检查是否有路径到下一个点
            try:
                distance = self.city_map.get_distance(current_location.id, point.location.id)
                travel_time = distance / route.vehicle.average_speed
            except ValueError:
                violations.append(f"点{i+1}: 从{current_location.name}到{point.location.name}没有可达路径")
                # 继续检查其他约束，使用直线距离作为估计
                distance = current_location.distance_to(point.location)
                travel_time = distance / route.vehicle.average_speed
            
            # 更新到达时间
            arrival_time = current_vehicle_time + timedelta(hours=travel_time)
            
            # 检查时间窗约束
            if point.is_pickup:
                if arrival_time > point.order.latest_pickup_time:
                    violations.append(f"点{i+1}: 预计到达时间{arrival_time}超过最晚取货时间{point.order.latest_pickup_time}")
                
                # 更新载荷
                current_volume += point.order.volume
                current_weight += point.order.weight
                picked_not_delivered.add(point.order.id)
                
                # 检查车辆容量约束
                if current_volume > route.vehicle.max_volume:
                    violations.append(f"点{i+1}: 取货后超出车辆最大容积 {current_volume}/{route.vehicle.max_volume}")
                if current_weight > route.vehicle.max_weight:
                    violations.append(f"点{i+1}: 取货后超出车辆最大载重 {current_weight}/{route.vehicle.max_weight}")
            else:  # 送货点
                if arrival_time > point.order.latest_delivery_time:
                    violations.append(f"点{i+1}: 预计到达时间{arrival_time}超过最晚送达时间{point.order.latest_delivery_time}")
                
                # 检查送货先后顺序约束
                if point.order.id not in picked_not_delivered:
                    violations.append(f"点{i+1}: 订单{point.order.id}尚未取货就试图送达")
                else:
                    picked_not_delivered.remove(point.order.id)
                
                # 更新载荷
                current_volume -= point.order.volume
                current_weight -= point.order.weight
            
            # 更新当前位置和时间
            current_location = point.location
            if point.is_pickup:
                # 如果早到，需要等待到最早取货时间
                current_vehicle_time = max(arrival_time, point.order.earliest_pickup_time)
            else:
                # 如果早到，需要等待到最早送达时间
                current_vehicle_time = max(arrival_time, point.order.earliest_delivery_time)
        
        # 检查是否所有取货的货物都已送达
        if picked_not_delivered:
            order_ids = ", ".join(str(oid) for oid in picked_not_delivered)
            violations.append(f"路线结束时还有未送达的订单: {order_ids}")
        
        return len(violations) == 0, violations
    
    def is_solution_feasible(self, solution: Solution, current_time: datetime) -> Tuple[bool, Dict[int, List[str]]]:
        """检查解决方案是否可行
        
        Args:
            solution: 要检查的解决方案
            current_time: 当前时间
            
        Returns:
            (是否可行, 按车辆ID分组的违反约束说明)
        """
        all_feasible = True
        vehicle_violations = {}
        
        for vehicle_id, route in solution.routes.items():
            feasible, violations = self.is_route_feasible(route, current_time)
            if not feasible:
                all_feasible = False
                vehicle_violations[vehicle_id] = violations
        
        return all_feasible, vehicle_violations