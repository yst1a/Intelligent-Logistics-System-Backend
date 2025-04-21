"""
算法层基础结构和接口定义
此模块定义了算法层的核心接口和数据结构
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Set
from datetime import datetime, timedelta
import time

from data_layer.order import Order
from data_layer.vehicle import Vehicle
from data_layer.location import Location
from data_layer.city_map import CityMap


class RoutePoint:
    """路线上的一个点，包含位置和动作（取货/送货）"""
    
    ACTION_PICKUP = "pickup"
    ACTION_DELIVERY = "delivery"
    
    def __init__(self, location: Location, action: str, order: Order, 
                 arrival_time: Optional[datetime] = None):
        """
        Args:
            location: 位置对象
            action: 动作类型，'pickup' 或 'delivery'
            order: 相关订单
            arrival_time: 预计到达时间
        """
        self.location = location
        self.action = action
        self.order = order
        self.arrival_time = arrival_time
    
    @property
    def is_pickup(self) -> bool:
        """是否为取货点"""
        return self.action == self.ACTION_PICKUP
    
    @property
    def is_delivery(self) -> bool:
        """是否为送货点"""
        return self.action == self.ACTION_DELIVERY
    
    def __str__(self) -> str:
        action_str = "取货" if self.is_pickup else "送货"
        time_str = "" if self.arrival_time is None else f", 预计{self.arrival_time.strftime('%H:%M:%S')}"
        return f"{action_str} {self.order.id} @{self.location.name}{time_str}"


class Route:
    """表示一条完整的路线，包含多个路线点"""
    
    def __init__(self, vehicle: Vehicle, points: List[RoutePoint] = None):
        """
        Args:
            vehicle: 执行该路线的车辆
            points: 路线点列表，按访问顺序排列
        """
        self.vehicle = vehicle
        self.points = points or []
        
        # 路线评价指标
        self.total_distance = 0.0
        self.total_time = 0.0
        self.violations = 0  # 违反约束的次数
    
    def add_point(self, point: RoutePoint) -> None:
        """添加路线点"""
        self.points.append(point)
    
    def insert_point(self, index: int, point: RoutePoint) -> None:
        """在指定位置插入路线点"""
        self.points.insert(index, point)
    
    def clear(self) -> None:
        """清空路线"""
        self.points.clear()
        self.total_distance = 0.0
        self.total_time = 0.0
        self.violations = 0
    
    def get_order_ids(self) -> Set[int]:
        """获取路线中包含的所有订单ID"""
        return {point.order.id for point in self.points}
    
    def copy(self) -> 'Route':
        """创建路线的深拷贝"""
        new_route = Route(self.vehicle)
        new_route.points = self.points.copy()
        new_route.total_distance = self.total_distance
        new_route.total_time = self.total_time
        new_route.violations = self.violations
        return new_route
    
    def __len__(self) -> int:
        """路线点数量"""
        return len(self.points)
    
    def __str__(self) -> str:
        result = f"车辆 {self.vehicle.id} 路线: {len(self.points)}个点, 总距离: {self.total_distance:.2f}km, 时间: {self.total_time:.2f}h"
        if self.violations > 0:
            result += f", 违反约束: {self.violations}次"
        
        if len(self.points) > 0:
            result += "\n  路线详情: "
            result += " -> ".join(str(p) for p in self.points)
        
        return result


class Solution:
    """表示问题的完整解决方案，包含多条路线"""
    
    def __init__(self):
        """初始化空解决方案"""
        self.routes: Dict[int, Route] = {}  # 车辆ID -> 路线
        self.unassigned_orders: Set[int] = set()  # 未分配的订单ID
        
        # 解决方案评价指标
        self.total_distance = 0.0
        self.total_time = 0.0
        self.total_violations = 0
    
    def add_route(self, route: Route) -> None:
        """添加路线到解决方案"""
        self.routes[route.vehicle.id] = route
    
    def add_unassigned_order(self, order_id: int) -> None:
        """添加未分配的订单"""
        self.unassigned_orders.add(order_id)
    
    def remove_unassigned_order(self, order_id: int) -> None:
        """移除未分配的订单"""
        if order_id in self.unassigned_orders:
            self.unassigned_orders.remove(order_id)
    
    def update_metrics(self) -> None:
        """更新解决方案的评价指标"""
        self.total_distance = sum(route.total_distance for route in self.routes.values())
        self.total_time = sum(route.total_time for route in self.routes.values())
        self.total_violations = sum(route.violations for route in self.routes.values())
    
    def copy(self) -> 'Solution':
        """创建解决方案的深拷贝"""
        new_solution = Solution()
        for vehicle_id, route in self.routes.items():
            new_solution.routes[vehicle_id] = route.copy()
        new_solution.unassigned_orders = self.unassigned_orders.copy()
        new_solution.total_distance = self.total_distance
        new_solution.total_time = self.total_time
        new_solution.total_violations = self.total_violations
        return new_solution
    
    def __str__(self) -> str:
        result = (f"解决方案: {len(self.routes)}条路线, 总距离: {self.total_distance:.2f}km, "
                 f"总时间: {self.total_time:.2f}h, 未分配订单: {len(self.unassigned_orders)}")
        if self.total_violations > 0:
            result += f", 违反约束: {self.total_violations}次"
        
        for route in self.routes.values():
            result += f"\n  {route}"
        
        return result


class Algorithm(ABC):
    """算法基类，定义算法接口"""
    
    def __init__(self, city_map: CityMap):
        """
        Args:
            city_map: 城市地图对象
        """
        self.city_map = city_map
    
    @abstractmethod
    def solve(self, orders: List[Order], vehicles: List[Vehicle]) -> Solution:
        """求解问题
        
        Args:
            orders: 订单列表
            vehicles: 车辆列表
            
        Returns:
            求解得到的方案
        """
        pass
    
    def solve_with_timeout(self, orders: List[Order], vehicles: List[Vehicle], 
                        timeout_seconds: float) -> Solution:
        """带超时的求解
        
        Args:
            orders: 订单列表
            vehicles: 车辆列表
            timeout_seconds: 超时时间（秒）
            
        Returns:
            超时前找到的最佳方案
        """
        start_time = time.time()
        solution = self.solve(orders, vehicles)
        elapsed = time.time() - start_time
        
        print(f"算法执行时间: {elapsed:.2f} 秒 (超时设置: {timeout_seconds} 秒)")
        return solution