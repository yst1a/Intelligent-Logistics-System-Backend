from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

from .location import Location
from .order import Order


@dataclass
class Vehicle:
    """表示物流车辆的类"""
    id: int
    name: str
    current_location: Location
    max_volume: float  # 最大容积，单位立方米
    max_weight: float  # 最大载重，单位千克
    average_speed: float  # 平均速度，单位km/h
    
    # 车辆状态
    current_volume: float = 0.0
    current_weight: float = 0.0
    current_orders: List[Order] = field(default_factory=list)
    route_plan: List[Tuple[Location, str, Optional[Order]]] = field(default_factory=list)  # (位置, 操作类型, 订单)
    last_location_update_time: datetime = field(default_factory=datetime.now)
    
    def can_accommodate(self, order: Order) -> bool:
        """检查是否能容纳新订单"""
        return (self.current_volume + order.volume <= self.max_volume and 
                self.current_weight + order.weight <= self.max_weight)
    
    def add_order(self, order: Order) -> bool:
        """添加订单到车辆"""
        if not self.can_accommodate(order):
            return False
        
        # 检查车辆是否能在时间窗口内服务该订单
        # 这里可以添加时间窗口检查的逻辑

        self.current_orders.append(order)
        self.current_volume += order.volume
        self.current_weight += order.weight
        order.assigned_vehicle_id = self.id

        # 将取货和送货点加到路线末尾
        self.route_plan.append((order.pickup_location, "pickup", order))
        self.route_plan.append((order.delivery_location, "delivery", order))

        return True
    
    def complete_pickup(self, order: Order, current_time: datetime) -> bool:
        """完成订单取货"""
        if order not in self.current_orders:
            return False
        
        order.pickup_time = current_time
        return True
    
    def complete_delivery(self, order: Order, current_time: datetime) -> bool:
        """完成订单送达"""
        if order not in self.current_orders:
            return False
        order.delivery_time = current_time
        self.current_volume -= order.volume
        self.current_weight -= order.weight
        self.current_orders.remove(order)
        return True

    def update_location(self, new_location: Location, current_time: datetime) -> None:
        """更新车辆位置"""
        self.current_location = new_location
        self.last_location_update_time = current_time

    def estimated_travel_time(self, from_loc: Location, to_loc: Location) -> float:
        """估计从一个位置到另一个位置的行驶时间（小时）"""
        distance = from_loc.distance_to(to_loc)
        return distance / self.average_speed

    def __str__(self) -> str:
        return (f"车辆{self.id}({self.name}): 位置: {self.current_location.name}, "
                f"载重: {self.current_weight}/{self.max_weight}kg, "
                f"容积: {self.current_volume}/{self.max_volume}m³, "
                f"当前订单数: {len(self.current_orders)}")