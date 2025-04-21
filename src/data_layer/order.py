from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .location import Location


@dataclass
class Order:
    """表示货物订单的类"""
    id: int
    pickup_location: Location
    delivery_location: Location
    volume: float  # 体积，单位立方米
    weight: float  # 重量，单位千克
    earliest_pickup_time: datetime  # 最早取货时间
    latest_pickup_time: datetime    # 最晚取货时间
    earliest_delivery_time: datetime  # 最早送达时间
    latest_delivery_time: datetime    # 最晚送达时间
    creation_time: datetime  # 订单创建时间
    
    # 订单状态跟踪
    assigned_vehicle_id: Optional[int] = None
    pickup_time: Optional[datetime] = None
    delivery_time: Optional[datetime] = None
    
    @property
    def is_assigned(self) -> bool:
        """检查订单是否已分配"""
        return self.assigned_vehicle_id is not None
    
    @property
    def is_picked_up(self) -> bool:
        """检查订单是否已取货"""
        return self.pickup_time is not None
    
    @property
    def is_delivered(self) -> bool:
        """检查订单是否已送达"""
        return self.delivery_time is not None
    
    @property
    def status(self) -> str:
        """获取订单状态"""
        if self.is_delivered:
            return "已送达"
        elif self.is_picked_up:
            return "运送中"
        elif self.is_assigned:
            return "已分配"
        else:
            return "待分配"
    
    def __str__(self) -> str:
        return (f"订单{self.id}: 从{self.pickup_location.name}到{self.delivery_location.name}, "
                f"体积{self.volume}m³, 重量{self.weight}kg, 状态: {self.status}")