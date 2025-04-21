from typing import List, Optional, Dict
from datetime import datetime, timedelta

from .location import Location
from .order import Order
from .vehicle import Vehicle
from .city_map import CityMap
from .data_generator import DataGenerator


class DataLayer:
    """数据层主类，管理所有数据对象"""
    
    def __init__(self):
        self.city_map: Optional[CityMap] = None
        self.orders: Dict[int, Order] = {}
        self.vehicles: Dict[int, Vehicle] = {}
        self.current_time: datetime = datetime.now()
        self.data_generator = DataGenerator()
    
    def initialize_city_map(self) -> None:
        """初始化城市地图"""
        self.city_map = self.data_generator.generate_dongying_map()
    
    def initialize_test_data(self, order_count: int = 30, vehicle_count: int = 10) -> None:
        """初始化测试数据
        
        Args:
            order_count: 生成的订单数量
            vehicle_count: 生成的车辆数量
        """
        if self.city_map is None:
            self.initialize_city_map()
        
        # 生成车辆
        vehicles = self.data_generator.generate_random_vehicles(self.city_map, vehicle_count)
        for vehicle in vehicles:
            self.vehicles[vehicle.id] = vehicle
        
        # 生成订单
        orders = self.data_generator.generate_random_orders(self.city_map, order_count)
        for order in orders:
            self.orders[order.id] = order
    
    def get_unassigned_orders(self) -> List[Order]:
        """获取未分配的订单列表"""
        return [order for order in self.orders.values() if not order.is_assigned]
    
    def get_active_orders(self) -> List[Order]:
        """获取已分配但未完成的订单列表"""
        return [order for order in self.orders.values() 
                if order.is_assigned and not order.is_delivered]
    
    def get_completed_orders(self) -> List[Order]:
        """获取已完成的订单列表"""
        return [order for order in self.orders.values() if order.is_delivered]
    
    def get_available_vehicles(self) -> List[Vehicle]:
        """获取可用车辆列表（未满载）"""
        return [vehicle for vehicle in self.vehicles.values() 
                if vehicle.current_volume < vehicle.max_volume or 
                vehicle.current_weight < vehicle.max_weight]
    
    def assign_order_to_vehicle(self, order_id: int, vehicle_id: int) -> bool:
        """将订单分配给车辆
        
        Args:
            order_id: 订单ID
            vehicle_id: 车辆ID
            
        Returns:
            是否分配成功
        """
        if order_id not in self.orders or vehicle_id not in self.vehicles:
            return False
        
        order = self.orders[order_id]
        vehicle = self.vehicles[vehicle_id]
        
        if order.is_assigned:
            return False
        
        return vehicle.add_order(order)
    
    def update_vehicle_location(self, vehicle_id: int, location_id: int) -> bool:
        """更新车辆位置
        
        Args:
            vehicle_id: 车辆ID
            location_id: 新位置ID
            
        Returns:
            是否更新成功
        """
        if vehicle_id not in self.vehicles or location_id not in self.city_map.locations:
            return False
        
        vehicle = self.vehicles[vehicle_id]
        new_location = self.city_map.locations[location_id]
        
        vehicle.update_location(new_location, self.current_time)
        return True
    
    def advance_time(self, minutes: float) -> None:
        """推进模拟时间
        
        Args:
            minutes: 推进的分钟数
        """
        self.current_time += timedelta(minutes=minutes)
    
    def __str__(self) -> str:
        return (f"数据层: {len(self.orders)}个订单, {len(self.vehicles)}辆车, "
                f"当前时间: {self.current_time.strftime('%Y-%m-%d %H:%M:%S')}")