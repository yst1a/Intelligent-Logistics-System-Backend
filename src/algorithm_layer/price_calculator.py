"""
运费计算模块
负责计算订单的运费，考虑距离、时间、紧急程度等因素
"""
from typing import Optional
from datetime import datetime, timedelta

from data_layer.order import Order
from data_layer.city_map import CityMap


class PriceCalculator:
    """运费计算器，计算订单的运费"""
    
    def __init__(self, city_map: CityMap):
        """
        Args:
            city_map: 城市地图对象
        """
        self.city_map = city_map
        
        # 基础价格参数
        self.base_fee = 10.0  # 起步价
        self.distance_price_per_km = 2.0  # 每公里费用
        self.volume_price_per_m3 = 5.0  # 每立方米费用
        self.weight_price_per_kg = 0.1  # 每公斤费用
        
        # 时间相关参数
        self.peak_hours = [(7, 9), (17, 19)]  # 高峰时段，格式为(开始小时, 结束小时)
        self.peak_hour_factor = 1.5  # 高峰时段价格因子
        self.night_hours = [(22, 24), (0, 6)]  # 夜间时段
        self.night_factor = 1.3  # 夜间价格因子
        
        # 紧急订单参数
        self.urgent_time_threshold = 2.0  # 紧急订单时间阈值（小时）
        self.urgent_factor = 1.8  # 紧急订单价格因子
        
        # 拼单折扣
        self.shared_discount = 0.8  # 拼单折扣系数
    
    def calculate_price(self, order: Order, is_shared: bool = False) -> float:
        """计算订单价格
        
        Args:
            order: 订单对象
            is_shared: 是否是拼单
            
        Returns:
            订单价格
        """
        # 计算路径距离
        try:
            distance = self.city_map.get_distance(order.pickup_location.id, order.delivery_location.id)
        except ValueError:
            # 如果没有路径，使用直线距离
            distance = order.pickup_location.distance_to(order.delivery_location)
        
        # 基础价格计算
        distance_fee = distance * self.distance_price_per_km
        volume_fee = order.volume * self.volume_price_per_m3
        weight_fee = order.weight * self.weight_price_per_kg
        
        # 计算基础价格
        base_price = self.base_fee + distance_fee + volume_fee + weight_fee
        
        # 应用时间因子
        time_factor = self._calculate_time_factor(order.earliest_pickup_time)
        
        # 计算紧急程度因子
        urgency_factor = self._calculate_urgency_factor(order)
        
        # 计算最终价格
        final_price = base_price * time_factor * urgency_factor
        
        # 应用拼单折扣
        if is_shared:
            final_price *= self.shared_discount
        
        return round(final_price, 2)  # 四舍五入到分
    
    def _calculate_time_factor(self, time: datetime) -> float:
        """计算时间因子
        
        Args:
            time: 订单时间
            
        Returns:
            时间因子
        """
        hour = time.hour
        
        # 检查是否在高峰时段
        for start, end in self.peak_hours:
            if start <= hour < end:
                return self.peak_hour_factor
        
        # 检查是否在夜间时段
        for start, end in self.night_hours:
            if start <= hour < end:
                return self.night_factor
        
        # 正常时段
        return 1.0
    
    def _calculate_urgency_factor(self, order: Order) -> float:
        """计算紧急程度因子
        
        Args:
            order: 订单对象
            
        Returns:
            紧急程度因子
        """
        # 计算时间窗口大小（小时）
        time_window = (order.latest_delivery_time - order.earliest_pickup_time).total_seconds() / 3600.0
        
        # 如果时间窗口小于阈值，视为紧急订单
        if time_window < self.urgent_time_threshold:
            return self.urgent_factor
        
        # 正常订单
        return 1.0
    
    def estimate_delivery_time(self, order: Order) -> datetime:
        """估算送达时间
        
        Args:
            order: 订单对象
            
        Returns:
            估计送达时间
        """
        # 计算从取货到送货的行驶时间
        try:
            travel_time = self.city_map.get_travel_time(
                order.pickup_location.id, order.delivery_location.id)
        except ValueError:
            # 如果没有路径，使用直线距离估计
            distance = order.pickup_location.distance_to(order.delivery_location)
            travel_time = distance / 40.0  # 假设40km/h的平均速度
        
        # 估计送达时间 = 最早取货时间 + 行驶时间
        return order.earliest_pickup_time + timedelta(hours=travel_time)
    
    def calculate_shared_price(self, orders: list[Order]) -> dict[int, float]:
        """计算拼单价格
        
        Args:
            orders: 拼单的订单列表
            
        Returns:
            订单ID到价格的映射
        """
        result = {}
        
        # 分别计算每个订单的原始价格
        original_prices = {}
        for order in orders:
            original_prices[order.id] = self.calculate_price(order, is_shared=False)
        
        # 计算总距离（简化的示例，实际中可能需要更复杂的逻辑）
        total_distance = 0
        for order in orders:
            try:
                distance = self.city_map.get_distance(
                    order.pickup_location.id, order.delivery_location.id)
            except ValueError:
                distance = order.pickup_location.distance_to(order.delivery_location)
            total_distance += distance
        
        # 应用拼单折扣
        total_original_price = sum(original_prices.values())
        total_discounted_price = total_original_price * self.shared_discount
        
        # 按原始价格比例分配折扣后的总价
        for order_id, original_price in original_prices.items():
            ratio = original_price / total_original_price
            result[order_id] = round(total_discounted_price * ratio, 2)
        
        return result