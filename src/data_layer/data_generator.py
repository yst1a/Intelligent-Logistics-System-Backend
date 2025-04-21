from datetime import datetime, timedelta
import random
import numpy as np
from typing import List, Tuple

from .location import Location
from .order import Order
from .vehicle import Vehicle
from .city_map import CityMap


class DataGenerator:
    """生成测试数据的类"""
    
    def __init__(self, seed: int = None):
        """初始化数据生成器
        
        Args:
            seed: 随机数种子，用于生成可重复的随机数据
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_dongying_map(self) -> CityMap:
        """生成东营市简化地图
        
        Returns:
            包含东营市主要位置和道路的CityMap对象
        """
        city_map = CityMap("东营市")
        
        # 添加主要位置（基于真实位置，但经纬度为近似值）
        locations = [
            Location(1, "东营市政府", 37.4359, 118.6742),
            Location(2, "东营区商业中心", 37.4488, 118.5820),
            Location(3, "垦利区中心", 37.5853, 118.5475),
            Location(4, "河口区中心", 37.8861, 118.5253),
            Location(5, "广饶县中心", 37.0650, 118.4073),
            Location(6, "利津县中心", 37.6560, 118.2556),
            Location(7, "东营港", 38.0800, 118.9600),
            Location(8, "东营站", 37.4496, 118.4932),
            Location(9, "东营胜利机场", 37.2167, 118.2833),
            Location(10, "黄河口生态旅游区", 37.7592, 119.1431),
            Location(11, "胜利油田", 37.7758, 118.6708),
            Location(12, "东营职业学院", 37.4359, 118.6133),
            Location(13, "东营经济开发区", 37.4359, 118.7767),
            Location(14, "东营市第一人民医院", 37.4633, 118.4878),
            Location(15, "东营市物流园区", 37.3912, 118.6211)
        ]
        
        for loc in locations:
            city_map.add_location(loc)
        
        # 添加主要道路（简化版）
        # 格式：(起点ID, 终点ID, 距离(km), 行驶时间(小时))
        roads = [
            (1, 2, 10.5, 0.25), (2, 1, 10.5, 0.25),  # 市政府 <-> 商业中心
            (1, 12, 5.8, 0.12), (12, 1, 5.8, 0.12),  # 市政府 <-> 职业学院
            (1, 13, 9.7, 0.2), (13, 1, 9.7, 0.2),    # 市政府 <-> 经济开发区
            (1, 15, 6.2, 0.15), (15, 1, 6.2, 0.15),  # 市政府 <-> 物流园区
            (2, 8, 8.3, 0.18), (8, 2, 8.3, 0.18),    # 商业中心 <-> 东营站
            (2, 14, 4.2, 0.1), (14, 2, 4.2, 0.1),    # 商业中心 <-> 第一人民医院
            (3, 1, 17.6, 0.35), (1, 3, 17.6, 0.35),  # 垦利区中心 <-> 市政府
            (3, 11, 21.2, 0.4), (11, 3, 21.2, 0.4),  # 垦利区中心 <-> 胜利油田
            (4, 11, 25.3, 0.45), (11, 4, 25.3, 0.45),# 河口区中心 <-> 胜利油田
            (4, 7, 15.8, 0.3), (7, 4, 15.8, 0.3),    # 河口区中心 <-> 东营港
            (4, 10, 38.6, 0.7), (10, 4, 38.6, 0.7),  # 河口区中心 <-> 黄河口生态旅游区
            (5, 9, 18.5, 0.35), (9, 5, 18.5, 0.35),  # 广饶县中心 <-> 东营胜利机场
            (5, 2, 42.3, 0.75), (2, 5, 42.3, 0.75),  # 广饶县中心 <-> 商业中心
            (6, 2, 31.7, 0.6), (2, 6, 31.7, 0.6),    # 利津县中心 <-> 商业中心
            (8, 15, 11.4, 0.22), (15, 8, 11.4, 0.22),# 东营站 <-> 物流园区
            (9, 1, 36.5, 0.65), (1, 9, 36.5, 0.65),  # 东营胜利机场 <-> 市政府
            (11, 10, 45.3, 0.8), (10, 11, 45.3, 0.8),# 胜利油田 <-> 黄河口生态旅游区
            (12, 14, 12.4, 0.25), (14, 12, 12.4, 0.25),# 职业学院 <-> 第一人民医院
            (13, 15, 15.5, 0.3), (15, 13, 15.5, 0.3) # 经济开发区 <-> 物流园区
        ]
        
        for road in roads:
            city_map.add_road(*road)
        
        return city_map
    
    def generate_random_orders(self, city_map: CityMap, count: int, 
                              start_time: datetime = None,
                              time_span_hours: float = 24.0) -> List[Order]:
        """生成随机订单"""
        if start_time is None:
            start_time = datetime.now()
        
        orders = []
        location_ids = list(city_map.locations.keys())
        
        # 预先计算可达性矩阵
        reachable_pairs = {}
        for pickup_id in location_ids:
            reachable_pairs[pickup_id] = []
            for delivery_id in location_ids:
                if pickup_id != delivery_id:
                    try:
                        path = city_map.get_shortest_path(pickup_id, delivery_id)
                        if path:  # 如果有路径
                            reachable_pairs[pickup_id].append(delivery_id)
                    except:
                        pass  # 忽略无路径的情况

        attempts = 0
        max_attempts = count * 10  # 设置最大尝试次数，避免死循环

        while len(orders) < count and attempts < max_attempts:
            attempts += 1

            # 选择有可达目的地的起点
            valid_pickup_ids = [pid for pid in location_ids if reachable_pairs.get(pid)]
            if not valid_pickup_ids:
                continue

            pickup_id = random.choice(valid_pickup_ids)
            # 从可达目的地列表中选择
            if not reachable_pairs[pickup_id]:
                continue
            delivery_id = random.choice(reachable_pairs[pickup_id])

            pickup_location = city_map.locations[pickup_id]
            delivery_location = city_map.locations[delivery_id]

            # 随机生成货物属性
            volume = random.uniform(0.1, 3.0)  # 0.1-3立方米
            weight = random.uniform(1.0, 200.0)  # 1-200千克

            # 随机生成订单创建时间
            creation_delta = timedelta(hours=random.uniform(0, time_span_hours))
            creation_time = start_time + creation_delta

            # 计算最早和最晚取货时间
            earliest_pickup_time = creation_time + timedelta(minutes=random.uniform(15, 60))
            latest_pickup_time = earliest_pickup_time + timedelta(hours=random.uniform(1, 3))

            # 估计从取货点到送货点的行驶时间
            try:
                travel_time = city_map.get_travel_time(pickup_id, delivery_id)
            except ValueError:
                # 如果没有路径，使用默认值
                travel_time = pickup_location.distance_to(delivery_location) / 40.0  # 假设40km/h

            # 计算最早和最晚送达时间
            earliest_delivery_time = earliest_pickup_time + timedelta(hours=travel_time)
            latest_delivery_time = latest_pickup_time + timedelta(hours=travel_time + random.uniform(1, 2))

            # 创建订单
            order = Order(
                id=len(orders) + 1,
                pickup_location=pickup_location,
                delivery_location=delivery_location,
                volume=volume,
                weight=weight,
                earliest_pickup_time=earliest_pickup_time,
                latest_pickup_time=latest_pickup_time,
                earliest_delivery_time=earliest_delivery_time,
                latest_delivery_time=latest_delivery_time,
                creation_time=creation_time
            )

            orders.append(order)

        # 按创建时间排序
        orders.sort(key=lambda o: o.creation_time)
        return orders

    def generate_random_vehicles(self, city_map: CityMap, count: int) -> List[Vehicle]:
        """生成随机车辆

        Args:
            city_map: 城市地图对象
            count: 要生成的车辆数量

        Returns:
            Vehicle对象列表
        """
        vehicles = []
        location_ids = list(city_map.locations.keys())

        # 车辆类型定义：(名称前缀, 最大体积, 最大载重, 平均速度)
        vehicle_types = [
            ("小型货车", 5.0, 500.0, 40.0),
            ("中型货车", 15.0, 2000.0, 35.0),
            ("大型货车", 30.0, 5000.0, 30.0)
        ]

        for i in range(count):
            # 随机选择位置和车辆类型
            location_id = random.choice(location_ids)
            vehicle_type = random.choice(vehicle_types)

            # 生成车辆
            vehicle = Vehicle(
                id=i + 1,
                name=f"{vehicle_type[0]}-{i+1}",
                current_location=city_map.locations[location_id],
                max_volume=vehicle_type[1],
                max_weight=vehicle_type[2],
                average_speed=vehicle_type[3]
            )

            vehicles.append(vehicle)

        return vehicles