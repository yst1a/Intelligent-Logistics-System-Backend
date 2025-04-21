"""
数据层模拟模块
为测试报告生成器提供必要的类和函数
"""

import random
import math
from datetime import datetime, timedelta

class Location:
    """位置类模拟"""
    def __init__(self, location_id=None, name=None, latitude=None, longitude=None):
        self.id = location_id or f"L{random.randint(1000, 9999)}"
        self.name = name or f"位置_{self.id}"
        self.latitude = latitude or random.uniform(30.5, 31.5)  # 模拟上海区域
        self.longitude = longitude or random.uniform(121.0, 122.0)
        
    def distance_to(self, other_location):
        """计算到另一个位置的直线距离（公里）"""
        # 使用简化的距离计算方法
        lat_diff = self.latitude - other_location.latitude
        lon_diff = self.longitude - other_location.longitude
        return math.sqrt(lat_diff**2 + lon_diff**2) * 111  # 粗略的距离换算

class Order:
    """订单类模拟"""
    def __init__(self, order_id=None):
        self.id = order_id or f"O{random.randint(10000, 99999)}"
        self.name = f"订单_{self.id}"
        self.volume = random.uniform(0.1, 5.0)  # 立方米
        self.weight = random.uniform(1, 200)  # 千克
        
        # 创建取货和配送时间窗口
        base_time = datetime.now()
        self.pickup_earliest = base_time + timedelta(hours=random.uniform(0, 8))
        self.pickup_latest = self.pickup_earliest + timedelta(hours=random.uniform(1, 4))
        self.delivery_earliest = self.pickup_latest + timedelta(hours=random.uniform(0.5, 2))
        self.delivery_latest = self.delivery_earliest + timedelta(hours=random.uniform(1, 6))
        
        # 位置
        self.pickup_location = Location()
        self.delivery_location = Location()
        
        # 分配状态
        self.assigned_vehicle_id = None if random.random() > 0.7 else f"V{random.randint(100, 999)}"

class Vehicle:
    """车辆类模拟"""
    def __init__(self, vehicle_id=None):
        self.id = vehicle_id or f"V{random.randint(100, 999)}"
        
        # 随机确定车辆类型
        vehicle_types = ["小型货车", "中型货车", "大型货车"]
        type_weights = [0.5, 0.3, 0.2]  # 各类型概率权重
        self.type = random.choices(vehicle_types, weights=type_weights)[0]
        
        self.name = f"{self.type}_{self.id}"
        
        # 基于车辆类型设置容量
        if "小型" in self.type:
            self.max_volume = random.uniform(1, 5)
            self.max_weight = random.uniform(100, 800)
        elif "中型" in self.type:
            self.max_volume = random.uniform(5, 15)
            self.max_weight = random.uniform(800, 2000)
        else:  # 大型货车
            self.max_volume = random.uniform(15, 30)
            self.max_weight = random.uniform(2000, 5000)
        
        # 当前状态
        self.status = random.choice(['idle', 'loading', 'busy', 'en_route', 'unloading'])
        self.current_location = Location()
        
        # 当前载荷
        self.current_volume = min(random.uniform(0, self.max_volume), self.max_volume)
        self.current_weight = min(random.uniform(0, self.max_weight), self.max_weight)
        
        # 当前订单
        self.current_orders = [f"O{random.randint(10000, 99999)}" for _ in range(random.randint(0, 5))]

class CityMap:
    """城市地图模拟"""
    def __init__(self, node_count=50):
        self.locations = {}
        self.graph = MockGraph()
        
        # 生成随机位置
        for i in range(node_count):
            loc_id = f"L{i+1000}"
            self.locations[loc_id] = Location(loc_id, f"位置{i}")
        
        # 生成随机连接
        for i, loc_id in enumerate(self.locations.keys()):
            # 为每个节点连接5-10个其他节点
            connections = random.sample(list(self.locations.keys()), min(random.randint(5, 10), len(self.locations)))
            for conn in connections:
                if conn != loc_id:  # 避免自环
                    self.graph.add_edge(loc_id, conn)
    
    def get_shortest_path(self, start_id, end_id):
        """模拟获取最短路径"""
        # 简单地返回随机路径
        path_length = random.randint(3, 8)
        path = [start_id]
        
        for _ in range(path_length - 2):
            # 添加一些随机中间节点
            available_nodes = [n for n in self.locations.keys() if n != start_id and n != end_id and n not in path]
            if not available_nodes:
                break
            path.append(random.choice(available_nodes))
        
        path.append(end_id)
        return path
    
    def get_distance(self, start_id, end_id):
        """模拟获取两点间距离"""
        if start_id in self.locations and end_id in self.locations:
            direct = self.locations[start_id].distance_to(self.locations[end_id])
            # 真实距离通常比直线距离长
            return direct * random.uniform(1.1, 1.5)
        return 0

class MockGraph:
    """模拟图结构"""
    def __init__(self):
        self.edges_data = []
        self._edges = {}
    
    def add_edge(self, u, v, weight=None):
        """添加一条边"""
        if u not in self._edges:
            self._edges[u] = []
        self._edges[u].append(v)
        
        # 为有向图存储反向边
        if v not in self._edges:
            self._edges[v] = []
        self._edges[v].append(u)
        
        # 存储边数据
        weight = weight or random.uniform(0.1, 5.0)
        self.edges_data.append((u, v, {'weight': weight}))
    
    def edges(self, data=False):
        """返回所有边"""
        if data:
            return self.edges_data
        else:
            return [(u, v) for u, v, _ in self.edges_data]

class DataLayer:
    """数据层模拟"""
    def __init__(self, order_count=20, vehicle_count=8):
        self.orders = {f"O{i+10000}": Order(f"O{i+10000}") for i in range(order_count)}
        self.vehicles = {f"V{i+100}": Vehicle(f"V{i+100}") for i in range(vehicle_count)}
        self.city_map = CityMap()
        
        # 模拟交通
        self.traffic = MockTraffic()

class MockTraffic:
    """交通模拟"""
    def __init__(self, light_count=20):
        # 创建随机交通信号灯
        self.traffic_lights = []
        for i in range(light_count):
            light = {
                'id': f"TL{i+1}",
                'status': random.choice(['red', 'green', 'yellow']),
                'location': Location()
            }
            self.traffic_lights.append(MockObject(**light))

class MockObject:
    """通用模拟对象类"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)