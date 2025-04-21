import numpy as np
from typing import Dict, List, Tuple, Optional
import networkx as nx

from .location import Location


class CityMap:
    """表示城市地图的类，包含位置和路径信息"""
    
    def __init__(self, name: str = "东营市"):
        self.name = name
        self.locations: Dict[int, Location] = {}
        self.distance_matrix: Optional[np.ndarray] = None
        self.graph = nx.DiGraph()  # 使用有向图表示道路网络
    
    def add_location(self, location: Location) -> None:
        """添加位置到地图"""
        self.locations[location.id] = location
        self.graph.add_node(location.id, 
                           name=location.name, 
                           latitude=location.latitude, 
                           longitude=location.longitude)
        
        # 添加新位置后，重置距离矩阵
        self.distance_matrix = None
    
    def add_road(self, from_id: int, to_id: int, distance: float, travel_time: float) -> None:
        """添加从一个位置到另一个位置的道路"""
        if from_id not in self.locations or to_id not in self.locations:
            raise ValueError("位置ID不存在")
        
        self.graph.add_edge(from_id, to_id, 
                           distance=distance, 
                           travel_time=travel_time)
        
        # 添加新道路后，重置距离矩阵
        self.distance_matrix = None
    
    def get_distance(self, from_id: int, to_id: int) -> float:
        """获取两个位置之间的最短距离"""
        if self.distance_matrix is None:
            self._compute_distance_matrix()
        
        if from_id not in self.locations or to_id not in self.locations:
            raise ValueError("位置ID不存在")
        
        # 获取位置ID在距离矩阵中的索引
        from_idx = list(self.locations.keys()).index(from_id)
        to_idx = list(self.locations.keys()).index(to_id)
        
        return self.distance_matrix[from_idx, to_idx]
    
    def get_shortest_path(self, from_id: int, to_id: int) -> List[int]:
        """获取两个位置之间的最短路径（位置ID列表）"""
        if from_id not in self.locations or to_id not in self.locations:
            raise ValueError("位置ID不存在")
        
        try:
            path = nx.shortest_path(self.graph, from_id, to_id, weight='distance')
            return path
        except nx.NetworkXNoPath:
            raise ValueError(f"从位置 {from_id} 到位置 {to_id} 没有可达路径")
    
    def get_travel_time(self, from_id: int, to_id: int) -> float:
        """获取两个位置之间的估计行驶时间（小时）"""
        if from_id not in self.locations or to_id not in self.locations:
            raise ValueError("位置ID不存在")
        
        try:
            path = nx.shortest_path(self.graph, from_id, to_id, weight='travel_time')
            return sum(self.graph[u][v]['travel_time'] for u, v in zip(path[:-1], path[1:]))
        except nx.NetworkXNoPath:
            raise ValueError(f"从位置 {from_id} 到位置 {to_id} 没有可达路径")
    
    def _compute_distance_matrix(self) -> None:
        """计算所有位置对之间的最短距离矩阵"""
        location_ids = list(self.locations.keys())
        n = len(location_ids)
        self.distance_matrix = np.zeros((n, n))
        
        # 使用Floyd-Warshall算法计算最短路径
        dist_dict = dict(nx.floyd_warshall(self.graph, weight='distance'))
        
        for i, from_id in enumerate(location_ids):
            for j, to_id in enumerate(location_ids):
                if from_id in dist_dict and to_id in dist_dict.get(from_id, {}):
                    self.distance_matrix[i, j] = dist_dict[from_id][to_id]
                else:
                    # 如果没有路径，设置一个大值
                    self.distance_matrix[i, j] = float('inf')
    
    def __str__(self) -> str:
        return f"{self.name}地图: {len(self.locations)}个位置, {self.graph.number_of_edges()}条道路"