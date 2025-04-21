import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Location:
    """表示地理位置的类"""
    id: int
    name: str
    latitude: float
    longitude: float
    
    def distance_to(self, other: 'Location') -> float:
        """计算到另一个位置的欧几里得距离（公里）"""
        # 使用哈弗辛公式计算地球表面两点间的距离
        R = 6371  # 地球半径，单位km
        
        lat1_rad = math.radians(self.latitude)
        lon1_rad = math.radians(self.longitude)
        lat2_rad = math.radians(other.latitude)
        lon2_rad = math.radians(other.longitude)
        
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance
    
    def __str__(self) -> str:
        return f"{self.name}({self.id}): ({self.latitude:.6f}, {self.longitude:.6f})"