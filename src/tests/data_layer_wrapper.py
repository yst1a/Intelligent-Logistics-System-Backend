"""
数据层模块导入包装器
处理Python导入路径问题，提供统一的模块接口
"""

import os
import sys
import importlib.util

# 添加src目录到Python路径，这样我们可以访问src/data_layer
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# 导入所需的模块
try:
    from data_layer.location import Location
    from data_layer.order import Order
    from data_layer.vehicle import Vehicle
    from data_layer.city_map import CityMap
    from data_layer.data_layer import DataLayer
    from data_layer.data_generator import DataGenerator
    
    # 表明我们使用真实数据层
    USING_MOCK = False
    print("成功导入数据层模块")
    
except ImportError as e:
    print(f"警告: 无法导入数据层模块 ({e})，使用模拟数据")
    
    # 使用模拟数据层
    from mock_data_layer import (
        Location, Order, Vehicle, CityMap, DataLayer
    )
    
    USING_MOCK = True

def get_data_layer_instance(order_count=20, vehicle_count=8):
    """获取数据层实例，处理真实或模拟数据层的差异"""
    return DataLayer(order_count=order_count, vehicle_count=vehicle_count)