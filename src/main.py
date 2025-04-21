#!/usr/bin/env python3
from datetime import datetime
import random
import argparse

from data_layer.data_layer import DataLayer


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='城市内货物取送运输路线规划系统')
    parser.add_argument('--orders', type=int, default=20, help='生成订单数量')
    parser.add_argument('--vehicles', type=int, default=5, help='生成车辆数量')
    parser.add_argument('--seed', type=int, default=None, help='随机数种子')
    args = parser.parse_args()
    
    # 设置随机数种子
    if args.seed is not None:
        random.seed(args.seed)
    
    print("初始化数据层...")
    data_layer = DataLayer()
    data_layer.initialize_city_map()
    data_layer.initialize_test_data(order_count=args.orders, vehicle_count=args.vehicles)
    
    print(f"成功生成地图: {data_layer.city_map}")
    print(f"成功生成车辆: {args.vehicles}辆")
    print(f"成功生成订单: {args.orders}个")
    
    # 打印部分订单和车辆信息作为示例
    print("\n示例订单:")
    for i, order in enumerate(list(data_layer.orders.values())[:3]):
        print(f"  {order}")
    
    print("\n示例车辆:")
    for i, vehicle in enumerate(list(data_layer.vehicles.values())[:3]):
        print(f"  {vehicle}")
    
    print("\n示例位置:")
    for i, location in enumerate(list(data_layer.city_map.locations.values())[:5]):
        print(f"  {location}")
    
    print("\n数据层初始化完成，可进行下一步算法层实现。")


if __name__ == '__main__':
    main()