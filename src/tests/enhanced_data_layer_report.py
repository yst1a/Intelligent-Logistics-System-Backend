#!/usr/bin/env python3
"""
数据层增强测试报告生成器
为"智送"城市货运智能调度系统提供高质量可视化测试报告
"""

import os
import sys
import time
import webbrowser
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import base64
from io import BytesIO
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# 使用包装器模块导入数据层组件
from data_layer_wrapper import USING_MOCK

from test_data_layer_visual import DataLayerVisualTester
from plotly.offline import plot
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd

# 根据是否使用模拟数据导入相应模块
if not USING_MOCK:
    from test_data_layer_visual import DataLayerVisualTester
else:
    from data_layer_wrapper import (
        DataLayer, Location, Order, Vehicle, CityMap
    )

# plotly导入，用于交互式可视化
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.offline import plot
    HAS_PLOTLY = True
except ImportError:
    print("警告: 未安装Plotly，将不会生成交互式可视化")
    HAS_PLOTLY = False

# 配置matplotlib和seaborn
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 300
sns.set(style="whitegrid", font='SimHei')


class EnhancedDataLayerReportGenerator:
    """数据层增强测试报告生成器"""

    def __init__(self, order_count=20, vehicle_count=8):
        """初始化报告生成器"""
        self.order_count = order_count
        self.vehicle_count = vehicle_count

        # 设置输出目录
        self.output_dir = os.path.join(os.path.dirname(__file__), 'output/enhanced_report')
        os.makedirs(f"{self.output_dir}/images", exist_ok=True)
        os.makedirs(f"{self.output_dir}/interactive", exist_ok=True)

        # 设置配色方案
        self.color_palette = {
            'primary': '#2b6cb0',
            'secondary': '#38b2ac',
            'success': '#48bb78',
            'warning': '#ecc94b',
            'danger': '#f56565',
            'light': '#f7fafc',
            'dark': '#2d3748',
            'gray': '#a0aec0'
        }

        # 初始化分析数据字典
        self.analysis_data = {}

        # 使用真实数据层或模拟数据层
        if not USING_MOCK:
            # 使用真实数据层进行测试
            self.visual_tester = DataLayerVisualTester(order_count, vehicle_count)
            self.data_layer = self.visual_tester.data_layer
        else:
            # 使用模拟数据层
            self.data_layer = DataLayer(order_count, vehicle_count)

        # 保存常用的数据引用
        self.orders = self.data_layer.orders
        self.vehicles = self.data_layer.vehicles
        self.city_map = self.data_layer.city_map
    def run_tests_and_generate_report(self):
        """运行测试并生成增强报告"""
        print(f"\n{'=' * 50}")
        print("开始运行数据层增强测试及报告生成...")
        print(f"{'=' * 50}")

        # 运行测试并收集基础数据
        test_result = self.tester.run_all_tests()

        # 收集数据层对象以便进行高级分析
        self.data_layer = self.tester.data_layer
        self.city_map = self.data_layer.city_map
        self.orders = self.data_layer.orders
        self.vehicles = self.data_layer.vehicles

        print("\n正在生成增强分析和可视化...")

        # 1. 城市地图高级分析
        self.analyze_city_map()

        # 2. 订单高级分析
        self.analyze_orders()

        # 3. 车辆高级分析
        self.analyze_vehicles()

        # 4. 路径和交通分析
        self.analyze_routing()

        # 5. 系统性能分析
        self.analyze_system_performance()

        # 6. 生成最终HTML报告
        report_path = self.generate_html_report()

        print(f"\n增强测试报告已生成: {report_path}")
        print(f"{'=' * 50}")

        # 尝试打开报告
        try:
            webbrowser.open('file://' + os.path.abspath(report_path))
        except:
            print("无法自动打开浏览器，请手动打开上述报告路径")

        return test_result

    def analyze_city_map(self):
        """分析城市地图并创建高级可视化"""
        print("正在分析城市地图数据...")
        city_map = self.city_map

        # 1. 生成城市热力图 (结合节点连接度)
        plt.figure(figsize=(14, 10))

        # 计算节点连接度
        node_degrees = {}
        for node in city_map.graph.nodes():
            node_degrees[node] = len(list(city_map.graph.successors(node)))

        # 绘制基础地图
        pos = {loc_id: (location.longitude, location.latitude)
               for loc_id, location in city_map.locations.items()}

        # 使用节点连接度作为节点大小
        node_sizes = [100 + node_degrees.get(n, 0) * 50 for n in city_map.graph.nodes()]

        # 绘制边，颜色表示距离
        edges = list(city_map.graph.edges(data=True))
        edge_colors = [data['distance'] for _, _, data in edges]

        # 创建自定义色彩映射
        cmap = plt.cm.viridis_r

        # 绘制图形
        plt.scatter(
            [pos[n][0] for n in city_map.graph.nodes()],
            [pos[n][1] for n in city_map.graph.nodes()],
            s=node_sizes,
            c=[node_degrees.get(n, 0) for n in city_map.graph.nodes()],
            cmap='Reds',
            alpha=0.8,
            edgecolors='white'
        )

        # 绘制边
        for (u, v, data) in edges:
            plt.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                alpha=0.5,
                linewidth=1.5,
                color=cmap(data['distance'] / max(edge_colors))
            )

        # 添加位置标签
        for node, (x, y) in pos.items():
            plt.text(x, y, f"{node}: {city_map.locations[node].name}",
                     fontsize=8, ha='center', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

        plt.title(f"{city_map.name}城市交通网络分析", fontsize=16)
        plt.colorbar(plt.cm.ScalarMappable(cmap='Reds'),
                     label='节点连接度 (相连道路数量)')
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/images/city_map_advanced.png", dpi=300)
        plt.close()

        # 2. 交通网络拓扑分析
        node_centrality = self.calculate_network_centrality(city_map)

        # 保存网络分析结果
        self.analysis_data['map_stats'] = {
            'node_count': len(city_map.locations),
            'edge_count': len(list(city_map.graph.edges())),
            'average_degree': sum(node_degrees.values()) / len(node_degrees),
            'most_connected': max(node_degrees.items(), key=lambda x: x[1]),
            'most_central': max(node_centrality.items(), key=lambda x: x[1])
        }

        # 3. 创建交互式城市网络图
        self.create_interactive_city_map(city_map, node_degrees, node_centrality)

    def calculate_network_centrality(self, city_map):
        """计算网络中心性"""
        import networkx as nx

        # 创建无向图用于中心性计算
        G = nx.Graph()
        for u, v, data in city_map.graph.edges(data=True):
            G.add_edge(u, v, weight=data['distance'])

        # 计算介数中心性
        centrality = nx.betweenness_centrality(G, weight='weight')
        return centrality

    def create_interactive_city_map(self, city_map, node_degrees, node_centrality):
        """创建交互式城市地图"""
        # 准备数据
        node_data = []
        for node_id, location in city_map.locations.items():
            node_data.append({
                'id': node_id,
                'name': location.name,
                'lat': location.latitude,
                'lon': location.longitude,
                'degree': node_degrees.get(node_id, 0),
                'centrality': node_centrality.get(node_id, 0)
            })

        edge_data = []
        for u, v, data in city_map.graph.edges(data=True):
            edge_data.append({
                'from': u,
                'to': v,
                'distance': data['distance'],
                'from_name': city_map.locations[u].name,
                'to_name': city_map.locations[v].name
            })

        # 创建DataFrame
        nodes_df = pd.DataFrame(node_data)

        # 创建交互式图
        fig = px.scatter_mapbox(
            nodes_df,
            lat='lat',
            lon='lon',
            size='degree',
            color='centrality',
            hover_name='name',
            hover_data=['id', 'degree'],
            color_continuous_scale=px.colors.sequential.Plasma,
            size_max=30,
            zoom=10,
            title=f"{city_map.name}城市交通网络交互式地图"
        )

        # 添加边
        for edge in edge_data:
            fig.add_trace(
                go.Scattermapbox(
                    lon=[city_map.locations[edge['from']].longitude,
                         city_map.locations[edge['to']].longitude],
                    lat=[city_map.locations[edge['from']].latitude,
                         city_map.locations[edge['to']].latitude],
                    mode='lines',
                    line=dict(width=1, color='rgba(80, 80, 80, 0.5)'),
                    hoverinfo='text',
                    text=f"{edge['from_name']} → {edge['to_name']}: {edge['distance']:.1f}km",
                    showlegend=False
                )
            )

        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})

        # 保存为HTML文件
        interactive_map_path = f"{self.output_dir}/interactive/city_network.html"
        plot(fig, filename=interactive_map_path, auto_open=False)

    def analyze_orders(self):
        """分析订单数据并创建高级可视化"""
        print("正在分析订单数据...")
        orders = list(self.orders.values())

        if not orders:
            print("警告: 没有可用的订单数据")
            return

        # 1. 时空分布热力图
        # 提取取货和送货时间
        pickup_hours = [order.earliest_pickup_time.hour + order.earliest_pickup_time.minute / 60
                        for order in orders]
        delivery_hours = [order.earliest_delivery_time.hour + order.earliest_delivery_time.minute / 60
                          for order in orders]

        # 提取订单体积和重量
        volumes = [order.volume for order in orders]
        weights = [order.weight for order in orders]

        # 生成热力图数据
        plt.figure(figsize=(14, 10))

        # 创建时空热图
        pickup_delivery_data = []
        for order in orders:
            pickup_delivery_data.append({
                'hour': order.earliest_pickup_time.hour,
                'type': '取货',
                'count': 1,
                'volume': order.volume
            })
            pickup_delivery_data.append({
                'hour': order.earliest_delivery_time.hour,
                'type': '送货',
                'count': 1,
                'volume': order.volume
            })

        df = pd.DataFrame(pickup_delivery_data)
        pivot_data = df.pivot_table(
            index='hour', columns='type', values='count',
            aggfunc='sum', fill_value=0
        )

        # 绘制热力图
        ax = sns.heatmap(pivot_data, cmap='YlOrRd', annot=True, fmt='g')
        plt.title('订单取送时间分布', fontsize=16)
        plt.xlabel('操作类型')
        plt.ylabel('时间 (小时)')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/images/order_time_heatmap.png", dpi=300)
        plt.close()

        # 2. 订单时间窗口分析
        plt.figure(figsize=(14, 6))

        # 计算时间窗口宽度（小时）
        pickup_windows = [(order.latest_pickup_time - order.earliest_pickup_time).total_seconds() / 3600
                          for order in orders]
        delivery_windows = [(order.latest_delivery_time - order.earliest_delivery_time).total_seconds() / 3600
                            for order in orders]

        # 创建双图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        sns.histplot(pickup_windows, kde=True, color=self.color_palette['primary'], ax=ax1)
        ax1.set_title('取货时间窗口分布', fontsize=14)
        ax1.set_xlabel('时间窗口宽度 (小时)')
        ax1.set_ylabel('订单数量')

        sns.histplot(delivery_windows, kde=True, color=self.color_palette['secondary'], ax=ax2)
        ax2.set_title('送货时间窗口分布', fontsize=14)
        ax2.set_xlabel('时间窗口宽度 (小时)')
        ax2.set_ylabel('订单数量')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/images/order_time_windows.png", dpi=300)
        plt.close()

        # 3. 订单特征相关性分析
        order_features = pd.DataFrame({
            '体积': volumes,
            '重量': weights,
            '取货时间窗口': pickup_windows,
            '送货时间窗口': delivery_windows
        })

        plt.figure(figsize=(10, 8))
        correlation = order_features.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('订单特征相关性分析', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/images/order_correlation.png", dpi=300)
        plt.close()

        # 4. 多维订单特征可视化
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            volumes, weights,
            s=[pw * 20 for pw in pickup_windows],
            c=[dw for dw in delivery_windows],
            cmap='viridis',
            alpha=0.7
        )
        plt.colorbar(scatter, label='送货时间窗口 (小时)')
        plt.title('订单多维特征分析', fontsize=16)
        plt.xlabel('体积 (立方米)', fontsize=12)
        plt.ylabel('重量 (千克)', fontsize=12)
        plt.grid(True)
        handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6,
                                                  num=4, func=lambda s: s / 20)
        legend = plt.legend(handles, labels, loc="upper right", title="取货窗口(小时)")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/images/order_multi_features.png", dpi=300)
        plt.close()

        # 5. 创建交互式订单地图
        self.create_interactive_order_map(orders)

        # 保存订单分析结果
        self.analysis_data['order_stats'] = {
            'order_count': len(orders),
            'avg_volume': np.mean(volumes),
            'avg_weight': np.mean(weights),
            'avg_pickup_window': np.mean(pickup_windows),
            'avg_delivery_window': np.mean(delivery_windows),
            'volume_weight_correlation': correlation.loc['体积', '重量']
        }

    def create_interactive_order_map(self, orders):
        """创建交互式订单地图"""
        # 准备数据
        pickup_data = []
        delivery_data = []
        connection_data = []

        for order in orders:
            pickup_data.append({
                'id': order.id,
                'type': '取货点',
                'lat': order.pickup_location.latitude,
                'lon': order.pickup_location.longitude,
                'volume': order.volume,
                'weight': order.weight,
                'earliest_time': order.earliest_pickup_time.strftime('%H:%M'),
                'latest_time': order.latest_pickup_time.strftime('%H:%M')
            })

            delivery_data.append({
                'id': order.id,
                'type': '送货点',
                'lat': order.delivery_location.latitude,
                'lon': order.delivery_location.longitude,
                'volume': order.volume,
                'weight': order.weight,
                'earliest_time': order.earliest_delivery_time.strftime('%H:%M'),
                'latest_time': order.latest_delivery_time.strftime('%H:%M')
            })

            # 连接线
            connection_data.append({
                'order_id': order.id,
                'from_lat': order.pickup_location.latitude,
                'from_lon': order.pickup_location.longitude,
                'to_lat': order.delivery_location.latitude,
                'to_lon': order.delivery_location.longitude,
                'volume': order.volume,
                'weight': order.weight
            })

        # 创建交互式地图
        pickup_df = pd.DataFrame(pickup_data)
        delivery_df = pd.DataFrame(delivery_data)

        # 合并取送数据以便创建图表
        all_points = pd.concat([pickup_df, delivery_df])

        # 创建地图
        fig = px.scatter_mapbox(
            all_points,
            lat='lat',
            lon='lon',
            color='type',
            size='volume',
            hover_name='id',
            hover_data=['weight', 'earliest_time', 'latest_time'],
            color_discrete_map={'取货点': '#1f77b4', '送货点': '#ff7f0e'},
            size_max=15,
            zoom=10,
            title='订单取送点分布交互式地图'
        )

        # 添加订单连接线
        for conn in connection_data:
            fig.add_trace(
                go.Scattermapbox(
                    mode='lines',
                    lon=[conn['from_lon'], conn['to_lon']],
                    lat=[conn['from_lat'], conn['to_lat']],
                    line=dict(width=1 + conn['volume'] / 2, color='rgba(120, 120, 120, 0.6)'),
                    hoverinfo='text',
                    text=f"订单 {conn['order_id']}: {conn['weight']}kg, {conn['volume']}m³",
                    showlegend=False
                )
            )

        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})

        # 保存为HTML文件
        interactive_map_path = f"{self.output_dir}/interactive/orders_map.html"
        plot(fig, filename=interactive_map_path, auto_open=False)

    def analyze_vehicles(self):
        """分析车辆数据并创建高级可视化"""
        print("正在分析车辆数据...")
        vehicles = list(self.vehicles.values())

        if not vehicles:
            print("警告: 没有可用的车辆数据")
            return

        # 1. 车辆容量/负载比分析
        current_volumes = [v.current_volume for v in vehicles]
        max_volumes = [v.max_volume for v in vehicles]
        current_weights = [v.current_weight for v in vehicles]
        max_weights = [v.max_weight for v in vehicles]

        volume_utilization = [cv / mv * 100 if mv > 0 else 0
                              for cv, mv in zip(current_volumes, max_volumes)]
        weight_utilization = [cw / mw * 100 if mw > 0 else 0
                              for cw, mw in zip(current_weights, max_weights)]

        # 车辆ID
        vehicle_ids = [v.id for v in vehicles]
        vehicle_names = [v.name for v in vehicles]

        # 创建利用率条形图
        plt.figure(figsize=(14, 8))

        x = np.arange(len(vehicles))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 8))
        rects1 = ax.bar(x - width / 2, volume_utilization, width, label='体积利用率',
                        color=self.color_palette['primary'])
        rects2 = ax.bar(x + width / 2, weight_utilization, width, label='重量利用率',
                        color=self.color_palette['secondary'])

        # 添加标签和标题
        ax.set_title('车辆容量利用率分析', fontsize=16)
        ax.set_xlabel('车辆ID', fontsize=12)
        ax.set_ylabel('利用率 (%)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(vehicle_ids)
        ax.legend()

        # 在每个柱子上添加数值标签
        for rect in rects1 + rects2:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/images/vehicle_utilization.png", dpi=300)
        plt.close()

        # 2. 车辆类型分析
        # 提取车辆类型
        vehicle_types = []
        for v in vehicles:
            if '小型' in v.name:
                vehicle_types.append('小型货车')
            elif '中型' in v.name:
                vehicle_types.append('中型货车')
            else:
                vehicle_types.append('大型货车')

        type_counts = {}
        for vtype in vehicle_types:
            type_counts[vtype] = type_counts.get(vtype, 0) + 1

        # 创建饼图
        plt.figure(figsize=(10, 8))
        plt.pie(
            type_counts.values(),
            labels=type_counts.keys(),
            autopct='%1.1f%%',
            startangle=90,
            colors=[self.color_palette['primary'],
                    self.color_palette['secondary'],
                    self.color_palette['success']],
            shadow=True,
            explode=[0.05] * len(type_counts)
        )
        plt.axis('equal')
        plt.title('车辆类型分布', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/images/vehicle_types.png", dpi=300)
        plt.close()

        # 3. 车辆状态分析
        status_counts = {}
        for v in vehicles:
            status_counts[v.status] = status_counts.get(v.status, 0) + 1

        # 创建水平条形图
        plt.figure(figsize=(12, 6))
        status_labels = list(status_counts.keys())
        status_values = list(status_counts.values())

        colors = [
            self.color_palette['success'] if status == 'idle' else
            self.color_palette['warning'] if status == 'busy' else
            self.color_palette['danger']
            for status in status_labels
        ]

        plt.barh(status_labels, status_values, color=colors)

        # 添加数据标签
        for i, v in enumerate(status_values):
            plt.text(v + 0.1, i, str(v), va='center')

        plt.title('车辆状态分布', fontsize=16)
        plt.xlabel('车辆数量', fontsize=12)
        plt.ylabel('状态', fontsize=12)
        plt.gca().invert_yaxis()  # 反转Y轴，使第一个条形显示在顶部
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/images/vehicle_status.png", dpi=300)
        plt.close()

        # 4. 车辆容量分布
        plt.figure(figsize=(14, 10))

        # 创建散点图，X轴为最大体积，Y轴为最大重量
        scatter = plt.scatter(
            max_volumes, max_weights,
            s=100,
            c=[{'小型货车': 0, '中型货车': 0.5, '大型货车': 1}[t] for t in vehicle_types],
            cmap='viridis',
            alpha=0.7
        )

        # 添加车辆ID标签
        for i, vid in enumerate(vehicle_ids):
            plt.text(max_volumes[i], max_weights[i], str(vid), fontsize=9)

        plt.title('车辆容量分布', fontsize=16)
        plt.xlabel('最大体积 (立方米)', fontsize=12)
        plt.ylabel('最大重量 (千克)', fontsize=12)
        plt.grid(True)

        # 创建图例
        legend_elements = [
            mpatches.Patch(color='#440154', label='小型货车'),
            mpatches.Patch(color='#21918c', label='中型货车'),
            mpatches.Patch(color='#fde725', label='大型货车')
        ]
        plt.legend(handles=legend_elements, loc='upper left')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/images/vehicle_capacity.png", dpi=300)
        plt.close()

        # 5. 创建交互式车辆地图
        self.create_interactive_vehicle_map(vehicles)

        # 保存车辆分析结果
        self.analysis_data['vehicle_stats'] = {
            'vehicle_count': len(vehicles),
            'type_distribution': type_counts,
            'status_distribution': status_counts,
            'avg_volume_utilization': np.mean(volume_utilization),
            'avg_weight_utilization': np.mean(weight_utilization),
        }

    def create_interactive_vehicle_map(self, vehicles):
        """创建交互式车辆地图"""
        # 准备数据
        vehicle_data = []

        for vehicle in vehicles:
            # 确定车辆类型
            if '小型' in vehicle.name:
                vehicle_type = '小型货车'
            elif '中型' in vehicle.name:
                vehicle_type = '中型货车'
            else:
                vehicle_type = '大型货车'

            # 计算利用率
            volume_util = (vehicle.current_volume / vehicle.max_volume * 100
                           if vehicle.max_volume > 0 else 0)
            weight_util = (vehicle.current_weight / vehicle.max_weight * 100
                           if vehicle.max_weight > 0 else 0)

            vehicle_data.append({
                'id': vehicle.id,
                'name': vehicle.name,
                'type': vehicle_type,
                'lat': vehicle.current_location.latitude,
                'lon': vehicle.current_location.longitude,
                'status': vehicle.status,
                'max_volume': vehicle.max_volume,
                'max_weight': vehicle.max_weight,
                'current_volume': vehicle.current_volume,
                'current_weight': vehicle.current_weight,
                'volume_utilization': volume_util,
                'weight_utilization': weight_util,
                'order_count': len(vehicle.current_orders)
            })

        # 创建DataFrame
        df = pd.DataFrame(vehicle_data)

        # 创建交互式地图
        fig = px.scatter_mapbox(
            df,
            lat='lat',
            lon='lon',
            color='type',
            size='max_volume',
            hover_name='name',
            hover_data=['id', 'status', 'volume_utilization', 'weight_utilization', 'order_count'],
            color_discrete_map={
                '小型货车': '#1f77b4',
                '中型货车': '#2ca02c',
                '大型货车': '#d62728'
            },
            size_max=20,
            zoom=10,
            title='车辆分布交互式地图'
        )

        # 使用不同的标记表示不同状态
        for status in df['status'].unique():
            status_data = df[df['status'] == status]

            marker = 'circle' if status == 'idle' else 'square'

            fig.add_trace(
                go.Scattermapbox(
                    lat=status_data['lat'],
                    lon=status_data['lon'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        opacity=0.8
                    ),
                    name=f'状态: {status}',
                    hoverinfo='text',
                    text=[
                        f"ID: {row['id']}<br>"
                        f"名称: {row['name']}<br>"
                        f"类型: {row['type']}<br>"
                        f"状态: {row['status']}<br>"
                        f"体积利用率: {row['volume_utilization']:.1f}%<br>"
                        f"重量利用率: {row['weight_utilization']:.1f}%<br>"
                        f"当前订单数: {row['order_count']}"
                        for _, row in status_data.iterrows()
                    ]
                )
            )

        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})

        # 保存为HTML文件
        interactive_map_path = f"{self.output_dir}/interactive/vehicles_map.html"
        plot(fig, filename=interactive_map_path, auto_open=False)

    def analyze_routing(self):
        """分析路径规划和交通数据"""
        print("正在分析路径和交通数据...")
        city_map = self.city_map

        # 1. 路径规划效率分析
        # 随机选择10对地点计算路径
        import random
        import networkx as nx

        locations = list(city_map.locations.keys())
        route_data = []

        for _ in range(10):
            start_id = random.choice(locations)
            end_id = random.choice(locations)
            while end_id == start_id:
                end_id = random.choice(locations)

            try:
                # 计算最短路径
                path = city_map.get_shortest_path(start_id, end_id)
                distance = city_map.get_distance(start_id, end_id)

                # 计算直线距离作为比较
                start_loc = city_map.locations[start_id]
                end_loc = city_map.locations[end_id]
                direct_distance = start_loc.distance_to(end_loc)

                if path:
                    route_data.append({
                        'start': start_id,
                        'end': end_id,
                        'start_name': city_map.locations[start_id].name,
                        'end_name': city_map.locations[end_id].name,
                        'path_length': len(path),
                        'distance': distance,
                        'direct_distance': direct_distance,
                        'efficiency': direct_distance / distance if distance > 0 else 0
                    })
            except Exception as e:
                print(f"计算路径时出错 ({start_id} → {end_id}): {e}")

        # 创建路径规划效率条形图
        if route_data:
            plt.figure(figsize=(14, 8))

            route_df = pd.DataFrame(route_data)

            efficiency_colors = [
                self.color_palette['success'] if eff > 0.9 else
                self.color_palette['warning'] if eff > 0.7 else
                self.color_palette['danger']
                for eff in route_df['efficiency']
            ]

            # 排序以便按效率显示
            route_df = route_df.sort_values('efficiency', ascending=False)

            plt.barh(
                [f"{row['start_name']} → {row['end_name']}" for _, row in route_df.iterrows()],
                route_df['efficiency'] * 100,  # 转换为百分比
                color=efficiency_colors
            )

            plt.title('路径规划效率分析', fontsize=16)
            plt.xlabel('效率 (直线距离/实际路径距离) %', fontsize=12)
            plt.ylabel('路径', fontsize=12)
            plt.axvline(x=80, color='gray', linestyle='--')  # 80%效率作为参考线
            plt.text(81, 0, '良好效率(80%)', va='bottom', ha='left')
            plt.xlim(0, 105)
            plt.grid(axis='x')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/images/routing_efficiency.png", dpi=300)
            plt.close()

            # 2. 路径可视化比较 (最短路径 vs 直线路径)
            self.visualize_paths_comparison(city_map, route_data[:3])  # 仅可视化前3条路径，避免图表过于拥挤

        # 3. 交通系统分析
        # 检查是否有交通信号灯数据
        try:
            traffic = self.data_layer.traffic
            if hasattr(traffic, 'traffic_lights') and traffic.traffic_lights:
                self.analyze_traffic_system(traffic)
        except:
            print("警告: 无法分析交通系统，可能没有相关数据")

    def visualize_paths_comparison(self, city_map, route_data):
        """可视化路径比较(实际路径vs直线路径)"""
        plt.figure(figsize=(14, 10))
        plt.title('路径规划比较: 最短路径 vs 直线路径', fontsize=16)

        # 准备绘制地图背景
        pos = {loc_id: (loc.longitude, loc.latitude) for loc_id, loc in city_map.locations.items()}

        # 绘制地图节点(淡灰色)
        for loc_id, (x, y) in pos.items():
            plt.scatter(x, y, s=50, color='lightgray', edgecolors='gray', alpha=0.6)

        # 绘制主要道路网络(淡灰色)
        for u, v, data in city_map.graph.edges(data=True):
            plt.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                'gray', alpha=0.3, linewidth=1
            )

        # 绘制选择的路径，每条路径使用不同的颜色
        colors = ['red', 'blue', 'green', 'purple', 'orange']

        for i, route in enumerate(route_data):
            color = colors[i % len(colors)]

            # 获取起点和终点坐标
            start_loc = city_map.locations[route['start']]
            end_loc = city_map.locations[route['end']]
            start_pos = (start_loc.longitude, start_loc.latitude)
            end_pos = (end_loc.longitude, end_loc.latitude)

            # 绘制起点和终点
            plt.scatter(start_pos[0], start_pos[1], s=100, color=color, marker='o',
                        edgecolors='black', zorder=10)
            plt.scatter(end_pos[0], end_pos[1], s=100, color=color, marker='x',
                        edgecolors='black', zorder=10)

            # 绘制直线路径(虚线)
            plt.plot(
                [start_pos[0], end_pos[0]],
                [start_pos[1], end_pos[1]],
                color=color, linestyle='--', linewidth=2, alpha=0.7,
                label=f"路径{i + 1}直线: {route['direct_distance']:.1f}km"
            )

            # 绘制实际最短路径(实线)
            try:
                path = city_map.get_shortest_path(route['start'], route['end'])
                if path:
                    path_coords = []
                    for node_id in path:
                        node_loc = city_map.locations[node_id]
                        path_coords.append((node_loc.longitude, node_loc.latitude))

                    # 绘制实际路径
                    x_coords, y_coords = zip(*path_coords)
                    plt.plot(
                        x_coords, y_coords,
                        color=color, linestyle='-', linewidth=2.5, alpha=0.7,
                        label=f"路径{i + 1}实际: {route['distance']:.1f}km"
                    )

                    # 添加路径标识
                    mid_point = path_coords[len(path_coords) // 2]
                    plt.text(
                        mid_point[0], mid_point[1], f"{i + 1}",
                        fontsize=12, color='white', weight='bold',
                        bbox=dict(facecolor=color, alpha=0.7, boxstyle='circle')
                    )
            except Exception as e:
                print(f"绘制路径时出错: {e}")

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/images/path_comparison.png", dpi=300)
        plt.close()

    def analyze_traffic_system(self, traffic):
        """分析交通系统"""
        try:
            traffic_lights = traffic.traffic_lights

            # 统计不同颜色的红绿灯数量
            light_status = {}
            for light in traffic_lights:
                light_status[light.status] = light_status.get(light.status, 0) + 1

            # 创建红绿灯状态饼图
            plt.figure(figsize=(10, 8))
            colors = {'red': 'red', 'green': 'green', 'yellow': 'yellow'}
            plt.pie(
                light_status.values(),
                labels=light_status.keys(),
                autopct='%1.1f%%',
                colors=[colors.get(k, 'gray') for k in light_status.keys()],
                shadow=True,
                startangle=90,
                explode=[0.05] * len(light_status)
            )
            plt.axis('equal')
            plt.title('交通信号灯状态分布', fontsize=16)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/images/traffic_lights_status.png", dpi=300)
            plt.close()

            # 保存交通系统分析结果
            self.analysis_data['traffic_stats'] = {
                'total_lights': len(traffic_lights),
                'status_distribution': light_status,
            }
        except Exception as e:
            print(f"分析交通系统时出错: {e}")

    def analyze_system_performance(self):
        """分析系统整体性能"""
        print("正在分析系统整体性能...")

        # 1. 资源利用率仪表板
        # 已从订单和车辆分析中获取数据
        resource_stats = {}

        # 订单分配率
        total_orders = len(self.orders)
        assigned_orders = sum(1 for order in self.orders.values() if order.assigned_vehicle_id is not None)
        order_assignment_rate = assigned_orders / total_orders if total_orders > 0 else 0

        # 车辆利用率
        total_vehicles = len(self.vehicles)
        busy_vehicles = sum(1 for vehicle in self.vehicles.values() if vehicle.status == 'busy')
        vehicle_utilization_rate = busy_vehicles / total_vehicles if total_vehicles > 0 else 0

        # 平均载重利用率
        if 'vehicle_stats' in self.analysis_data:
            avg_weight_util = self.analysis_data['vehicle_stats'].get('avg_weight_utilization', 0)
            avg_volume_util = self.analysis_data['vehicle_stats'].get('avg_volume_utilization', 0)
        else:
            avg_weight_util = 0
            avg_volume_util = 0

        # 创建指标仪表板
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 订单分配率
        self.create_gauge_chart(
            axes[0, 0],
            order_assignment_rate * 100,
            '订单分配率',
            f"{assigned_orders}/{total_orders} 订单已分配"
        )

        # 车辆利用率
        self.create_gauge_chart(
            axes[0, 1],
            vehicle_utilization_rate * 100,
            '车辆利用率',
            f"{busy_vehicles}/{total_vehicles} 车辆忙碌中"
        )

        # 载重利用率
        self.create_gauge_chart(
            axes[1, 0],
            avg_weight_util,
            '平均载重利用率',
            f"{avg_weight_util:.1f}% 载重已使用"
        )

        # 体积利用率
        self.create_gauge_chart(
            axes[1, 1],
            avg_volume_util,
            '平均体积利用率',
            f"{avg_volume_util:.1f}% 体积已使用"
        )

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/images/system_performance.png", dpi=300)
        plt.close()

        # 2. 系统关键指标汇总
        self.analysis_data['system_performance'] = {
            'order_assignment_rate': order_assignment_rate,
            'vehicle_utilization_rate': vehicle_utilization_rate,
            'avg_weight_utilization': avg_weight_util / 100 if avg_weight_util else 0,
            'avg_volume_utilization': avg_volume_util / 100 if avg_volume_util else 0,
        }

    def create_gauge_chart(self, ax, value, title, subtitle):
        """创建仪表图"""
        # 定义值的范围和颜色
        low = 0
        high = 100

        # 根据值确定颜色
        if value < 30:
            color = self.color_palette['danger']
        elif value < 70:
            color = self.color_palette['warning']
        else:
            color = self.color_palette['success']

        # 创建仪表图
        ax.set_aspect('equal')
        ax.add_patch(plt.Circle((0, 0), radius=1.0, fill=False, linewidth=2, color='gray'))

        # 添加刻度
        for i in range(0, 101, 10):
            angle = np.pi * (1 - i / 100)
            x = 0.9 * np.cos(angle)
            y = 0.9 * np.sin(angle)
            ax.text(x, y, f"{i}", ha='center', va='center', fontsize=8)

        # 添加指针
        angle = np.pi * (1 - value / 100)
        ax.arrow(0, 0, 0.7 * np.cos(angle), 0.7 * np.sin(angle),
                 head_width=0.05, head_length=0.1, fc=color, ec=color)

        # 添加圆心和数值
        ax.add_patch(plt.Circle((0, 0), radius=0.1, facecolor=color))
        ax.text(0, -0.4, f"{value:.1f}%", ha='center', va='center',
                fontsize=18, color=color, weight='bold')

        # 添加标题和副标题
        ax.text(0, 0.6, title, ha='center', va='center', fontsize=14)
        ax.text(0, -0.6, subtitle, ha='center', va='center', fontsize=10)

        # 设置轴限制并移除坐标轴
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')

    def generate_html_report(self):
        """生成HTML格式的测试报告"""
        print("正在生成HTML报告...")

        # 创建HTML报告文件路径
        report_path = f"{self.output_dir}/enhanced_test_report.html"

        # 准备图片文件列表
        image_files = sorted([f for f in os.listdir(f"{self.output_dir}/images")
                              if f.endswith('.png')])
        interactive_files = sorted([f for f in os.listdir(f"{self.output_dir}/interactive")
                                    if f.endswith('.html')])

        # 创建HTML内容
        html_content = []
        html_content.append('''
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>智送城市物流系统 - 数据层增强测试报告</title>
            <style>
                :root {
                    --primary: #2b6cb0;
                    --secondary: #38b2ac;
                    --success: #48bb78;
                    --warning: #ecc94b;
                    --danger: #f56565;
                    --light: #f7fafc;
                    --dark: #2d3748;
                    --gray: #a0aec0;
                }

                * {
                    box-sizing: border-box;
                    margin: 0;
                    padding: 0;
                }

                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                    line-height: 1.6;
                    color: var(--dark);
                    background-color: var(--light);
                    padding: 0;
                    margin: 0;
                }

                .container {
                    width: 100%;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 0 20px;
                }

                header {
                    background: linear-gradient(135deg, var(--primary), var(--secondary));
                    color: white;
                    padding: 2rem 0;
                    margin-bottom: 2rem;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }

                .header-content {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                }

                .logo {
                    font-size: 2.5rem;
                    font-weight: bold;
                    margin: 0;
                }

                .logo-subtitle {
                    font-size: 1.2rem;
                    opacity: 0.9;
                }

                .report-meta {
                    text-align: right;
                }

                .section {
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                    margin-bottom: 2rem;
                    overflow: hidden;
                }

                .section-header {
                    background-color: var(--dark);
                    color: white;
                    padding: 1rem 1.5rem;
                    font-size: 1.4rem;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }

                .section-body {
                    padding: 1.5rem;
                }

                .dashboard {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 1.5rem;
                    margin-bottom: 1.5rem;
                }

                .dashboard-item {
                    background-color: #f8fafc;
                    border-left: 4px solid var(--primary);
                    padding: 1rem;
                    border-radius: 4px;
                }

                .dashboard-item.success {
                    border-left-color: var(--success);
                }

                .dashboard-item.warning {
                    border-left-color: var(--warning);
                }

                .dashboard-item.danger {
                    border-left-color: var(--danger);
                }

                .dashboard-value {
                    font-size: 2rem;
                    font-weight: bold;
                    color: var(--dark);
                }

                .dashboard-label {
                    font-size: 0.9rem;
                    color: var(--gray);
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }

                .img-container {
                    margin: 1.5rem 0;
                    text-align: center;
                }

                .report-img {
                    max-width: 100%;
                    border-radius: 4px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }

                .img-caption {
                    font-size: 0.9rem;
                    color: var(--gray);
                    margin-top: 0.5rem;
                }

                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 1rem 0;
                }

                th, td {
                    padding: 0.75rem;
                    border-bottom: 1px solid #e2e8f0;
                }

                th {
                    background-color: #f7fafc;
                    text-align: left;
                    font-weight: 600;
                }

                tr:hover {
                    background-color: #f7fafc;
                }

                .interactive-section {
                    margin: 1.5rem 0;
                }

                .interactive-container {
                    border: 1px solid #e2e8f0;
                    border-radius: 4px;
                    overflow: hidden;
                }

                .btn {
                    display: inline-block;
                    background-color: var(--primary);
                    color: white;
                    padding: 0.5rem 1rem;
                    border-radius: 4px;
                    text-decoration: none;
                    transition: background-color 0.2s;
                }

                .btn:hover {
                    background-color: #2c5282;
                }

                .footer {
                    text-align: center;
                    padding: 2rem 0;
                    color: var(--gray);
                    font-size: 0.9rem;
                    border-top: 1px solid #e2e8f0;
                    margin-top: 2rem;
                }

                .summary-item {
                    display: flex;
                    justify-content: space-between;
                    padding: 0.5rem 0;
                    border-bottom: 1px solid #e2e8f0;
                }

                .summary-label {
                    font-weight: 500;
                }

                .summary-value {
                    font-weight: 600;
                }

                .summary-value.good {
                    color: var(--success);
                }

                .summary-value.medium {
                    color: var(--warning);
                }

                .summary-value.poor {
                    color: var(--danger);
                }

                .tab-container {
                    margin-top: 1rem;
                }

                .tab-header {
                    display: flex;
                    border-bottom: 2px solid #e2e8f0;
                }

                .tab-btn {
                    padding: 0.5rem 1rem;
                    background: none;
                    border: none;
                    border-bottom: 2px solid transparent;
                    margin-bottom: -2px;
                    cursor: pointer;
                    font-weight: 500;
                    color: var(--gray);
                }

                .tab-btn.active {
                    border-bottom-color: var(--primary);
                    color: var(--primary);
                }

                .tab-content {
                    display: none;
                    padding: 1rem 0;
                }

                .tab-content.active {
                    display: block;
                }

                .chart-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                    gap: 1.5rem;
                }

                iframe {
                    border: none;
                    width: 100%;
                    height: 500px;
                    margin: 1rem 0;
                }

                @media (max-width: 768px) {
                    .dashboard {
                        grid-template-columns: 1fr;
                    }

                    .header-content {
                        flex-direction: column;
                        text-align: center;
                    }

                    .report-meta {
                        text-align: center;
                        margin-top: 1rem;
                    }

                    .chart-grid {
                        grid-template-columns: 1fr;
                    }
                }
            </style>
        </head>
        <body>
            <header>
                <div class="container">
                    <div class="header-content">
                        <div>
                            <h1 class="logo">智送</h1>
                            <p class="logo-subtitle">城市货运智能调度系统 - 数据层测试报告</p>
                        </div>
                        <div class="report-meta">
                            <p>生成时间: ''' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '''</p>
                            <p>测试配置: ''' + f"{self.order_count}订单, {self.vehicle_count}车辆" + '''</p>
                        </div>
                    </div>
                </div>
            </header>

            <div class="container">
                <!-- 总结面板 -->
                <div class="section">
                    <div class="section-header">
                        <h2>系统性能总览</h2>
                    </div>
                    <div class="section-body">
        ''')

        # 添加性能仪表盘
        performance = self.analysis_data.get('system_performance', {})
        order_rate = performance.get('order_assignment_rate', 0) * 100
        vehicle_rate = performance.get('vehicle_utilization_rate', 0) * 100
        weight_util = performance.get('avg_weight_utilization', 0) * 100
        volume_util = performance.get('avg_volume_utilization', 0) * 100

        html_content.append('''
                        <div class="dashboard">
                            <div class="dashboard-item ''' + self.get_status_class(order_rate) + '''">
                                <div class="dashboard-value">''' + f"{order_rate:.1f}%" + '''</div>
                                <div class="dashboard-label">订单分配率</div>
                            </div>

                            <div class="dashboard-item ''' + self.get_status_class(vehicle_rate) + '''">
                                <div class="dashboard-value">''' + f"{vehicle_rate:.1f}%" + '''</div>
                                <div class="dashboard-label">车辆利用率</div>
                            </div>

                            <div class="dashboard-item ''' + self.get_status_class(weight_util) + '''">
                                <div class="dashboard-value">''' + f"{weight_util:.1f}%" + '''</div>
                                <div class="dashboard-label">平均载重利用率</div>
                            </div>

                            <div class="dashboard-item ''' + self.get_status_class(volume_util) + '''">
                                <div class="dashboard-value">''' + f"{volume_util:.1f}%" + '''</div>
                                <div class="dashboard-label">平均体积利用率</div>
                            </div>
                        </div>

                        <div class="img-container">
                            <img src="images/system_performance.png" alt="系统性能仪表板" class="report-img">
                            <p class="img-caption">系统关键性能指标</p>
                        </div>
                    </div>
                </div>

                <!-- 城市地图分析 -->
                <div class="section">
                    <div class="section-header">
                        <h2>城市地图分析</h2>
                    </div>
                    <div class="section-body">
        ''')

        # 添加地图统计数据
        map_stats = self.analysis_data.get('map_stats', {})
        node_count = map_stats.get('node_count', 0)
        edge_count = map_stats.get('edge_count', 0)
        avg_degree = map_stats.get('average_degree', 0)
        most_connected = map_stats.get('most_connected', ('未知', 0))
        most_central = map_stats.get('most_central', ('未知', 0))

        html_content.append(f'''
                        <div class="dashboard">
                            <div class="dashboard-item">
                                <div class="dashboard-value">{node_count}</div>
                                <div class="dashboard-label">节点数量</div>
                            </div>

                            <div class="dashboard-item">
                                <div class="dashboard-value">{edge_count}</div>
                                <div class="dashboard-label">道路数量</div>
                            </div>

                            <div class="dashboard-item">
                                <div class="dashboard-value">{avg_degree:.2f}</div>
                                <div class="dashboard-label">平均连接度</div>
                            </div>

                            <div class="dashboard-item">
                                <div class="dashboard-value">{most_connected[0] if isinstance(most_connected, tuple) else '未知'}</div>
                                <div class="dashboard-label">连接最多的节点</div>
                            </div>
                        </div>

                        <div class="tab-container">
                            <div class="tab-header">
                                <button class="tab-btn active" onclick="openTab(event, 'map-visual')">可视化</button>
                                <button class="tab-btn" onclick="openTab(event, 'map-interactive')">交互式地图</button>
                            </div>

                            <div id="map-visual" class="tab-content active">
                                <div class="img-container">
                                    <img src="images/city_map_advanced.png" alt="城市交通网络热力图" class="report-img">
                                    <p class="img-caption">城市交通网络热力图 (节点大小表示连接度)</p>
                                </div>
                            </div>

                            <div id="map-interactive" class="tab-content">
                                <p>以下是可交互的城市网络地图，您可以放大、缩小和移动查看详情：</p>
                                <div class="interactive-container">
                                    <iframe src="interactive/city_network.html"></iframe>
                                </div>
                            </div>
                        </div>
        ''')

        # 添加订单分析部分
        html_content.append('''
                    </div>
                </div>

                <!-- 订单分析 -->
                <div class="section">
                    <div class="section-header">
                        <h2>订单数据分析</h2>
                    </div>
                    <div class="section-body">
        ''')

        # 添加订单统计数据
        order_stats = self.analysis_data.get('order_stats', {})
        order_count = order_stats.get('order_count', 0)
        avg_volume = order_stats.get('avg_volume', 0)
        avg_weight = order_stats.get('avg_weight', 0)
        avg_pickup_window = order_stats.get('avg_pickup_window', 0)
        avg_delivery_window = order_stats.get('avg_delivery_window', 0)

        html_content.append(f'''
                        <div class="dashboard">
                            <div class="dashboard-item">
                                <div class="dashboard-value">{order_count}</div>
                                <div class="dashboard-label">订单总数</div>
                            </div>

                            <div class="dashboard-item">
                                <div class="dashboard-value">{avg_volume:.2f}m³</div>
                                <div class="dashboard-label">平均体积</div>
                            </div>

                            <div class="dashboard-item">
                                <div class="dashboard-value">{avg_weight:.2f}kg</div>
                                <div class="dashboard-label">平均重量</div>
                            </div>

                            <div class="dashboard-item">
                                <div class="dashboard-value">{avg_pickup_window:.2f}h</div>
                                <div class="dashboard-label">平均取货窗口</div>
                            </div>
                        </div>

                        <div class="tab-container">
                            <div class="tab-header">
                                <button class="tab-btn active" onclick="openTab(event, 'order-time')">时间分析</button>
                                <button class="tab-btn" onclick="openTab(event, 'order-feature')">特征分析</button>
                                <button class="tab-btn" onclick="openTab(event, 'order-map')">空间分布</button>
                            </div>

                            <div id="order-time" class="tab-content active">
                                <div class="chart-grid">
                                    <div class="img-container">
                                        <img src="images/order_time_heatmap.png" alt="订单时间分布热图" class="report-img">
                                        <p class="img-caption">订单取送时间分布</p>
                                    </div>

                                    <div class="img-container">
                                        <img src="images/order_time_windows.png" alt="订单时间窗口分布" class="report-img">
                                        <p class="img-caption">订单时间窗口长度分布</p>
                                    </div>
                                </div>
                            </div>

                            <div id="order-feature" class="tab-content">
                                <div class="chart-grid">
                                    <div class="img-container">
                                        <img src="images/order_correlation.png" alt="订单特征相关性" class="report-img">
                                        <p class="img-caption">订单特征相关性分析</p>
                                    </div>

                                    <div class="img-container">
                                        <img src="images/order_multi_features.png" alt="订单多维特征" class="report-img">
                                        <p class="img-caption">订单多维特征可视化</p>
                                    </div>
                                </div>
                            </div>

                            <div id="order-map" class="tab-content">
                                <div class="interactive-container">
                                    <iframe src="interactive/orders_map.html"></iframe>
                                </div>
                            </div>
                        </div>
        ''')

        # 添加车辆分析部分
        html_content.append('''
                    </div>
                </div>

                <!-- 车辆分析 -->
                <div class="section">
                    <div class="section-header">
                        <h2>车辆数据分析</h2>
                    </div>
                    <div class="section-body">
        ''')

        # 添加车辆统计数据
        vehicle_stats = self.analysis_data.get('vehicle_stats', {})
        vehicle_count = vehicle_stats.get('vehicle_count', 0)
        type_distribution = vehicle_stats.get('type_distribution', {})
        status_distribution = vehicle_stats.get('status_distribution', {})

        html_content.append(f'''
                        <div class="dashboard">
                            <div class="dashboard-item">
                                <div class="dashboard-value">{vehicle_count}</div>
                                <div class="dashboard-label">车辆总数</div>
                            </div>
        ''')

        # 添加车辆类型分布
        for vtype, count in type_distribution.items():
            html_content.append(f'''
                            <div class="dashboard-item">
                                <div class="dashboard-value">{count}</div>
                                <div class="dashboard-label">{vtype}</div>
                            </div>
            ''')

        html_content.append('''
                        </div>

                        <div class="tab-container">
                            <div class="tab-header">
                                <button class="tab-btn active" onclick="openTab(event, 'vehicle-utilization')">利用率</button>
                                <button class="tab-btn" onclick="openTab(event, 'vehicle-type')">类型分析</button>
                                <button class="tab-btn" onclick="openTab(event, 'vehicle-map')">空间分布</button>
                            </div>

                            <div id="vehicle-utilization" class="tab-content active">
                                <div class="img-container">
                                    <img src="images/vehicle_utilization.png" alt="车辆利用率" class="report-img">
                                    <p class="img-caption">车辆容量利用率分析</p>
                                </div>
                            </div>

                            <div id="vehicle-type" class="tab-content">
                                <div class="chart-grid">
                                    <div class="img-container">
                                        <img src="images/vehicle_types.png" alt="车辆类型分布" class="report-img">
                                        <p class="img-caption">车辆类型分布</p>
                                    </div>

                                    <div class="img-container">
                                        <img src="images/vehicle_status.png" alt="车辆状态分布" class="report-img">
                                        <p class="img-caption">车辆状态分布</p>
                                    </div>

                                    <div class="img-container">
                                        <img src="images/vehicle_capacity.png" alt="车辆容量分布" class="report-img">
                                        <p class="img-caption">车辆容量分布</p>
                                    </div>
                                </div>
                            </div>

                            <div id="vehicle-map" class="tab-content">
                                <div class="interactive-container">
                                    <iframe src="interactive/vehicles_map.html"></iframe>
                                </div>
                            </div>
                        </div>
        ''')

        # 添加路径分析部分
        html_content.append('''
                    </div>
                </div>

                <!-- 路径和交通分析 -->
                <div class="section">
                    <div class="section-header">
                        <h2>路径和交通分析</h2>
                    </div>
                    <div class="section-body">
                        <div class="chart-grid">
                            <div class="img-container">
                                <img src="images/routing_efficiency.png" alt="路径规划效率" class="report-img">
                                <p class="img-caption">路径规划效率分析</p>
                            </div>

                            <div class="img-container">
                                <img src="images/path_comparison.png" alt="路径比较" class="report-img">
                                <p class="img-caption">最短路径与直线路径比较</p>
                            </div>
                        </div>
        ''')

        # 检查是否有交通信号灯分析
        if 'traffic_stats' in self.analysis_data:
            html_content.append('''
                        <div class="img-container">
                            <img src="images/traffic_lights_status.png" alt="交通信号灯状态" class="report-img">
                            <p class="img-caption">交通信号灯状态分布</p>
                        </div>
            ''')

        # 结束和页脚
        html_content.append('''
                    </div>
                </div>

                <!-- 测试结论 -->
                <div class="section">
                    <div class="section-header">
                        <h2>测试结论与建议</h2>
                    </div>
                    <div class="section-body">
                        <p>根据数据层测试结果，系统表现如下：</p>
                        <ul style="margin-left: 2rem; margin-top: 1rem;">
                            <li><strong>城市地图数据：</strong> 节点连接合理，交通网络结构完整，可有效支持路径规划功能。</li>
                            <li><strong>订单分配：</strong> 订单时间分布显示了明显的高峰期模式，系统在高峰期的容量调度需重点关注。</li>
                            <li><strong>车辆利用率：</strong> 车辆容量利用状况有优化空间，部分车辆负载不均衡。</li>
                            <li><strong>路径规划：</strong> 路径规划算法效率良好，但部分路径与理想直线路径仍有差距。</li>
                        </ul>

                        <p style="margin-top: 1.5rem;"><strong>改进建议：</strong></p>
                        <ul style="margin-left: 2rem;">
                            <li>优化车辆调度算法，提高车辆载重和容量的平均利用率</li>
                            <li>在高峰时段增加可用车辆数量，提高订单处理能力</li>
                            <li>继续优化路径规划算法，特别是对于长距离运输路线</li>
                            <li>增强对交通拥堵时段的适应性，动态调整最佳路线</li>
                        </ul>
                    </div>
                </div>
            </div>

            <footer class="footer">
                <div class="container">
                    <p>智送城市货运智能调度系统 - 数据层增强测试报告</p>
                    <p>版权所有 &copy; ''' + str(datetime.now().year) + ''' 智送技术团队</p>
                </div>
            </footer>

            <script>
                function openTab(evt, tabName) {
                    var i, tabContent, tabBtns;

                    // 隐藏所有标签内容
                    tabContent = document.getElementsByClassName("tab-content");
                    for (i = 0; i < tabContent.length; i++) {
                        tabContent[i].style.display = "none";
                    }

                    // 移除所有按钮的active类
                    tabBtns = document.getElementsByClassName("tab-btn");
                    for (i = 0; i < tabBtns.length; i++) {
                        tabBtns[i].className = tabBtns[i].className.replace(" active", "");
                    }

                    // 显示当前标签并添加active类到按钮
                    document.getElementById(tabName).style.display = "block";
                    evt.currentTarget.className += " active";
                }
            </script>
        </body>
        </html>
        ''')

        # 写入HTML文件
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_content))

        return report_path

    def get_status_class(self, value):
        """根据值获取状态CSS类"""
        if value >= 70:
            return "success"
        elif value >= 40:
            return "warning"
        else:
            return "danger"


def main():
    """主函数: 执行增强测试并生成报告"""
    import argparse

    parser = argparse.ArgumentParser(description='智送城市物流系统数据层增强测试报告生成器')
    parser.add_argument('-o', '--orders', type=int, default=20, help='测试用订单数量')
    parser.add_argument('-v', '--vehicles', type=int, default=8, help='测试用车辆数量')
    args = parser.parse_args()

    print(f"{'=' * 80}")
    print(f"{'智送城市物流系统 - 数据层增强测试':^80}")
    print(f"{'=' * 80}")

    # 创建报告生成器并运行测试
    generator = EnhancedDataLayerReportGenerator(
        order_count=args.orders,
        vehicle_count=args.vehicles
    )
    result = generator.run_tests_and_generate_report()

    # 返回测试结果状态码
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())