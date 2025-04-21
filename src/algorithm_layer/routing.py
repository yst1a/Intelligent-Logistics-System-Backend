"""
路径规划算法模块
负责优化车辆行驶路径
"""
from typing import List, Dict, Tuple, Optional, Set
import random
from datetime import datetime

from data_layer.order import Order
from data_layer.vehicle import Vehicle
from data_layer.city_map import CityMap

from algorithm_layer.base import Route, RoutePoint, Solution, Algorithm
from algorithm_layer.evaluation import RouteEvaluator, CostCalculator, FeasibilityChecker


class BasicRoutingAlgorithm(Algorithm):
    """基本路径规划算法
    
    对已分配给车辆的订单进行简单的路径优化。
    采用贪婪策略，不断选择当前位置最近的下一个点。
    """
    
    def __init__(self, city_map: CityMap):
        """
        Args:
            city_map: 城市地图对象
        """
        super().__init__(city_map)
        self.route_evaluator = RouteEvaluator(city_map)
        self.cost_calculator = CostCalculator(city_map)

    def solve(self, orders: List[Order], vehicles: List[Vehicle]) -> Solution:
        """使用贪心策略优化每辆车的路线

        Args:
            orders: 订单列表
            vehicles: 车辆列表

        Returns:
            优化后的解决方案
        """
        # 初始化解决方案
        solution = Solution()
        current_time = datetime.now()

        # 为每辆车创建订单集合
        vehicle_orders: Dict[int, List[Order]] = {v.id: [] for v in vehicles}

        # 按车辆ID分组已分配的订单
        for order in orders:
            if order.is_assigned:
                vehicle_id = order.assigned_vehicle_id
                if vehicle_id in vehicle_orders:
                    vehicle_orders[vehicle_id].append(order)

        # 为每辆车规划路线
        for vehicle in vehicles:
            # 创建新路线
            route = Route(vehicle)

            # 获取该车辆的订单
            vehicle_order_list = vehicle_orders[vehicle.id]

            # 如果没有订单，添加空路线
            if not vehicle_order_list:
                solution.add_route(route)
                continue

            # 使用贪心算法规划取货和送货顺序
            current_location = vehicle.current_location

            # 创建订单ID到订单对象的映射
            order_map = {order.id: order for order in vehicle_order_list}

            # 使用订单ID代替订单对象
            remaining_pickup_ids = set(order.id for order in vehicle_order_list)
            remaining_delivery_ids = set()

            # 当还有点需要访问时
            while remaining_pickup_ids or remaining_delivery_ids:
                next_point = None
                best_distance = float('inf')
                best_id = None

                # 考虑所有可能的下一个取货点
                for order_id in remaining_pickup_ids:
                    order = order_map[order_id]
                    distance = current_location.distance_to(order.pickup_location)
                    if distance < best_distance:
                        best_distance = distance
                        next_point = RoutePoint(order.pickup_location, RoutePoint.ACTION_PICKUP, order)
                        best_id = order_id

                # 考虑所有可能的下一个送货点
                for order_id in remaining_delivery_ids:
                    order = order_map[order_id]
                    distance = current_location.distance_to(order.delivery_location)
                    if distance < best_distance:
                        best_distance = distance
                        next_point = RoutePoint(order.delivery_location, RoutePoint.ACTION_DELIVERY, order)
                        best_id = order_id

                # 添加下一个点到路线
                route.add_point(next_point)

                # 更新当前位置
                current_location = next_point.location

                # 更新剩余点集合
                if next_point.is_pickup:
                    remaining_pickup_ids.remove(best_id)
                    remaining_delivery_ids.add(best_id)
                else:  # 送货点
                    remaining_delivery_ids.remove(best_id)

            # 评估路线
            self.route_evaluator.evaluate_route(route, current_time)

            # 添加路线到解决方案
            solution.add_route(route)

        # 标记未分配的订单
        for order in orders:
            if not order.is_assigned:
                solution.add_unassigned_order(order.id)

        # 更新解决方案的总体指标
        solution.update_metrics()
        return solution


class OrderFirstRoutingAlgorithm(Algorithm):
    """订单优先路径规划算法

    先完成一个订单的取送，再开始下一个订单，适合货物时间窗约束严格的场景。
    """

    def __init__(self, city_map: CityMap):
        """
        Args:
            city_map: 城市地图对象
        """
        super().__init__(city_map)
        self.route_evaluator = RouteEvaluator(city_map)
        self.cost_calculator = CostCalculator(city_map)

    def solve(self, orders: List[Order], vehicles: List[Vehicle]) -> Solution:
        """使用订单优先策略优化每辆车的路线

        Args:
            orders: 订单列表
            vehicles: 车辆列表

        Returns:
            优化后的解决方案
        """
        # 初始化解决方案
        solution = Solution()
        current_time = datetime.now()

        # 为每辆车创建订单集合
        vehicle_orders: Dict[int, List[Order]] = {v.id: [] for v in vehicles}

        # 按车辆ID分组已分配的订单
        for order in orders:
            if order.is_assigned:
                vehicle_id = order.assigned_vehicle_id
                if vehicle_id in vehicle_orders:
                    vehicle_orders[vehicle_id].append(order)

        # 为每辆车规划路线
        for vehicle in vehicles:
            # 创建新路线
            route = Route(vehicle)

            # 获取该车辆的订单
            vehicle_order_list = vehicle_orders[vehicle.id]

            # 如果没有订单，添加空路线
            if not vehicle_order_list:
                solution.add_route(route)
                continue

            # 按最早取货时间对订单排序
            sorted_orders = sorted(vehicle_order_list, key=lambda o: o.earliest_pickup_time)

            # 为每个订单添加取货和送货点
            for order in sorted_orders:
                pickup_point = RoutePoint(order.pickup_location, RoutePoint.ACTION_PICKUP, order)
                delivery_point = RoutePoint(order.delivery_location, RoutePoint.ACTION_DELIVERY, order)

                route.add_point(pickup_point)
                route.add_point(delivery_point)

            # 评估路线
            self.route_evaluator.evaluate_route(route, current_time)

            # 添加路线到解决方案
            solution.add_route(route)

        # 标记未分配的订单
        for order in orders:
            if not order.is_assigned:
                solution.add_unassigned_order(order.id)

        # 更新解决方案的总体指标
        solution.update_metrics()
        return solution


class LocalSearchRoutingAlgorithm(Algorithm):
    """局部搜索路径优化算法

    在初始解的基础上，通过局部搜索操作不断改进解决方案。
    包括2-opt操作、重定位操作和交换操作。
    """

    def __init__(self, city_map: CityMap, max_iterations: int = 100):
        """
        Args:
            city_map: 城市地图对象
            max_iterations: 最大迭代次数
        """
        super().__init__(city_map)
        self.route_evaluator = RouteEvaluator(city_map)
        self.cost_calculator = CostCalculator(city_map)
        self.feasibility_checker = FeasibilityChecker(city_map)
        self.max_iterations = max_iterations

    def solve(self, orders: List[Order], vehicles: List[Vehicle]) -> Solution:
        """使用局部搜索算法优化路径

        Args:
            orders: 订单列表
            vehicles: 车辆列表

        Returns:
            优化后的解决方案
        """
        # 使用订单优先算法生成初始解
        initial_algorithm = OrderFirstRoutingAlgorithm(self.city_map)
        solution = initial_algorithm.solve(orders, vehicles)
        current_time = datetime.now()

        # 评估初始解
        self.route_evaluator.evaluate_solution(solution, current_time)
        initial_cost = self.cost_calculator.calculate_solution_cost(solution, current_time)

        print(f"初始解成本: {initial_cost:.2f}")

        # 局部搜索优化
        current_solution = solution
        current_cost = initial_cost

        for iteration in range(self.max_iterations):
            # 尝试局部搜索操作
            improved = False

            # 对每辆车的路线尝试优化
            for vehicle_id, route in current_solution.routes.items():
                if len(route.points) < 4:
                    continue  # 至少需要两个订单才能进行优化

                # 尝试2-opt操作
                improved_by_2opt = self._apply_2opt(route, current_time)

                # 尝试重定位操作
                improved_by_relocate = self._apply_relocate(route, current_time)

                # 尝试交换操作
                improved_by_swap = self._apply_swap(route, current_time)

                if improved_by_2opt or improved_by_relocate or improved_by_swap:
                    improved = True

            # 如果没有任何改进，结束迭代
            if not improved:
                break

            # 重新评估解决方案
            self.route_evaluator.evaluate_solution(current_solution, current_time)
            new_cost = self.cost_calculator.calculate_solution_cost(current_solution, current_time)

            # 更新当前成本
            current_cost = new_cost

        print(f"优化后成本: {current_cost:.2f}, 迭代次数: {iteration+1}")
        return current_solution

    def _apply_2opt(self, route: Route, current_time: datetime) -> bool:
        """应用2-opt操作来优化路线

        在满足取送顺序约束的前提下反转部分路径。

        Args:
            route: 要优化的路线
            current_time: 当前时间

        Returns:
            是否有改进
        """
        if len(route.points) < 4:
            return False

        original_cost = self.cost_calculator.calculate_route_cost(route, current_time)
        improved = False

        # 尝试反转路径的各个子段
        for i in range(0, len(route.points) - 3):
            for j in range(i + 2, len(route.points) - 1):
                # 创建路线的临时副本
                temp_route = route.copy()

                # 反转子段
                temp_route.points[i+1:j+1] = reversed(temp_route.points[i+1:j+1])

                # 检查反转后的路线是否仍然可行
                is_feasible, _ = self.feasibility_checker.is_route_feasible(temp_route, current_time)
                if not is_feasible:
                    continue

                # 评估新路线
                self.route_evaluator.evaluate_route(temp_route, current_time)
                new_cost = self.cost_calculator.calculate_route_cost(temp_route, current_time)

                # 如果有改进，应用该操作
                if new_cost < original_cost:
                    route.points = temp_route.points
                    route.total_distance = temp_route.total_distance
                    route.total_time = temp_route.total_time
                    route.violations = temp_route.violations
                    original_cost = new_cost
                    improved = True

        return improved

    def _apply_relocate(self, route: Route, current_time: datetime) -> bool:
        """应用重定位操作来优化路线

        将一个订单的取送点移动到新位置，保持取送顺序约束。

        Args:
            route: 要优化的路线
            current_time: 当前时间

        Returns:
            是否有改进
        """
        if len(route.points) < 4:
            return False

        original_cost = self.cost_calculator.calculate_route_cost(route, current_time)
        improved = False

        # 获取订单到点的映射
        order_points = {}
        for i, point in enumerate(route.points):
            if point.order.id not in order_points:
                order_points[point.order.id] = [i, -1]
            else:
                order_points[point.order.id][1] = i

        # 对每个订单尝试重定位
        for order_id, (pickup_idx, delivery_idx) in order_points.items():
            # 尝试所有可能的新位置
            for new_pickup_idx in range(len(route.points)):
                if new_pickup_idx == pickup_idx:
                    continue

                # 对于每个新取货位置，尝试所有新送货位置
                for new_delivery_idx in range(new_pickup_idx + 1, len(route.points) + 1):
                    if new_delivery_idx == delivery_idx:
                        continue

                    # 创建路线的临时副本
                    temp_route = Route(route.vehicle)

                    # 取出原来的点
                    pickup_point = route.points[pickup_idx]
                    delivery_point = route.points[delivery_idx]
                    temp_points = [p for i, p in enumerate(route.points) if i != pickup_idx and i != delivery_idx]

                    # 调整新位置的索引（考虑到删除点后的索引变化）
                    adjusted_new_pickup_idx = new_pickup_idx
                    if pickup_idx < new_pickup_idx:
                        adjusted_new_pickup_idx -= 1
                    if delivery_idx < new_pickup_idx:
                        adjusted_new_pickup_idx -= 1

                    adjusted_new_delivery_idx = new_delivery_idx
                    if pickup_idx < new_delivery_idx:
                        adjusted_new_delivery_idx -= 1
                    if delivery_idx < new_delivery_idx:
                        adjusted_new_delivery_idx -= 1

                    # 插入点到新位置
                    temp_points.insert(adjusted_new_pickup_idx, pickup_point)
                    if adjusted_new_delivery_idx <= adjusted_new_pickup_idx:
                        adjusted_new_delivery_idx += 1
                    temp_points.insert(adjusted_new_delivery_idx, delivery_point)

                    temp_route.points = temp_points

                    # 检查新路线是否可行
                    is_feasible, _ = self.feasibility_checker.is_route_feasible(temp_route, current_time)
                    if not is_feasible:
                        continue

                    # 评估新路线
                    self.route_evaluator.evaluate_route(temp_route, current_time)
                    new_cost = self.cost_calculator.calculate_route_cost(temp_route, current_time)

                    # 如果有改进，应用该操作
                    if new_cost < original_cost:
                        route.points = temp_route.points
                        route.total_distance = temp_route.total_distance
                        route.total_time = temp_route.total_time
                        route.violations = temp_route.violations
                        original_cost = new_cost
                        improved = True

                        # 更新订单点映射
                        order_points = {}
                        for i, point in enumerate(route.points):
                            if point.order.id not in order_points:
                                order_points[point.order.id] = [i, -1]
                            else:
                                order_points[point.order.id][1] = i

        return improved

    def _apply_swap(self, route: Route, current_time: datetime) -> bool:
        """应用交换操作来优化路线

        交换两个订单的位置，保持取送顺序约束。

        Args:
            route: 要优化的路线
            current_time: 当前时间

        Returns:
            是否有改进
        """
        # 获取路线中的唯一订单
        order_ids = set(point.order.id for point in route.points)
        if len(order_ids) < 2:
            return False

        # 计算原始路线成本
        original_cost = self.cost_calculator.calculate_route_cost(route, current_time)
        improved = False

        # 获取订单到点的映射
        # 格式: {order_id: [pickup_index, delivery_index]}
        order_points = {}
        for i, point in enumerate(route.points):
            if point.order.id not in order_points:
                order_points[point.order.id] = [i, -1]  # 初始化取货点索引
            else:
                order_points[point.order.id][1] = i  # 设置送货点索引

        # 尝试交换每对不同的订单
        order_ids = list(order_points.keys())
        for i in range(len(order_ids)):
            for j in range(i + 1, len(order_ids)):
                order_id1 = order_ids[i]
                order_id2 = order_ids[j]

                # 获取两个订单的取送点索引
                pickup_idx1, delivery_idx1 = order_points[order_id1]
                pickup_idx2, delivery_idx2 = order_points[order_id2]

                # 创建路线的临时副本用于尝试交换
                temp_route = Route(route.vehicle)
                temp_points = route.points.copy()

                # 获取要交换的四个点
                pickup_point1 = temp_points[pickup_idx1]
                delivery_point1 = temp_points[delivery_idx1]
                pickup_point2 = temp_points[pickup_idx2]
                delivery_point2 = temp_points[delivery_idx2]

                # 如果点1在点2之后，交换它们以简化操作
                if pickup_idx1 > pickup_idx2:
                    pickup_idx1, pickup_idx2 = pickup_idx2, pickup_idx1
                    delivery_idx1, delivery_idx2 = delivery_idx2, delivery_idx1
                    pickup_point1, pickup_point2 = pickup_point2, pickup_point1
                    delivery_point1, delivery_point2 = delivery_point2, delivery_point1

                # 从路线中移除四个点(从后向前删除以避免索引变化)
                points_to_remove = [pickup_idx1, delivery_idx1, pickup_idx2, delivery_idx2]
                points_to_remove.sort(reverse=True)
                removed_points = []
                for idx in points_to_remove:
                    removed_points.insert(0, temp_points.pop(idx))

                # 按新顺序插入点:
                # 2的取货点 -> 2的送货点 -> 1的取货点 -> 1的送货点
                insertion_points = [pickup_point2, delivery_point2, pickup_point1, delivery_point1]

                # 在原始位置插入交换后的点
                adjusted_indices = [pickup_idx1, pickup_idx1 + 1, pickup_idx1 + 2, pickup_idx1 + 3]
                for idx, point in zip(adjusted_indices, insertion_points):
                    temp_points.insert(min(idx, len(temp_points)), point)

                temp_route.points = temp_points

                # 检查新路线是否满足约束条件
                is_feasible, _ = self.feasibility_checker.is_route_feasible(temp_route, current_time)
                if not is_feasible:
                    continue

                # 评估新路线并计算成本
                self.route_evaluator.evaluate_route(temp_route, current_time)
                new_cost = self.cost_calculator.calculate_route_cost(temp_route, current_time)

                # 如果新路线成本更低，应用交换操作
                if new_cost < original_cost:
                    route.points = temp_route.points
                    route.total_distance = temp_route.total_distance
                    route.total_time = temp_route.total_time
                    route.violations = temp_route.violations
                    original_cost = new_cost
                    improved = True

                    # 更新订单点映射
                    order_points = {}
                    for i, point in enumerate(route.points):
                        if point.order.id not in order_points:
                            order_points[point.order.id] = [i, -1]
                        else:
                            order_points[point.order.id][1] = i
                    break

            if improved:
                break

        return improved