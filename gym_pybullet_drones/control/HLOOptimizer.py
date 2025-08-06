# gym_pybullet_drones/control/HLOOptimizer.py
"""
人类学习优化算法(HLO)用于PID参数调优
"""

import numpy as np
import copy
from typing import Callable, Dict, Tuple


class HLOOptimizer:
    """人类学习优化算法类"""

    def __init__(self,
                 population_size: int = 20,
                 max_iterations: int = 30,
                 learning_rate: float = 0.9):

        self.population_size = population_size
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.random_learning_prob = 0.15
        self.individual_learning_prob = 0.5
        self.social_learning_prob = 0.35

        # F450 PID参数边界
        # [P_FOR_x, P_FOR_y, P_FOR_z, I_FOR_x, I_FOR_y, I_FOR_z, D_FOR_x, D_FOR_y, D_FOR_z,
        #  P_TOR_roll, P_TOR_pitch, P_TOR_yaw, I_TOR_roll, I_TOR_pitch, I_TOR_yaw,
        #  D_TOR_roll, D_TOR_pitch, D_TOR_yaw]
        self.lower_bounds = np.array([
            0.1, 0.1, 0.3,  # 位置P系数
            0.005, 0.005, 0.02,  # 位置I系数
            0.05, 0.05, 0.1,  # 位置D系数
            2000, 2000, 1000,  # 姿态P系数
            50, 50, 50,  # 姿态I系数
            500, 500, 300  # 姿态D系数
        ])

        self.upper_bounds = np.array([
            1.2, 1.2, 3.0,  # 位置P系数
            0.08, 0.08, 0.3,  # 位置I系数
            0.8, 0.8, 1.0,  # 位置D系数
            15000, 15000, 8000,  # 姿态P系数
            800, 800, 800,  # 姿态I系数
            6000, 6000, 3000  # 姿态D系数
        ])

        self.dim = len(self.lower_bounds)
        self.population = []
        self.fitness_values = []
        self.best_individual = None
        self.best_fitness = float('inf')
        self.fitness_history = []

    def initialize_population(self):
        """初始化种群"""
        self.population = []
        for i in range(self.population_size):
            individual = np.random.uniform(self.lower_bounds, self.upper_bounds)
            self.population.append(individual)

    def random_learning_operator(self, individual: np.ndarray) -> np.ndarray:
        """随机学习算子"""
        new_individual = individual.copy()
        dims_to_modify = np.random.choice(self.dim,
                                          size=np.random.randint(1, self.dim // 3 + 1),
                                          replace=False)

        for dim in dims_to_modify:
            learning_range = (self.upper_bounds[dim] - self.lower_bounds[dim]) * 0.15
            new_individual[dim] = np.clip(
                individual[dim] + np.random.uniform(-learning_range, learning_range),
                self.lower_bounds[dim],
                self.upper_bounds[dim]
            )
        return new_individual

    def individual_learning_operator(self, individual: np.ndarray, individual_best: np.ndarray) -> np.ndarray:
        """个体学习算子"""
        new_individual = individual.copy()
        for i in range(self.dim):
            if np.random.random() < self.learning_rate:
                direction = individual_best[i] - individual[i]
                step = np.random.uniform(0, 1) * direction
                new_individual[i] = np.clip(
                    individual[i] + step,
                    self.lower_bounds[i],
                    self.upper_bounds[i]
                )
        return new_individual

    def social_learning_operator(self, individual: np.ndarray) -> np.ndarray:
        """社会学习算子"""
        new_individual = individual.copy()
        other_individuals = [p for p in self.population if not np.array_equal(p, individual)]
        if len(other_individuals) > 0:
            teacher = other_individuals[np.random.randint(len(other_individuals))]
            for i in range(self.dim):
                if np.random.random() < self.learning_rate * 0.7:
                    direction = teacher[i] - individual[i]
                    step = np.random.uniform(0, 0.4) * direction
                    new_individual[i] = np.clip(
                        individual[i] + step,
                        self.lower_bounds[i],
                        self.upper_bounds[i]
                    )
        return new_individual

    def optimize(self, fitness_function: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float]:
        """
        执行HLO优化

        参数:
        - fitness_function: 适应度评估函数，输入PID参数数组，输出适应度值

        返回:
        - best_individual: 最佳个体参数
        - best_fitness: 最佳适应度
        """
        print("开始HLO-PID参数优化...")
        print(f"种群大小: {self.population_size}, 最大迭代数: {self.max_iterations}")

        # 初始化
        self.initialize_population()
        individual_best = copy.deepcopy(self.population)
        individual_best_fitness = [float('inf')] * self.population_size

        # 评估初始种群
        print("评估初始种群...")
        for i in range(self.population_size):
            fitness = fitness_function(self.population[i])
            self.fitness_values.append(fitness)
            individual_best_fitness[i] = fitness

            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = self.population[i].copy()

            print(f"个体 {i + 1}/{self.population_size}, 适应度: {fitness:.4f}")

        self.fitness_history.append(self.best_fitness)
        print(f"初始最佳适应度: {self.best_fitness:.6f}")

        # 主优化循环
        for iteration in range(self.max_iterations):
            print(f"\n迭代 {iteration + 1}/{self.max_iterations}")
            new_population = []

            for i in range(self.population_size):
                current_individual = self.population[i]

                # 选择学习策略
                rand_num = np.random.random()

                if rand_num < self.random_learning_prob:
                    new_individual = self.random_learning_operator(current_individual)
                elif rand_num < self.random_learning_prob + self.individual_learning_prob:
                    new_individual = self.individual_learning_operator(current_individual, individual_best[i])
                else:
                    new_individual = self.social_learning_operator(current_individual)

                # 评估新个体
                new_fitness = fitness_function(new_individual)

                # 更新策略
                if new_fitness < self.fitness_values[i]:
                    new_population.append(new_individual)
                    self.fitness_values[i] = new_fitness

                    if new_fitness < individual_best_fitness[i]:
                        individual_best[i] = new_individual.copy()
                        individual_best_fitness[i] = new_fitness

                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_individual = new_individual.copy()
                        print(f"  新最佳适应度: {self.best_fitness:.6f}")
                else:
                    new_population.append(current_individual)

            self.population = new_population
            self.fitness_history.append(self.best_fitness)

            # 动态调整学习概率
            progress = (iteration + 1) / self.max_iterations
            self.random_learning_prob = 0.15 * (1 - progress)
            self.individual_learning_prob = 0.5 + 0.2 * progress

            print(f"迭代 {iteration + 1} 完成, 当前最佳适应度: {self.best_fitness:.6f}")

        return self.best_individual, self.best_fitness

    def get_optimized_params_dict(self) -> Dict:
        """获取优化后的参数字典，用于传递给DSLPIDControl"""
        if self.best_individual is None:
            raise ValueError("请先运行optimize()方法")

        return {
            'P_COEFF_FOR': self.best_individual[0:3],
            'I_COEFF_FOR': self.best_individual[3:6],
            'D_COEFF_FOR': self.best_individual[6:9],
            'P_COEFF_TOR': self.best_individual[9:12],
            'I_COEFF_TOR': self.best_individual[12:15],
            'D_COEFF_TOR': self.best_individual[15:18]
        }
