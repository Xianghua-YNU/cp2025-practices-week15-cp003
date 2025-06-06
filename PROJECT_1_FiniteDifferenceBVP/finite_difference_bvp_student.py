#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目1：二阶常微分方程边值问题数值解法 - 学生代码模板

本项目要求实现两种数值方法求解边值问题：
1. 有限差分法 (Finite Difference Method)
2. scipy.integrate.solve_bvp 方法

问题设定：
y''(x) + sin(x) * y'(x) + exp(x) * y(x) = x^2
边界条件：y(0) = 0, y(5) = 3

学生需要完成所有标记为 TODO 的函数实现。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.linalg import solve


# ============================================================================ #
# 方法1：有限差分法 (Finite Difference Method)
# ============================================================================ #
def solve_bvp_finite_difference(n):
    h = 5 / (n + 1)
    x = np.linspace(0, 5, n + 2)

    A = np.zeros((n, n))
    b = np.zeros(n)

    for i in range(n):
        xi = x[i + 1]
        a = 1 / h**2 - np.sin(xi) / (2 * h)
        c = 1 / h**2 + np.sin(xi) / (2 * h)
        d = -2 / h**2 + np.exp(xi)

        if i > 0:
            A[i, i - 1] = a
        A[i, i] = d
        if i < n - 1:
            A[i, i + 1] = c

        b[i] = xi**2

    # 边界条件影响右端项
    b[0] -= (1 / h**2 - np.sin(x[1]) / (2 * h)) * 0    # y(0) = 0
    b[-1] -= (1 / h**2 + np.sin(x[-2]) / (2 * h)) * 3  # y(5) = 3

    y_inner = solve(A, b)
    y = np.zeros(n + 2)
    y[0] = 0
    y[-1] = 3
    y[1:-1] = y_inner

    return x, y


# ============================================================================ #
# 方法2：solve_bvp
# ============================================================================ #
def ode_system_for_solve_bvp(x, y):
    return np.vstack((y[1], -np.sin(x) * y[1] - np.exp(x) * y[0] + x**2))


def boundary_conditions_for_solve_bvp(ya, yb):
    return np.array([ya[0], yb[0] - 3])


def solve_bvp_scipy(n_initial_points=11):
    x_init = np.linspace(0, 5, n_initial_points)
    y_init = np.zeros((2, n_initial_points))

    solution = solve_bvp(ode_system_for_solve_bvp, boundary_conditions_for_solve_bvp,
                         x_init, y_init)

    if not solution.success:
        raise RuntimeError("solve_bvp failed to converge")

    return solution.x, solution.y[0]


# ============================================================================ #
# 主程序
# ============================================================================ #
if __name__ == "__main__":
    print("=" * 60)
    print("二阶常微分方程边值问题数值解法比较")
    print("方程：y''(x) + sin(x) * y'(x) + exp(x) * y(x) = x^2")
    print("边界条件：y(0) = 0, y(5) = 3")
    print("=" * 60)

    n_points = 50

    try:
        print("\n1. 有限差分法求解...")
        x_fd, y_fd = solve_bvp_finite_difference(n_points)
        print(f"   网格点数：{len(x_fd)}")
        print(f"   y(0) = {y_fd[0]:.6f}, y(5) = {y_fd[-1]:.6f}")
    except Exception as e:
        print(f"   有限差分法异常：{e}")
        x_fd, y_fd = None, None

    try:
        print("\n2. scipy.integrate.solve_bvp 求解...")
        x_scipy, y_scipy = solve_bvp_scipy()
        print(f"   网格点数：{len(x_scipy)}")
        print(f"   y(0) = {y_scipy[0]:.6f}, y(5) = {y_scipy[-1]:.6f}")
    except Exception as e:
        print(f"   solve_bvp 异常：{e}")
        x_scipy, y_scipy = None, None

    # 绘图比较
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    if x_fd is not None and y_fd is not None:
        plt.plot(x_fd, y_fd, 'b-o', markersize=3, label='Finite Difference Method', linewidth=2)
    if x_scipy is not None and y_scipy is not None:
        plt.plot(x_scipy, y_scipy, 'r--', label='scipy.solve_bvp', linewidth=2)

    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Comparison of Numerical Solutions for BVP')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    if (x_fd is not None and y_fd is not None and 
        x_scipy is not None and y_scipy is not None):
        y_interp = np.interp(x_fd, x_scipy, y_scipy)
        diff = np.abs(y_fd - y_interp)
        plt.semilogy(x_fd, diff, 'g-', linewidth=2, label='|FD - solve_bvp|')
        plt.xlabel('x')
        plt.ylabel('Absolute Difference')
        plt.title('Absolute Difference Between Methods')
        plt.legend()
        plt.grid(True, alpha=0.3)

        print(f"\n数值比较：")
        print(f"   最大绝对误差：{np.max(diff):.2e}")
        print(f"   平均绝对误差：{np.mean(diff):.2e}")
    else:
        plt.text(0.5, 0.5, 'Need both methods implemented\nfor comparison',
                 ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Difference Plot (Not Available)')

    plt.tight_layout()
    plt.show()

    print("=" * 60)
    print("实验完成！请分析两种方法的精度、效率与适用性。")
    print("=" * 60)
