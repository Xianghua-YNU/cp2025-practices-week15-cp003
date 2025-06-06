#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：打靶法与scipy.solve_bvp求解边值问题 - 学生代码模板

本项目要求实现打靶法和scipy.solve_bvp两种方法来求解二阶线性常微分方程边值问题：
u''(x) = -π(u(x)+1)/4
边界条件：u(0) = 1, u(1) = 1

学生姓名：[杨梅婷]
学号：[20231050157]
完成日期：[2025.6.4]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp, solve_bvp
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')


def ode_system_shooting(t, y):
    """
    Define the ODE system for shooting method.
    
    Convert the second-order ODE u'' = -π(u+1)/4 into a first-order system:
    y1 = u, y2 = u'
    y1' = y2
    y2' = -π(y1+1)/4
    
    Args:
        t (float): Independent variable (time/position)
        y (array): State vector [y1, y2] where y1=u, y2=u'
    
    Returns:
        list: Derivatives [y1', y2']
    """
    return [y[1], -np.pi * (y[0] + 1) / 4]


def boundary_conditions_scipy(ya, yb):
    """
    Define boundary conditions for scipy.solve_bvp.
    
    Boundary conditions: u(0) = 1, u(1) = 1
    ya[0] should equal 1, yb[0] should equal 1
    
    Args:
        ya (array): Values at left boundary [u(0), u'(0)]
        yb (array): Values at right boundary [u(1), u'(1)]
    
    Returns:
        array: Boundary condition residuals
    """
    return np.array([ya[0] - 1, yb[0] - 1])


def ode_system_scipy(x, y):
    """
    Define the ODE system for scipy.solve_bvp.
    
    Note: scipy.solve_bvp uses (x, y) parameter order, different from odeint
    
    Args:
        x (float): Independent variable
        y (array): State vector [y1, y2]
    
    Returns:
        array: Derivatives as column vector
    """
    return np.vstack((y[1], -np.pi * (y[0] + 1) / 4))


def solve_bvp_shooting_method(x_span, boundary_conditions, n_points=100, max_iterations=10, tolerance=1e-6):
    """
    Solve boundary value problem using shooting method.
    
    Algorithm:
    1. Guess initial slope m1
    2. Solve IVP with initial conditions [u(0), m1]
    3. Check if u(1) matches boundary condition
    4. If not, adjust slope using secant method and repeat
    
    Args:
        x_span (tuple): Domain (x_start, x_end)
        boundary_conditions (tuple): (u_left, u_right)
        n_points (int): Number of discretization points
        max_iterations (int): Maximum iterations for shooting
        tolerance (float): Convergence tolerance
    
    Returns:
        tuple: (x_array, y_array) solution arrays
    """
    # 验证输入参数
    if x_span[0] >= x_span[1]:
        raise ValueError("x_span must have x_start < x_end")
    
    # 添加对n_points的验证
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    
    if not isinstance(boundary_conditions, tuple) or len(boundary_conditions) != 2:
        raise TypeError("boundary_conditions must be a tuple of (left_value, right_value)")
    
    u_left, u_right = boundary_conditions
    x_eval = np.linspace(x_span[0], x_span[1], n_points)
    
    # 初始斜率猜测
    m0 = 0.0
    m1 = 1.0
    
    # 首次尝试
    y0_0 = np.array([u_left, m0])
    sol0 = solve_ivp(
        ode_system_shooting, 
        x_span, 
        y0_0, 
        t_eval=x_eval
    )
    u1_0 = sol0.y[0, -1]
    f0 = u1_0 - u_right
    
    # 第二次尝试
    for i in range(max_iterations):
        y0_1 = np.array([u_left, m1])
        sol1 = solve_ivp(
            ode_system_shooting, 
            x_span, 
            y0_1, 
            t_eval=x_eval
        )
        u1_1 = sol1.y[0, -1]
        f1 = u1_1 - u_right
        
        # 检查收敛性
        if abs(f1) < tolerance:
            return sol1.t, sol1.y[0]
        
        # 割线法更新斜率
        # 添加防止除零的微小值
        m2 = m1 - f1 * (m1 - m0) / (f1 - f0 + 1e-15)
        
        # 更新迭代变量
        m0, m1, f0 = m1, m2, f1
    
    # 迭代次数达上限时返回最后一次尝试的结果
    return sol1.t, sol1.y[0]


def solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points=50):
    """
    Solve boundary value problem using scipy.solve_bvp.
    
    Args:
        x_span (tuple): Domain (x_start, x_end)
        boundary_conditions (tuple): (u_left, u_right)
        n_points (int): Number of initial mesh points
    
    Returns:
        tuple: (x_array, y_array) solution arrays
    """
    # 设置初始网格和初始猜测
    x = np.linspace(x_span[0], x_span[1], n_points)
    
    # 使用满足边界条件的线性函数作为初始猜测
    u_left, u_right = boundary_conditions
    u_guess = u_left + (u_right - u_left) * (x - x_span[0]) / (x_span[1] - x_span[0])
    y_guess = np.vstack((u_guess, np.zeros_like(u_guess)))
    
    # 使用更可靠的solve_bvp求解器
    sol = solve_bvp(
        ode_system_scipy, 
        boundary_conditions_scipy, 
        x, 
        y_guess,
        tol=1e-6,  # 容差设置
        max_nodes=10000  # 允许更大的网格尺寸
    )
    
    # 检查求解是否成功
    if not sol.success:
        # 改进错误处理，提供详细错误信息
        error_msg = f"SciPy solve_bvp failed: {sol.message}"
        if "Singular Jacobian" in sol.message:
            error_msg += ". Try better initial guess or different n_points."
        raise RuntimeError(error_msg)
    
    # 在更精细的网格上评估解
    x_fine = np.linspace(x_span[0], x_span[1], n_points)
    u_fine = sol.sol(x_fine)[0]
    
    return x_fine, u_fine


def compare_methods_and_plot(x_span=(0, 1), boundary_conditions=(1, 1), n_points=100):
    """
    Compare shooting method and scipy.solve_bvp, generate comparison plot.
    
    Args:
        x_span (tuple): Domain for the problem
        boundary_conditions (tuple): Boundary values (left, right)
        n_points (int): Number of points for plotting
    
    Returns:
        dict: Dictionary containing solutions and analysis
    """
    # 使用两种方法求解
    x_shooting, y_shooting = solve_bvp_shooting_method(
        x_span, 
        boundary_conditions, 
        n_points
    )
    
    try:
        x_scipy, y_scipy = solve_bvp_scipy_wrapper(
            x_span, 
            boundary_conditions, 
            n_points
        )
    except Exception as e:
        print(f"SciPy solver failed: {str(e)}")
        # 返回默认值
        x_scipy, y_scipy = np.array([]), np.array([])
    
    # 创建公共网格进行比较
    x_common = np.linspace(x_span[0], x_span[1], n_points)
    
    # 插值到公共网格
    y_shooting_interp = np.interp(x_common, x_shooting, y_shooting) if len(x_shooting) > 0 else np.zeros_like(x_common)
    
    if len(x_scipy) > 0:
        y_scipy_interp = np.interp(x_common, x_scipy, y_scipy)
    else:
        y_scipy_interp = np.zeros_like(x_common)
    
    # 计算差异
    diff = y_shooting_interp - y_scipy_interp
    max_difference = np.max(np.abs(diff)) if len(diff) > 0 else 0.0
    rms_difference = np.sqrt(np.mean(diff**2)) if len(diff) > 0 else 0.0
    
    # 创建比较图表
    plt.figure(figsize=(10, 8))
    
    # 解的比较
    plt.subplot(2, 1, 1)
    plt.plot(x_shooting, y_shooting, 'b-', label='Shooting Method')
    if len(x_scipy) > 0:
        plt.plot(x_scipy, y_scipy, 'r--', label='SciPy solve_bvp')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Solution Comparison')
    plt.legend()
    plt.grid(True)
    
    # 差异图
    plt.subplot(2, 1, 2)
    plt.plot(x_common, diff, 'g-')
    plt.xlabel('x')
    plt.ylabel('Difference')
    plt.title(f'Difference Between Methods\nMax: {max_difference:.2e}, RMS: {rms_difference:.2e}')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('method_comparison.png')
    plt.close()
    
    # 返回比较结果
    return {
        'x_shooting': x_shooting,
        'y_shooting': y_shooting,
        'x_scipy': x_scipy,
        'y_scipy': y_scipy,
        'max_difference': max_difference,
        'rms_difference': rms_difference
    }


# 测试函数
def test_ode_system():
    """Test ODE system implementation."""
    print("Testing ODE system...")
    try:
        # Test point
        t_test = 0.5
        y_test = np.array([1.0, 0.5])
        
        # Test shooting method ODE system
        dydt = ode_system_shooting(t_test, y_test)
        print(f"ODE system (shooting): dydt = {dydt}")
        
        # Test scipy ODE system
        dydt_scipy = ode_system_scipy(t_test, y_test)
        print(f"ODE system (scipy): dydt = {dydt_scipy}")
        
    except NotImplementedError:
        print("ODE system functions not yet implemented.")


def test_boundary_conditions():
    """Test boundary conditions implementation."""
    print("Testing boundary conditions...")
    try:
        ya = np.array([1.0, 0.5])  # Left boundary
        yb = np.array([1.0, -0.3])  # Right boundary
        
        bc_residual = boundary_conditions_scipy(ya, yb)
        print(f"Boundary condition residuals: {bc_residual}")
        
    except NotImplementedError:
        print("Boundary conditions function not yet implemented.")


if __name__ == "__main__":
    print("项目2：打靶法与scipy.solve_bvp求解边值问题")
    print("=" * 50)
    
    # Run basic tests
    test_ode_system()
    test_boundary_conditions()
    
    # Run comparison
    try:
        print("\nTesting method comparison...")
        results = compare_methods_and_plot()
        print("Comparison results:")
        print(f"Max difference: {results['max_difference']:.6e}")
        print(f"RMS difference: {results['rms_difference']:.6e}")
        print("Method comparison completed successfully!")
    except Exception as e:
        print(f"Method comparison failed: {str(e)}")
    
    print("\n项目已完成，请查看比较图表'method_comparison.png'")
