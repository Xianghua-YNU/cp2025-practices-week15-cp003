import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 常量定义
G_CONST = 9.81  # 重力加速度 (m/s^2)
L_CONST = 0.4   # 每个摆臂的长度 (m)
M_CONST = 1.0   # 每个摆锤的质量 (kg)

def derivatives(y, t, L1, L2, m1, m2, g_param):
    """
    返回双摆状态向量 y 的时间导数。

    参数:
        y (list or np.array): 当前状态向量 [theta1, omega1, theta2, omega2]。
        t (float): 当前时间（在这些自治方程中未直接使用，但 odeint 需要）。
        L1 (float): 第一个摆臂的长度。
        L2 (float): 第二个摆臂的长度。
        m1 (float): 第一个摆锤的质量。
        m2 (float): 第二个摆锤的质量。
        g (float): 重力加速度。

    返回:
        list: 时间导数 [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]。
    """
    theta1, omega1, theta2, omega2 = y

    dtheta1_dt = omega1
    dtheta2_dt = omega2

    # domega1_dt 的分子和分母
    num1 = -omega1**2 * np.sin(2*theta1 - 2*theta2) \
           - 2 * omega2**2 * np.sin(theta1 - theta2) \
           - (g_param/L1) * (np.sin(theta1 - 2*theta2) + 3*np.sin(theta1))
    den1 = 3 - np.cos(2*theta1 - 2*theta2)
    domega1_dt = num1 / den1

    # domega2_dt 的分子和分母
    num2 = 4 * omega1**2 * np.sin(theta1 - theta2) \
           + omega2**2 * np.sin(2*theta1 - 2*theta2) \
           + 2 * (g_param/L1) * (np.sin(2*theta1 - theta2) - np.sin(theta2))
    den2 = 3 - np.cos(2*theta1 - 2*theta2)
    domega2_dt = num2 / den2
    
    return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]

def solve_double_pendulum(initial_conditions, t_span, t_points, L_param=L_CONST, g_param=G_CONST):
    """
    求解双摆的常微分方程。

    参数:
        initial_conditions (dict): {'theta1': val, 'omega1': val, 'theta2': val, 'omega2': val}，单位为弧度和弧度/秒。
        t_span (tuple): (t_start, t_end) 表示模拟的时间范围。
        t_points (int): 生成的时间点数。
        L_param (float): 摆臂长度。
        g_param (float): 重力加速度。

    返回:
        tuple: (t_arr, sol_arr)
               t_arr: 一维 numpy 数组，表示时间点。
               sol_arr: 二维 numpy 数组，包含每个时间点的状态 [theta1, omega1, theta2, omega2]。
    """
    y0 = [initial_conditions['theta1'], initial_conditions['omega1'], 
          initial_conditions['theta2'], initial_conditions['omega2']]
    t_arr = np.linspace(t_span[0], t_span[1], t_points)
    
    sol_arr = odeint(derivatives, y0, t_arr, args=(L_param, L_param, M_CONST, M_CONST, g_param), rtol=1e-9, atol=1e-9)
    return t_arr, sol_arr

def calculate_energy(sol_arr, L_param=L_CONST, m_param=M_CONST, g_param=G_CONST):
    """
    计算双摆系统的总能量。

    参数:
        sol_arr (np.array): odeint 的解数组（行表示时间点，列表示 [theta1, omega1, theta2, omega2]）。
        L_param (float): 摆臂长度。
        m_param (float): 摆锤质量。
        g_param (float): 重力加速度。

    返回:
        np.array: 一维数组，表示每个时间点的总能量。
    """
    theta1 = sol_arr[:, 0]
    omega1 = sol_arr[:, 1]
    theta2 = sol_arr[:, 2]
    omega2 = sol_arr[:, 3]

    # 势能 (V)
    V = -m_param * g_param * L_param * (2 * np.cos(theta1) + np.cos(theta2))

    # 动能 (T)
    T = m_param * L_param**2 * (omega1**2 + 0.5 * omega2**2 + omega1 * omega2 * np.cos(theta1 - theta2))
    
    return T + V

if __name__ == "__main__":
    # 初始条件
    initial_conditions_rad = {
        'theta1': np.pi/2,  # 90 度
        'omega1': 0.0,
        'theta2': np.pi/2,  # 90 度
        'omega2': 0.0
    }
    t_start = 0
    t_end = 100
    t_points_sim = 2000 

    # 1. 求解常微分方程
    print(f"Solving ODEs for t = {t_start}s to {t_end}s...")
    t_solution, sol_solution = solve_double_pendulum(initial_conditions_rad, (t_start, t_end), t_points_sim, L_param=L_CONST, g_param=G_CONST)
    print("ODE solving complete.")

    # 2. 计算能量
    print("Calculating energy...")
    energy_solution = calculate_energy(sol_solution, L_param=L_CONST, m_param=M_CONST, g_param=G_CONST)
    print("Energy calculation complete.")

    # 3. 绘制能量随时间的变化曲线
    plt.figure(figsize=(10, 5))
    plt.plot(t_solution, energy_solution, label='Total Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (Joules)')
    plt.title('Total Energy of the Double Pendulum vs. Time')
    plt.grid(True)
    plt.legend()

    # 检查能量守恒
    initial_energy = energy_solution[0]
    final_energy = energy_solution[-1]
    energy_variation = np.max(energy_solution) - np.min(energy_solution)
    print(f"Initial Energy: {initial_energy:.7f} J")
    print(f"Final Energy:   {final_energy:.7f} J")
    print(f"Max Energy Variation: {energy_variation:.7e} J")
    if energy_variation < 1e-5:
        print("Energy conservation target (< 1e-5 J) met.")
    else:
        print(f"Energy conservation target (< 1e-5 J) NOT met. Variation: {energy_variation:.2e} J. Try increasing t_points or tightening odeint tolerances.")
    plt.ylim(initial_energy - 5*energy_variation if energy_variation > 1e-7 else initial_energy - 1e-5, 
             initial_energy + 5*energy_variation if energy_variation > 1e-7 else initial_energy + 1e-5)
    plt.show()

    print("Double Pendulum Simulation finished.")
