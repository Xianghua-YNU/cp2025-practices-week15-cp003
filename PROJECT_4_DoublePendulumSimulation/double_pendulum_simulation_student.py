"""
模块：双摆模拟解决方案
作者：由Trae助手生成
描述：完整的双摆动力学模拟解决方案，包括能量计算和可选动画功能
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.animation as animation

# 物理常数
G_CONST = 9.81  # 重力加速度(m/s²)
L_CONST = 0.4   # 摆臂长度(m)
M_CONST = 1.0   # 摆锤质量(kg)

def derivatives(y, t, L1, L2, m1, m2, g_param):
    """
    计算双摆系统状态变量的时间导数(运动方程)
    
    参数:
        y (list/np.array): 当前状态向量[theta1, omega1, theta2, omega2]
                          theta1 - 第一个摆的角度(弧度)
                          omega1 - 第一个摆的角速度(弧度/秒)
                          theta2 - 第二个摆的角度(弧度) 
                          omega2 - 第二个摆的角速度(弧度/秒)
        t (float): 当前时间(未直接使用)
        L1 (float): 第一个摆臂长度
        L2 (float): 第二个摆臂长度
        m1 (float): 第一个摆锤质量
        m2 (float): 第二个摆锤质量
        g_param (float): 重力加速度

    返回:
        list: 时间导数[dtheta1/dt, domega1/dt, dtheta2/dt, domega2/dt]
    
    注意:
        这里实现的方程针对L1=L2=L和m1=m2=m的简化情况
        更一般情况需要不同的运动方程
    """
    theta1, omega1, theta2, omega2 = y

    # 角速度(一阶导数)
    dtheta1_dt = omega1
    dtheta2_dt = omega2

    # 计算角加速度(二阶导数)
    delta = theta1 - theta2  # 两摆角度差
    
    # 公共分母项
    den = 3 - np.cos(2*delta)
    
    # 第一个摆的角加速度方程
    num1 = (-omega1**2 * np.sin(2*delta) 
           - 2 * omega2**2 * np.sin(delta) 
           - (g_param/L1) * (np.sin(theta1 - 2*theta2) + 3*np.sin(theta1)))
    domega1_dt = num1 / den

    # 第二个摆的角加速度方程
    num2 = (4 * omega1**2 * np.sin(delta) 
           + omega2**2 * np.sin(2*delta) 
           + 2 * (g_param/L1) * (np.sin(2*theta1 - theta2) - np.sin(theta2)))
    domega2_dt = num2 / den
    
    return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]

def solve_double_pendulum(initial_conditions, t_span, t_points, L_param=L_CONST, g_param=G_CONST):
    """
    使用数值积分求解双摆运动微分方程
    
    参数:
        initial_conditions (dict): 初始条件字典:
            {'theta1': 值, 'omega1': 值, 'theta2': 值, 'omega2': 值} (弧度和弧度/秒)
        t_span (tuple): 模拟时间范围(t_start, t_end)(秒)
        t_points (int): 生成的时间点数
        L_param (float): 摆臂长度(默认L_CONST)
        g_param (float): 重力加速度(默认G_CONST)

    返回:
        tuple: (t_arr, sol_arr)
               t_arr: 时间点数组
               sol_arr: 状态数组[theta1, omega1, theta2, omega2]
    """
    # 从字典提取初始条件
    y0 = [initial_conditions['theta1'], initial_conditions['omega1'],
          initial_conditions['theta2'], initial_conditions['omega2']]
    
    # 创建时间数组
    t_arr = np.linspace(t_span[0], t_span[1], t_points)
    
    # 使用odeint求解ODE系统
    # 使用L_param作为摆长，M_CONST作为质量(等长等质量情况)
    sol_arr = odeint(derivatives, y0, t_arr,
                     args=(L_param, L_param, M_CONST, M_CONST, g_param),
                     rtol=1e-9, atol=1e-9)
    return t_arr, sol_arr

def calculate_energy(sol_arr, L_param=L_CONST, m_param=M_CONST, g_param=G_CONST):
    """
    计算双摆系统在每个时间点的总能量(动能+势能)
    
    参数:
        sol_arr (np.array): odeint的解数组(形状(n_points, 4))
        L_param (float): 摆臂长度
        m_param (float): 摆锤质量
        g_param (float): 重力加速度

    返回:
        np.array: 每个时间点的总能量数组
    """
    # 从解数组提取角度和角速度
    theta1 = sol_arr[:, 0]
    omega1 = sol_arr[:, 1]
    theta2 = sol_arr[:, 2]
    omega2 = sol_arr[:, 3]

    # 势能计算(V = -mgL(2cosθ₁ + cosθ₂))
    V = -m_param * g_param * L_param * (2 * np.cos(theta1) + np.cos(theta2))

    # 动能计算(来自拉格朗日量推导)
    T = m_param * L_param**2 * (omega1**2 + 0.5 * omega2**2 + omega1 * omega2 * np.cos(theta1 - theta2))
    
    return T + V

def animate_double_pendulum(t_arr, sol_arr, L_param=L_CONST, skip_frames=10):
    """
    创建双摆运动动画
    
    参数:
        t_arr (np.array): 时间数组
        sol_arr (np.array): odeint的解数组
        L_param (float): 摆臂长度
        skip_frames (int): 动画帧之间的间隔数

    返回:
        matplotlib.animation.FuncAnimation: 动画对象
    """
    # 从解数组提取角度数据(按间隔跳过)
    theta1_all = sol_arr[:, 0]
    theta2_all = sol_arr[:, 2]
    theta1_anim = theta1_all[::skip_frames]
    theta2_anim = theta2_all[::skip_frames]
    t_anim = t_arr[::skip_frames]

    # 将极坐标转换为笛卡尔坐标用于绘图
    x1 = L_param * np.sin(theta1_anim)  # 第一个摆的x位置
    y1 = -L_param * np.cos(theta1_anim)  # 第一个摆的y位置
    x2 = x1 + L_param * np.sin(theta2_anim)  # 第二个摆的x位置
    y2 = y1 - L_param * np.cos(theta2_anim)  # 第二个摆的y位置

    # 设置图形和坐标轴
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, autoscale_on=False,
                        xlim=(-2*L_param - 0.1, 2*L_param + 0.1),
                        ylim=(-2*L_param - 0.1, 0.1))
    ax.set_aspect('equal')  # 等比例坐标轴
    ax.grid()
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Double Pendulum Animation')

    # 创建摆线和时间文本对象
    line, = ax.plot([], [], 'o-', lw=2, markersize=8, color='blue')
    time_template ='Time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        """初始化动画数据"""
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        """动画更新函数"""
        # 更新摆的位置
        thisx = [0, x1[i], x2[i]]  # x坐标: [固定点, 摆1, 摆2]
        thisy = [0, y1[i], y2[i]]  # y坐标: [固定点, 摆1, 摆2]
        line.set_data(thisx, thisy)
        
        # 更新时间显示
        time_text.set_text(time_template % t_anim[i])
        return line, time_text

    # 创建动画对象
    ani = animation.FuncAnimation(fig, animate, frames=len(t_anim),
                                interval=25, blit=True, init_func=init)
    return ani

if __name__ == "__main__":
    # 主程序 - 直接运行脚本时执行
    
    # 设置初始条件(两个摆都从90度静止开始)
    initial_conditions_rad = {
        'theta1': np.pi/2,  # 90度
        'omega1': 0.0,
        'theta2': np.pi/2,  # 90度
        'omega2': 0.0
    }
    
    # 模拟参数
    t_start = 0  # 开始时间(s)
    t_end = 100  # 结束时间(s)
    t_points_sim = 2000  # 时间点数
    
    # 1. 求解双摆运动微分方程
    print(f"求解双摆运动方程，时间范围 {t_start}s 到 {t_end}s...")
    t_solution, sol_solution = solve_double_pendulum(initial_conditions_rad,
                                                   (t_start, t_end),
                                                   t_points_sim)
    print("方程求解完成")

    # 2. 计算系统能量
    print("计算系统能量...")
    energy_solution = calculate_energy(sol_solution)
    print("能量计算完成")

    # 3. 绘制能量-时间图并检查能量守恒
    plt.figure(figsize=(10, 5))
    plt.plot(t_solution, energy_solution, label='Total Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (Joules)')
    plt.title('Total Energy of the Double Pendulum vs. Time')
    plt.grid(True)
    plt.legend()
    
    # 计算能量守恒指标
    initial_energy = energy_solution[0]
    final_energy = energy_solution[-1]
    energy_variation = np.max(energy_solution) - np.min(energy_solution)
    
    print(f"初始能量: {initial_energy:.7f} J")
    print(f"最终能量: {final_energy:.7f} J")
    print(f"最大能量变化: {energy_variation:.7e} J")
    
    if energy_variation < 1e-5:
        print("能量守恒目标(< 1e-5 J)达成")
    else:
        print(f"能量守恒目标(< 1e-5 J)未达成，变化量: {energy_variation:.2e} J")
    
    # 根据能量变化设置y轴范围
    plt.ylim(initial_energy - 5*energy_variation if energy_variation > 1e-7 else initial_energy - 1e-5,
             initial_energy + 5*energy_variation if energy_variation > 1e-7 else initial_energy + 1e-5)
    plt.show()

    # 4. 可选的双摆运动动画
    run_animation = True  # 设为False可跳过动画
    if run_animation:
        print("创建动画中...可能需要一些时间")
        # 创建动画，使用skip_frames控制速度
        anim_object = animate_double_pendulum(t_solution, sol_solution,
                                             skip_frames=max(1, t_points_sim // 1000) * 5)
        plt.show()
        print("动画显示完成")
    else:
        print("跳过动画")

    print("双摆模拟完成")
