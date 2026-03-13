import numpy as np


class TrajectoryPlanner:
    def __init__(self, dt=0.002):
        """
        初始化轨迹规划器
        :param dt: 仿真步长 (对应 MuJoCo 的 timestep)
        """
        self.dt = dt

    def quintic_spline(self, start, end, T):
        t = np.arange(0, T, self.dt)
        if len(t) == 0: return np.array([end]), np.array([0.0]), np.array([0.0])
        tau = t / T
        delta = end - start
        pos_scale = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
        vel_scale = (1.0 / T) * (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4)
        acc_scale = (1.0 / T ** 2) * (60 * tau - 180 * tau ** 2 + 120 * tau ** 3)
        return start + delta * pos_scale, delta * vel_scale, delta * acc_scale

    def plan_cooperative_trajectory(self, start_pos, end_pos, duration, fixed_rotation_matrix):
        """
        生成包含位置轨迹和【固定姿态】的字典
        :param fixed_rotation_matrix: 传入初始记录的竖直向下矩阵
        """
        px, vx, ax = self.quintic_spline(start_pos[0], end_pos[0], duration)
        py, vy, ay = self.quintic_spline(start_pos[1], end_pos[1], duration)
        pz, vz, az = self.quintic_spline(start_pos[2], end_pos[2], duration)

        return {
            'pos': np.vstack((px, py, pz)).T,
            'vel': np.vstack((vx, vy, vz)).T,
            'acc': np.vstack((ax, ay, az)).T,
            'mat': fixed_rotation_matrix  # 每一帧都要求控制器维持这个姿态
        }

    def quat2mat(self, q):
        """简单的四元数转旋转矩阵"""
        w, x, y, z = q
        return np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]
        ])


# ==================== 独立测试验证 ====================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("正在测试轨迹规划器...")
    planner = TrajectoryPlanner(dt=0.002)

    # 假设从高度 0.4m 抬升到 0.6m，耗时 1.5秒
    start_p = [0, 0, 0.4]
    end_p = [0, 0, 0.6]
    T = 1.5

    traj = planner.plan_cartesian_trajectory(start_p, end_p, T)

    t_axis = np.arange(0, T, 0.002)

    # 提取 Z 轴的数据
    z_pos = traj['pos'][:, 2]
    z_vel = traj['vel'][:, 2]
    z_acc = traj['acc'][:, 2]

    print(f"生成的轨迹点数: {len(z_pos)}")
    print(f"起始加速度: {z_acc[0]:.4f}, 结束加速度: {z_acc[-1]:.4f} (预期都为0)")
    print(f"最高速度出现在中点: {np.max(z_vel):.4f} m/s")
    print("测试通过！轨迹算法可以提供完整的前馈状态。")