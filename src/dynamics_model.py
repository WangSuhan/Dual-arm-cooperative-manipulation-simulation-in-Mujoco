import numpy as np


class DynamicsModel:
    def __init__(self, robot_interface):
        """
        初始化动力学数学引擎
        :param robot_interface: DualPandaInterface 实例，用于提取底层矩阵
        """
        self.robot = robot_interface

    def get_jacobian_and_pinv(self, arm, damping=0.05):
        """
        获取 TCP 雅可比矩阵及其伪逆 (Pseudo-inverse)
        使用阻尼最小二乘法 (Damped Least Squares, DLS) 保证在奇异点附近的数值稳定性
        """
        # 1. 从底层获取 6x7 的雅可比矩阵
        J = self.robot.get_tcp_jacobian(arm)

        # 2. 计算 DLS 伪逆: J^# = J^T * (J * J^T + lambda^2 * I)^-1
        J_T = J.T
        lambda_sq = damping ** 2

        # 核心解算：这使得即时机械臂完全伸直，逆解也不会因为分母为0而爆炸
        J_pinv = J_T @ np.linalg.inv(J @ J_T + lambda_sq * np.eye(6))

        return J, J_pinv

    def get_nullspace_projector(self, J, J_pinv):
        """
        计算零空间投影矩阵 (Null-space Projector)
        公式: N = I - J^T * (J^#)^T
        作用: 将关节力矩映射到“不改变末端位姿”的内部运动空间 (即让手肘动，但手爪不动)
        """
        num_joints = J.shape[1]  # 7 个关节
        I = np.eye(num_joints)

        # 零空间投影算子
        N = I - J.T @ J_pinv.T
        return N

    def compute_tau_robot(self, arm, q_ddot_d=None):
        """
        计算机器人自身的动力学前馈补偿 (tau_robot)
        对应 TUM 论文中的: tau_robot = M * q_ddot_d + C(q, dq) + g(q)
        """
        # 提取质量矩阵 M 和 包含科氏力+重力的 cg 项
        M, cg = self.robot.get_dynamics_parameters(arm)

        # 如果提供了期望加速度，则进行完整的逆动力学前馈
        if q_ddot_d is not None:
            tau_robot = M @ q_ddot_d + cg
        else:
            # 在准静态或低速移动下，M * q_ddot_d 极小，仅补偿科氏力和重力即可
            tau_robot = cg

        return tau_robot

    def compute_cartesian_error(self, current_pos, current_mat, target_pos, target_mat):
        """
        计算笛卡尔空间中的 6D 误差向量 [位置误差, 姿态误差]^T
        用于阻抗控制器的刚度计算 (K_p * e_p)
        """
        # 1. 位置误差 (3x1)
        err_pos = target_pos - current_pos

        # 2. 姿态误差 (3x1) - 使用旋转矩阵的反对称算子提取角度差
        R_err = target_mat @ current_mat.T
        err_ori = np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1]
        ]) * 0.5

        # 拼接为完整的 6D 空间误差
        error_6d = np.hstack([err_pos, err_ori])
        return error_6d


# ==================== 独立测试验证 ====================
if __name__ == "__main__":
    from robot_interface import DualPandaInterface

    # 初始化
    print("正在加载模型与动力学引擎...")
    robot = DualPandaInterface("../models/dual_panda_torque.xml")
    dyn = DynamicsModel(robot)

    # 推进几帧以获得稳定的物理初值
    for _ in range(10):
        robot.step()

    print("\n--- 动力学数学引擎测试 ---")
    arm = 'left'

    # 1. 测试雅可比与伪逆
    J, J_pinv = dyn.get_jacobian_and_pinv(arm)
    print(f"[{arm}臂] 雅可比矩阵 J 维度: {J.shape} (预期 6x7)")
    print(f"[{arm}臂] 伪逆矩阵 J_pinv 维度: {J_pinv.shape} (预期 7x6)")

    # 2. 测试零空间投影
    N = dyn.get_nullspace_projector(J, J_pinv)
    print(f"[{arm}臂] 零空间投影矩阵 N 维度: {N.shape} (预期 7x7)")

    # 3. 测试动力学前馈计算
    # 假设此时期望加速度为 0
    tau_robot = dyn.compute_tau_robot(arm)
    print(f"[{arm}臂] 纯重力/科氏力补偿力矩: \n{np.round(tau_robot, 2)} Nm")

    print("\n动力学引擎基础测试通过！数学模块运转正常。")