import numpy as np


class ForceEstimator:
    def __init__(self, dynamics_model, m_obj=0.1):
        """
        初始化内力观测器
        对应 TUM 论文 Section V-D-2: Internal Object Force Estimation

        :param dynamics_model: DynamicsModel 实例
        :param m_obj: 被抓取物体的质量 (kg)，默认 0.1kg (我们在 XML 中定义的重量)
        """
        self.dyn = dynamics_model
        self.m_obj = m_obj
        self.gravity = np.array([0, 0, -9.81])  # 重力加速度向量

    def compute_grasp_matrix(self, tcp_pos, obj_pos):
        """
        计算抓取矩阵 G (Grasp Matrix) 的逆
        用于将物体质心的力映射到 TCP (工具中心点) 坐标系
        """
        # 计算从末端 TCP 指向物体质心(Center of Mass)的偏置向量
        r_p2k = obj_pos - tcp_pos

        # 构造反对称矩阵 (Skew-symmetric matrix)
        S = np.array([
            [0, -r_p2k[2], r_p2k[1]],
            [r_p2k[2], 0, -r_p2k[0]],
            [-r_p2k[1], r_p2k[0], 0]
        ])

        # 构造 6x6 抓取矩阵 G
        # G = [ I   0 ]
        #     [ S^T I ]
        G = np.eye(6)
        G[3:6, 0:3] = S.T

        # 返回 G 的逆矩阵
        return np.linalg.inv(G)

    def estimate_internal_force(self, arm, tau_measured, obj_pos=None, delta=0.5):
        """
        核心公式 (6): F_int = (J^T)^# * (tau_sensor - tau_robot) - delta * G^-1 * F_obj

        :param arm: 'left' 或 'right'
        :param tau_measured: 从底层读取的实际关节力矩 (7x1)
        :param obj_pos: 物体的三维坐标。如果为 None，则假设空载 (不补偿物体重量)
        :param delta: 负载分配系数，0.5 表示双手各承担一半的物体重量
        :return: F_int (6x1 向量，前3个是力 N，后3个是力矩 Nm)
        """
        # 1. 获取雅可比矩阵 J 及其伪逆 J_pinv
        J, J_pinv = self.dyn.get_jacobian_and_pinv(arm)

        # 数学恒等式：雅可比转置的伪逆 == 雅可比伪逆的转置
        # (J^T)^# = (J^#)^T
        JT_pinv = J_pinv.T

        # 2. 计算机器人自身的动力学补偿 (准静态下主要为重力和科氏力)
        # 对应公式 (4) 的静态部分
        tau_robot = self.dyn.compute_tau_robot(arm)

        # 3. 计算末端接触力 (尚未剔除物体重量)
        # 将关节空间的纯外力矩映射到笛卡尔空间
        F_ext_tcp = JT_pinv @ (tau_measured - tau_robot)

        # 4. 剔除物体惯性力 F_obj (对应公式 5)
        if obj_pos is not None:
            # 准静态假设：物体加速度为0，仅受重力
            F_obj = np.zeros(6)
            F_obj[0:3] = -self.m_obj * self.gravity  # [0, 0, 0.981] N

            # 姿态惯性角动量 (论文中提到设为0)
            # 详见论文: "we set Theta_obj = 0 as we only weighed our manipulated objects..."

            # 计算抓取矩阵的逆
            tcp_pos, _ = self.dyn.robot.get_ee_pose(arm)
            G_inv = self.compute_grasp_matrix(tcp_pos, obj_pos)

            # 5. 结算最终的内力
            F_int = F_ext_tcp - delta * (G_inv @ F_obj)
        else:
            # 空载状态，接触力即为内力(或外部碰撞力)
            F_int = F_ext_tcp

        return F_int


# ==================== 独立测试验证 ====================
if __name__ == "__main__":
    from robot_interface import DualPandaInterface
    from dynamics_model import DynamicsModel
    import mujoco.viewer
    import time

    print("正在初始化观测器...")
    robot = DualPandaInterface("../models/dual_panda_torque.xml")
    dyn = DynamicsModel(robot)
    estimator = ForceEstimator(dyn, m_obj=0.1)

    # 推进以获得稳定初值
    for _ in range(50):
        robot.step()

    print("\n--- 内力观测器测试 ---")
    print("在弹出的窗口中，请用鼠标左键用力拉扯左臂的夹爪！")
    print("终端会实时打印观测到的外部施加的三维作用力 (N)。")
    print("----------------------")

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            arm = 'left'
            # 1. 重力补偿控制 (保持手臂浮空)
            tau_robot = dyn.compute_tau_robot(arm)
            robot.set_joint_torques(arm, tau_robot)

            # 2. 从物理引擎读取真实的执行力矩 (包含你用鼠标拉扯产生的反作用力矩)
            # data.qfrc_applied 包含了用户鼠标施加的扰动
            # 在真实的机器人上，这里应该读取电机的 tau_sensor
            dof_indices = robot._arm_joint_dof_adr[arm]
            tau_measured = robot.data.qfrc_bias[dof_indices] + robot.data.qfrc_applied[dof_indices]

            # 3. 估计外力
            F_ext = estimator.estimate_internal_force(arm, tau_measured)

            # 每隔一段时间打印一次受力情况
            if int(time.time() * 10) % 5 == 0:
                # 仅打印前三个维度的力 (X, Y, Z)
                force_xyz = np.round(F_ext[0:3], 1)
                # 过滤掉数值噪点
                if np.linalg.norm(force_xyz) > 1.0:
                    print(f"受到外力扰动 -> X: {force_xyz[0]:.1f} N, Y: {force_xyz[1]:.1f} N, Z: {force_xyz[2]:.1f} N")

            robot.step()
            viewer.sync()

            time_until_next = robot.model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)