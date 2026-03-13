import numpy as np


class ImpedanceController:
    def __init__(self, dynamics_model, force_estimator):
        """
        初始化双臂阻抗控制器
        对应 TUM 论文 Section V-D-3: Cooperative Manipulation Control
        """
        self.dyn = dynamics_model
        self.estimator = force_estimator

        # --- 阻抗控制参数 (相当于虚拟弹簧和阻尼器的系数) ---
        # 笛卡尔空间位置刚度 K_p (前3个平移N/m，后3个旋转Nm/rad)
        self.K_p = np.diag([2000, 2000, 2000, 100, 100, 100])
        # 笛卡尔空间速度阻尼 K_d (通常设为临界阻尼 2 * sqrt(K_p * Mass))
        self.K_d = np.diag([100, 100, 100, 10, 10, 10])

        # 零空间阻尼系数 k_n (抑制手肘的无意义晃动)
        self.k_n = 20.0

        # --- 力控制参数 (用于从臂) ---
        # 比例系数 K_p^f 和积分系数 K_i^f
        self.K_p_f = np.diag([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])

        # 积分器状态缓存
        self.force_error_integral = np.zeros(6)

    def _compute_common_terms(self, arm, current_pos, current_mat, target_pos, target_mat, obj_pos, delta=0.5):
        """提取主从臂计算中都需要的公共数学项"""
        # 1. 运动学状态
        q, dq = self.dyn.robot.get_joint_states(arm)
        J, J_pinv = self.dyn.get_jacobian_and_pinv(arm)
        N = self.dyn.get_nullspace_projector(J, J_pinv)

        # 2. 笛卡尔误差 (e_p 和 \dot{e}_p)
        err_p = self.dyn.compute_cartesian_error(current_pos, current_mat, target_pos, target_mat)

        # 实际末端速度 (6x1) = J * dq
        current_vel = J @ dq
        err_v = np.zeros(6) - current_vel  # 假设目标速度为0的准静态跟踪

        # 3. 基础动力学补偿 tau_robot
        tau_robot = self.dyn.compute_tau_robot(arm)

        # 4. 物体重量前馈 tau_obj
        tau_obj = np.zeros(7)
        if obj_pos is not None:
            F_obj = np.zeros(6)
            F_obj[0:3] = -self.estimator.m_obj * self.estimator.gravity
            G_inv = self.estimator.compute_grasp_matrix(current_pos, obj_pos)
            # 映射到关节力矩: J^T * (delta * G^-1 * F_obj)
            tau_obj = J.T @ (delta * G_inv @ F_obj)

        return q, dq, J, J_pinv, N, err_p, err_v, tau_robot, tau_obj

    def compute_master_torque(self, arm, current_pos, current_mat, target_pos, target_mat, obj_pos=None):
        """
        计算主臂力矩 (Master Control) - 对应论文公式 (7)
        目标：极致的刚性，完美跟踪轨迹
        """
        q, dq, J, J_pinv, N, err_p, err_v, tau_robot, tau_obj = self._compute_common_terms(
            arm, current_pos, current_mat, target_pos, target_mat, obj_pos, delta=0.5
        )

        # 1. 任务空间阻抗力矩: J^T * (K_p * e_p + K_d * e_v)
        F_impedance = self.K_p @ err_p + self.K_d @ err_v
        tau_task = J.T @ F_impedance

        # 2. 零空间耗散力矩: N * (-k_n * dq)
        tau_null = N @ (-self.k_n * dq)

        # 3. 最终叠加 (公式7)
        tau_master = tau_robot + tau_obj + tau_task + tau_null
        return tau_master

    def compute_slave_torque(self, arm, current_pos, current_mat, target_pos, target_mat,
                             tau_measured, obj_pos=None, S_diag=[1, 1, 0, 0, 0, 0]):
        """
        计算从臂力矩 (Slave Control) - 对应论文公式 (8)
        目标：在选择矩阵 S 指定的方向上实现柔顺的力控制，卸载内力
        """
        q, dq, J, J_pinv, N, err_p, err_v, tau_robot, tau_obj = self._compute_common_terms(
            arm, current_pos, current_mat, target_pos, target_mat, obj_pos, delta=0.5
        )

        # 1. 定义选择矩阵 S (对角阵，1表示该方向开启力控制，0表示保持位置控制)
        # 论文中提到对于搬运箱子，选择水平面 x 和 y 方向卸载内力
        S = np.diag(S_diag)
        I = np.eye(6)

        # 2. 内力观测 e_f = F_desired - F_int
        F_int = self.estimator.estimate_internal_force(arm, tau_measured, obj_pos, delta=0.5)
        F_desired = np.zeros(6)  # 搬运时希望挤压力为 0
        err_f = F_desired - F_int

        # 3. 任务空间位置力矩 (注意：根据论文，PD项保持，但积分项被 (I-S) 过滤)
        # 为了代码简洁并防止积分饱和，我们这里仅使用 PD 控制位置
        F_pos_pd = self.K_p @ err_p + self.K_d @ err_v
        tau_pos = J.T @ F_pos_pd

        # 4. 任务空间力矩控制: J^T * S * (K_p^f * e_f)
        F_force_pi = self.K_p_f @ err_f
        tau_force = J.T @ S @ F_force_pi

        # 5. 零空间耗散
        tau_null = N @ (-self.k_n * dq)

        # 6. 最终叠加 (公式8)
        tau_slave = tau_robot + tau_obj + tau_pos + tau_null + tau_force
        return tau_slave


# ==================== 独立测试验证：主从柔顺性对比 ====================
if __name__ == "__main__":
    from robot_interface import DualPandaInterface
    from dynamics_model import DynamicsModel
    from force_estimator import ForceEstimator
    import mujoco.viewer
    import time

    robot = DualPandaInterface("../models/dual_panda_torque.xml")
    dyn = DynamicsModel(robot)
    estimator = ForceEstimator(dyn)
    controller = ImpedanceController(dyn, estimator)

    # 预热并记录双手初始姿态作为锁定的目标
    for _ in range(50):
        robot.step()
    target_pos_l, target_mat_l = robot.get_ee_pose('left')
    target_pos_r, target_mat_r = robot.get_ee_pose('right')

    print("=====================================================")
    print(" 测试启动：[主从阻抗特性对比]")
    print(" 1. 【左臂】是 Master，刚度极高！你用鼠标拉它，它会像弹簧一样强力弹回原位。")
    print(" 2. 【右臂】是 Slave，开启了 X、Y 轴的内力泄放 (力控制)。")
    print("    你用力拉右臂的 X、Y 轴，它会“顺从”你的力量，放弃抵抗！但在 Z 轴依然坚挺。")
    print(" 3. 提示：按住 Ctrl + 鼠标左键/右键 进行拖拽体验。")
    print("=====================================================")

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # 读取当前状态
            curr_pos_l, curr_mat_l = robot.get_ee_pose('left')
            curr_pos_r, curr_mat_r = robot.get_ee_pose('right')

            # --- 左臂 (Master): 铁面无私的轨迹追踪者 ---
            tau_left = controller.compute_master_torque(
                'left', curr_pos_l, curr_mat_l, target_pos_l, target_mat_l
            )
            robot.set_joint_torques('left', tau_left)

            # --- 右臂 (Slave): 获取测量力矩并计算从臂指令 ---
            # 真实情况下这里读电机传感器，仿真中我们读受到的扰动
            dof_idx_r = robot._arm_joint_dof_adr['right']
            tau_measured_r = robot.data.qfrc_bias[dof_idx_r] + robot.data.qfrc_applied[dof_idx_r]

            # 在 X(索引0) 和 Y(索引1) 方向开启力控制，其它方向保持刚性位置控制
            tau_right = controller.compute_slave_torque(
                'right', curr_pos_r, curr_mat_r, target_pos_r, target_mat_r,
                tau_measured=tau_measured_r, S_diag=[1, 1, 0, 0, 0, 0]
            )
            robot.set_joint_torques('right', tau_right)

            robot.step()
            viewer.sync()

            time_until_next = robot.model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)