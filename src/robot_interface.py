import mujoco
import mujoco.viewer
import numpy as np
import time


class DualPandaInterface:
    def __init__(self, xml_path):
        """
        初始化 MuJoCo 模型和数据，建立基于力矩控制的双臂底层映射
        """
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.arms = ['left', 'right']
        self.num_arm_joints = 7

        # 缓存 ID 和地址索引
        self._arm_actuator_ids = {'left': [], 'right': []}
        self._arm_joint_qpos_adr = {'left': [], 'right': []}
        self._arm_joint_dof_adr = {'left': [], 'right': []}  # 用于读取速度和动力学矩阵
        self._gripper_actuator_id = {}
        self._ee_body_id = {}

        self._setup_mappings()
        self.reset_to_home()

    def _setup_mappings(self):
        """根据 XML 命名规范缓存 ID"""
        for arm in self.arms:
            for i in range(1, self.num_arm_joints + 1):
                joint_name = f"{arm}_joint{i}"
                actuator_name = f"{arm}_actuator{i}"

                jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                # qposadr 用于位置，dofadr 用于速度和受力
                self._arm_joint_qpos_adr[arm].append(self.model.jnt_qposadr[jnt_id])
                self._arm_joint_dof_adr[arm].append(self.model.jnt_dofadr[jnt_id])

                act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                self._arm_actuator_ids[arm].append(act_id)

            # 夹爪
            gripper_name = f"{arm}_actuator8"
            self._gripper_actuator_id[arm] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, gripper_name)

            # 末端执行器 Body
            ee_name = f"{arm}_hand"
            self._ee_body_id[arm] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ee_name)

    def reset_to_home(self):
        """重置到 XML 中定义的初始姿态"""
        keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home_left")
        if keyframe_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        """步进一帧仿真"""
        mujoco.mj_step(self.model, self.data)

    # ==================== 运动学状态读取 ====================

    def get_joint_states(self, arm):
        """获取指定机械臂的 7 个关节角度(q)和角速度(dq)"""
        q_adrs = self._arm_joint_qpos_adr[arm]
        dq_adrs = self._arm_joint_dof_adr[arm]

        q = np.array([self.data.qpos[adr] for adr in q_adrs])
        dq = np.array([self.data.qvel[adr] for adr in dq_adrs])
        return q, dq

    def get_ee_pose(self, arm):
        """
        获取带有 TCP (工具中心点) 偏移的末端位姿
        返回: tcp_pos (3,), rot_mat (3, 3)
        """
        body_id = self._ee_body_id[arm]
        wrist_pos = self.data.xpos[body_id].copy()
        rot_mat = self.data.xmat[body_id].copy().reshape(3, 3)

        # 将参考点从手腕法兰盘推至两指尖中间
        tcp_offset = np.array([0.0, 0.0, 0.105])
        tcp_pos = wrist_pos + rot_mat @ tcp_offset
        return tcp_pos, rot_mat

    # ==================== 动力学引擎 (核心升级) ====================

    def get_dynamics_parameters(self, arm):
        """
        获取机械臂当前的动力学参数：质量矩阵 M，以及科里奥利力+重力向量 cg
        返回:
            M_arm (7x7 矩阵): 关节空间质量矩阵
            cg_arm (7x1 向量): C(q, dq) + g(q)
        """
        dof_indices = self._arm_joint_dof_adr[arm]

        # 1. 提取质量矩阵 M(q)
        # MuJoCo 默认存储为一维压缩格式 data.qM，需转换成密集矩阵
        M_full = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M_full, self.data.qM)

        # 切片提取当前手臂对应的 7x7 矩阵
        M_arm = M_full[np.ix_(dof_indices, dof_indices)]

        # 2. 提取科里奥利力与重力补偿项 C(q, dq) + g(q)
        # qfrc_bias 内部正好包含了这两项的负作用力
        cg_arm = self.data.qfrc_bias[dof_indices].copy()

        return M_arm, cg_arm

    def get_tcp_jacobian(self, arm):
        """
        计算 TCP 点的 6x7 雅可比矩阵
        返回: J (6x7矩阵, 前3行为平移，后3行为旋转)
        """
        tcp_pos, _ = self.get_ee_pose(arm)
        body_id = self._ee_body_id[arm]

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        # 根据 TCP 空间坐标计算精确雅可比
        mujoco.mj_jac(self.model, self.data, jacp, jacr, tcp_pos, body_id)
        jac_full = np.vstack([jacp, jacr])

        # 提取当前手臂对应的 7 列
        dof_indices = self._arm_joint_dof_adr[arm]
        J_arm = jac_full[:, dof_indices]

        return J_arm

    # ==================== 指令发送 ====================

    def set_joint_torques(self, arm, torques):
        """
        向指定机械臂的纯力矩电机发送力矩指令 (Nm)
        """
        if len(torques) != self.num_arm_joints:
            raise ValueError(f"需要 {self.num_arm_joints} 个力矩值")

        act_ids = self._arm_actuator_ids[arm]
        for i, act_id in enumerate(act_ids):
            # 限幅保护（虽然 XML 里面有 ctrlrange，但代码层限幅更安全）
            ctrl_min = self.model.actuator_ctrlrange[act_id][0]
            ctrl_max = self.model.actuator_ctrlrange[act_id][1]
            safe_torque = np.clip(torques[i], ctrl_min, ctrl_max)

            self.data.ctrl[act_id] = safe_torque

    def set_gripper(self, arm, value):
        """"
    控制夹爪 (力控模式)
    :param arm: 'left' 或 'right'
    :param force_value: 目标夹持力 (单位: 牛顿 N)
                        通常负值是闭合，正值是张开 (需根据实际测试确认)
                        推荐范围: -30.0 (夹紧) 到 10.0 (微张)
    """
        act_id = self._gripper_actuator_id[arm]
        safe_force = np.clip(value, -40.0, 40.0)

        self.data.ctrl[act_id] = safe_force


# ==================== 独立测试：重力补偿 (Gravity Compensation) ====================
if __name__ == "__main__":
    XML_PATH = "../models/dual_panda_torque.xml"

    print("初始化动力学接口...")
    robot = DualPandaInterface(XML_PATH)

    print("=====================================================")
    print(" 测试启动：[零重力模式]")
    print(" 说明：")
    print(" 1. 由于我们使用的是纯力矩电机，如果不发送指令，手臂会软趴趴掉下来。")
    print(" 2. 在这个测试中，代码会实时读取机器人自身的重力，并将抵消重力的")
    print("    力矩反向发送给电机。")
    print(" 3. 请用鼠标左键拖拽机械臂（模拟用手推拉），你会感觉它像漂浮在太空中一样！")
    print("=====================================================")

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # --- 核心：重力与科里奥利力补偿 ---
            for arm in robot.arms:
                # 提取动力学参数 cg = C(q, dq) + g(q)
                _, cg_compensation = robot.get_dynamics_parameters(arm)

                # 直接将补偿力矩下发给电机
                robot.set_joint_torques(arm, cg_compensation)

                # 保持夹爪张开
                robot.set_gripper(arm, 255)

            robot.step()
            viewer.sync()

            time_until_next = robot.model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)