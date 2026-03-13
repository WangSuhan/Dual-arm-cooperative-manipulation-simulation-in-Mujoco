import sys
import os
import time
import numpy as np
import cv2
import mujoco.viewer

# 将 src 目录加入环境变量
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from robot_interface import DualPandaInterface
from dynamics_model import DynamicsModel
from force_estimator import ForceEstimator
from impedance_controller import ImpedanceController
from trajectory_planner import TrajectoryPlanner


def main():
    print("=====================================================")
    print(" 🚀 启动双臂协同搬运 (混合控制版: 7轴力矩 + 1轴位置)")
    print("=====================================================")

    xml_path = "../models/dual_panda_torque.xml"
    robot = DualPandaInterface(xml_path)
    dyn = DynamicsModel(robot)

    # 按照论文公式(6)设置内力观测器和物体补偿 [cite: 291-299]
    estimator = ForceEstimator(dyn, m_obj=0.2)
    controller = ImpedanceController(dyn, estimator)
    planner = TrajectoryPlanner(dt=robot.model.opt.timestep)

    for _ in range(100):
        robot.step()

    # 【姿态锁定】：使用你确认正确的 90 度旋转矩阵
    VERTICAL_MAT_90 = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]
    ])

    base_obj_pos = np.array([0.0, 0.3, 0.42])
    offset_l = np.array([-0.15, 0.0, 0.0])
    offset_r = np.array([0.15, 0.0, 0.0])

    # 任务序列：对应论文中的搬运逻辑 [cite: 162, 266]
    task_sequence = [
        {"name": "1. 悬停 (Hover)", "duration": 2.0, "z_offset": 0.25, "x_offset": 0.0, "grip": 0,
         "S": [0, 0, 0, 0, 0, 0], "has_obj": False},
        {"name": "2. 下降 (Descend)", "duration": 1.5, "z_offset": 0.08, "x_offset": 0.0, "grip": 0,
         "S": [0, 0, 0, 0, 0, 0], "has_obj": False},
        {"name": "3. 加紧抓取 (Grasp)", "duration": 1.5, "z_offset": 0.08, "x_offset": 0.0, "grip": -2,
         "S": [0, 0, 0, 0, 0, 0], "has_obj": False},

        # 协同搬运阶段：主从臂配合 [cite: 312, 323-326]
        {"name": "4. 协同抬起 (Lift)", "duration": 1.5, "z_offset": 0.25, "x_offset": 0.0, "grip": -7,
         "S": [1, 1, 0, 0, 0, 0], "has_obj": True},
        {"name": "5. 协同平移 (Transport)", "duration": 3.0, "z_offset": 0.25, "x_offset": -0.20, "grip": -7,
         "S": [1, 1, 0, 0, 0, 0], "has_obj": True},
        {"name": "6. 协同下降 (Drop)", "duration": 1.5, "z_offset": 0.08, "x_offset": -0.20, "grip": -7,
         "S": [1, 1, 0, 0, 0, 0], "has_obj": True},

        {"name": "7. 松开 (Release)", "duration": 2.0, "z_offset": 0.08, "x_offset": -0.20, "grip": 5,
         "S": [0, 0, 0, 0, 0, 0], "has_obj": False},
        {"name": "8. 撤离 (Retreat)", "duration": 1.5, "z_offset": 0.25, "x_offset": -0.20, "grip": 5,
         "S": [0, 0, 0, 0, 0, 0], "has_obj": False}
    ]

    # ================= 视频录制：初始化 =================
    data_dir = os.path.join("..", "data")

    # 如果 data 文件夹不存在，则自动创建它 (exist_ok=True 避免已存在时报错)
    os.makedirs(data_dir, exist_ok=True)
    print(f"📁 数据将保存至: {os.path.abspath(data_dir)}")
    FPS = 30  # 设定视频帧率
    # 计算每隔多少个物理步截取一帧画面 (例如 dt=0.002s，则 skip 约等于 16)
    render_skip = max(1, int((1.0 / FPS) / planner.dt))

    # 初始化 MuJoCo 渲染器 (高度 720, 宽度 1280)
    renderer = mujoco.Renderer(robot.model, 720, 1280)

    video_path = os.path.join(data_dir, 'grasping_process.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(video_path, fourcc, FPS, (1280, 720))
    print("启动物理仿真...")
    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        # === 调整初始视角 (从 +X 轴看向 -X 轴) ===
        # 水平视角：通常加上或减去 180 度可以实现视角的完全对调。
        # 如果现在的默认视角是 0，可以改成 180；如果是 90，就改成 270，你可以试一下哪个角度最正。
        viewer.cam.azimuth = 270

        # 俯仰角：负数表示从上往下俯视。-20 到 -30 度通常是看桌面的最佳角度。
        viewer.cam.elevation = -35

        # 相机距离：控制画面的缩放，数值越大离得越远。
        viewer.cam.distance = 2.5

        # 画面中心点 (Lookat)：让相机镜头死死盯住盒子的位置
        viewer.cam.lookat[:] = [0.0, 0.3, 0.42]

        for task in task_sequence:
            if not viewer.is_running(): break
            print(f">>> 当前阶段: {task['name']}")
            steps = int(task['duration'] / planner.dt)

            curr_pos_l, _ = robot.get_ee_pose('left')
            curr_pos_r, _ = robot.get_ee_pose('right')
            target_center = base_obj_pos + np.array([task['x_offset'], 0.0, task['z_offset']])

            # 规划 4 参数轨迹 (含位置和固定姿态矩阵)
            traj_l = planner.plan_cooperative_trajectory(curr_pos_l, target_center + offset_l, task['duration'],
                                                         VERTICAL_MAT_90)
            traj_r = planner.plan_cooperative_trajectory(curr_pos_r, target_center + offset_r, task['duration'],
                                                         VERTICAL_MAT_90)
            safe_task_name = task['name'].replace(" ", "_")
            filename1 = f"traj_l_xyz_{safe_task_name}.txt"
            filename2 = f"traj_r_xyz_{safe_task_name}.txt"
            file_path1 = os.path.join(data_dir, filename1)
            file_path2 = os.path.join(data_dir, filename2)
            np.savetxt(file_path1, traj_l['pos'], fmt='%.6f', delimiter=',', header='x,y,z', comments='')
            np.savetxt(file_path2, traj_r['pos'], fmt='%.6f', delimiter=',', header='x,y,z', comments='')

            for i in range(steps):
                if not viewer.is_running(): break
                step_start = time.time()

                # 1. 读取 7 轴手臂状态
                curr_p_l, curr_m_l = robot.get_ee_pose('left')
                curr_p_r, curr_m_r = robot.get_ee_pose('right')
                current_obj_pos = (curr_p_l + curr_p_r) / 2.0 if task['has_obj'] else None

                # 2. 计算 7 轴控制力矩 (Torque Commands) [cite: 318, 325]
                # 主臂 (左)
                tau_l = controller.compute_master_torque(
                    'left', curr_p_l, curr_m_l, traj_l['pos'][i], VERTICAL_MAT_90, current_obj_pos
                )
                # 从臂 (右)
                dof_idx_r = robot._arm_joint_dof_adr['right']
                tau_measured_r = robot.data.qfrc_bias[dof_idx_r] + robot.data.qfrc_applied[dof_idx_r]
                tau_r = controller.compute_slave_torque(
                    'right', curr_p_r, curr_m_r, traj_r['pos'][i], VERTICAL_MAT_90,
                    tau_measured_r, current_obj_pos, task['S']
                )

                # 3. 分别下发指令
                # 下发 7 维力矩到电机的 ctrl[0:7]
                robot.set_joint_torques('left', tau_l)
                robot.set_joint_torques('right', tau_r)
                # print("torque",tau_l)

                # 下发 1 维位置指令到夹爪的 ctrl[7]
                # 因为 XML 里的 biasprm 很大，0 会强制合拢，255 会强制张开
                robot.set_gripper('left', task['grip'])
                robot.set_gripper('right', task['grip'])

                robot.step()
                viewer.sync()
                # ================= 视频录制：按帧率渲染并写入 =================
                if i % render_skip == 0:
                    # 更新渲染器场景。如果 XML 里没有配置特定相机，默认使用自由视角
                    renderer.camera = mujoco.MjvCamera()  # 创建一个新的虚拟相机对象

                    # 复制你在 viewer 中设置的参数到这里
                    renderer.camera.azimuth = 270  # 水平旋转
                    renderer.camera.elevation = -35  # 俯仰角
                    renderer.camera.distance = 2.5  # 距离
                    renderer.camera.lookat[:] = [0.0, 0.3, 0.42]  # 注视点

                    # 2. 更新场景 (传入 data，MuJoCo 会根据上面的 camera 设置渲染)
                    renderer.update_scene(robot.data, camera=renderer.camera)

                    # 3. 渲染像素
                    pixels = renderer.render()

                    # 4. 颜色转换与保存
                    frame = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                    video_out.write(frame)
                # ============================================================

                time_until_next = planner.dt - (time.time() - step_start)
                if time_until_next > 0: time.sleep(time_until_next)
        video_out.release()
        print("🎉 搬运任务完成！机械臂将锁定姿态维持悬停。")
        while viewer.is_running():
            step_start = time.time()
            curr_p_l, curr_m_l = robot.get_ee_pose('left')
            curr_p_r, curr_m_r = robot.get_ee_pose('right')
            # 持续发送阻抗力矩维持悬停 [cite: 317, 322]
            tau_l = controller.compute_master_torque('left', curr_p_l, curr_m_l, traj_l['pos'][-1], VERTICAL_MAT_90)
            tau_r = controller.compute_slave_torque('right', curr_p_r, curr_m_r, traj_r['pos'][-1], VERTICAL_MAT_90,
                                                    robot.data.qfrc_bias[robot._arm_joint_dof_adr['right']],
                                                    S_diag=[0, 0, 0, 0, 0, 0])
            robot.set_joint_torques('left', tau_l)
            robot.set_joint_torques('right', tau_r)
            # 维持夹爪状态
            robot.set_gripper('left', task_sequence[-1]['grip'])
            robot.set_gripper('right', task_sequence[-1]['grip'])

            robot.step()
            viewer.sync()
            time.sleep(planner.dt)


if __name__ == "__main__":
    main()