import mujoco
import mujoco.viewer
import os
import time
# os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ['MUJOCO_GL'] = 'osmesa'
# 1. 路径处理：动态获取 world.xml 的位置
# 获取当前脚本 (visualize_scene.py) 所在的绝对路径
curr_dir = os.path.dirname(os.path.abspath(__file__))

# 根据层级结构推算：从 tests/ 向上跳一级到根目录，再进入 models/
# 结构：dual_arm_cooperation/tests/visualize_scene.py -> dual_arm_cooperation/models/world.xml
scene_path = os.path.join(curr_dir, "..", "models", "dual_panda_torque.xml")

# 规范化路径（去除 .. 等符号）
scene_path = os.path.normpath(scene_path)

def main():
    # 检查文件是否存在
    if not os.path.exists(scene_path):
        print(f"错误: 找不到文件 {scene_path}")
        print(f"当前尝试访问的路径为: {os.path.abspath(scene_path)}")
        return

    # 2. 加载 MuJoCo 模型
    try:
        # 加载时 MuJoCo 会根据 world.xml 内部的相对路径寻找 panda.xml
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 3. 启动交互式查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 将机器人和盒子重置到 XML 定义的初始位置
        mujoco.mj_resetData(model, data)
        
        print("查看器已启动，按 Ctrl+C 退出。")
        while viewer.is_running():
            step_start = time.time()

            # 运行物理仿真步
            mujoco.mj_step(model, data)

            # 同步渲染窗口
            viewer.sync()

            # 保持实时仿真频率 (1ms per step)
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
