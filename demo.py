import os
from collections import deque
from dataclasses import dataclass

from utils.image_tools import convert_to_uint8, resize_with_pad
from utils.websocket_client_policy import WebsocketClientPolicy
from utils.misc import _quat2axisangle, convert_numpy_to_video

import numpy as np
np.random.seed(42)

DUMMY_ACTION = [0.0] * 6 + [-1.0]
ENV_RESOLUTION = 256  # resolution used to render training data

@dataclass
class Args:
    host: str = "你的服务器地址"   # host address
    port: int = 8000             # host port
    
    resize_size: int = 224       
    
    replan_steps: int = 5
    max_steps: int = 100        
    num_steps_wait: int = 10
    
    video_out_path: str = "./output/videos"  # Path to save videos

def get_env():
    # create environment instance
    import robosuite as suite
    env = suite.make(
        env_name="Lift",  # try with other tasks like "Stack" and "Door"
        robots="Panda",   # try with other robots like "Sawyer" and "Jaco" UR5e
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        render_camera="agentview",
        camera_names=["frontview", "robot0_eye_in_hand"],
    )
    # Reset environment MUST
    env.reset()
    
    # 获取不同视野的相机ID
    # {'frontview': 0, 'birdview': 1, 'agentview': 2, 'sideview': 3, 'robot0_robotview': 4, 'robot0_eye_in_hand': 5}
    camera_id = env.sim.model.camera_name2id("agentview")
    camera_id2 = env.sim.model.camera_name2id("robot0_robotview")

    # 调整相机位置（x, y, z）和视角（四元数）
    env.sim.model.cam_pos[camera_id] = np.array([1, 0, 1.7])  # 示例位置
    # env.sim.model.cam_quat[camera_id] =                     # 示例朝向（四元数）
    print(env.sim.model.cam_quat[camera_id])
    print(env.sim.model.cam_quat[camera_id2])
    print(env.sim.model.cam_pos[camera_id])
    print(env.sim.model.cam_pos[camera_id2])

    return env


def eval_my_env(args: Args):
    client = WebsocketClientPolicy(args.host, args.port)

    # Initialize environment and task description
    env = get_env()
    task = input("输入任务指令：")

    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
    # and we need to wait for them to fall
    for _ in range(args.num_steps_wait):
        obs, reward, done, info = env.step(DUMMY_ACTION)
        
    replay_images = []
    action_deque = deque()
    for _ in range(args.max_steps):
        try:
            # Get preprocessed image
            ## IMPORTANT: rotate 180 degrees to match train preprocessing
            img = np.ascontiguousarray(obs["frontview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = convert_to_uint8(resize_with_pad(img, args.resize_size, args.resize_size))
            wrist_img = convert_to_uint8(resize_with_pad(wrist_img, args.resize_size, args.resize_size))

            if len(action_deque) == 0:  #
                # Finished executing previous action chunk -- compute new chunk
                element = { # Prepare observations dict
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"][:2],
                        )
                    ),
                    "prompt": str(task),
                }

                # Query model to get action
                action_chunk = client.infer(element)["actions"]
                assert (
                    len(action_chunk) >= args.replan_steps
                ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                
                action_deque.extend(action_chunk[: args.replan_steps])

            # Execute action in environment
            action = action_deque.popleft()
            obs, reward, done, info = env.step(action.tolist())
            
            print("action: ", action)
            # Save preprocessed image for replay video
            replay_images.append(img)


        except Exception as e:
            print(f"[error]: Caught exception: {e}")
            break
    
    
    os.makedirs(args.video_out_path, exist_ok=True)
    convert_numpy_to_video(replay_images, args.video_out_path + f"/task.mp4")


if __name__ == "__main__":
    # Build Args with params
    args = Args()

    eval_my_env(args)
