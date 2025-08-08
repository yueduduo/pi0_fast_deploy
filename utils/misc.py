
import math
import numpy as np
from moviepy import ImageSequenceClip


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
# 将replay_images list里有80个np.array里80帧图片转为视频，fps为10
def convert_numpy_to_video(frames, output_path, fps=10):
    """
    将NumPy数组帧列表转换为视频（不使用OpenCV）
    
    参数:
        frames: NumPy数组列表，每个数组代表一帧图像
        output_path: 输出视频路径
        fps: 视频帧率
    """
    if not frames:
        raise ValueError("帧列表不能为空")
    
    # 确保所有帧是uint8类型
    frames = [frame.astype(np.uint8) if frame.dtype != np.uint8 else frame for frame in frames]
    
    # 创建视频剪辑
    clip = ImageSequenceClip(frames, fps=fps)
    
    # 写入视频文件
    clip.write_videofile(output_path, codec="libx264")
    print(f"视频已保存至: {output_path}")
