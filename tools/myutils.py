import numpy as np

def normalize_quaternion(quaternion):
    """
    将四元数归一化为单位四元数。
    如果四元数的模接近零，则返回默认的单位四元数 [0, 0, 0, 1]。
    
    参数:
        quaternion: numpy数组或类似数组的对象，表示四元数 [qx, qy, qz, qw]
        _
    返回:
        归一化后的四元数，保持为 numpy 数组
    """
    quaternion = np.array(quaternion)  # 确保输入是numpy数组
    norm = np.linalg.norm(quaternion)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0])
    else:
        return quaternion / norm