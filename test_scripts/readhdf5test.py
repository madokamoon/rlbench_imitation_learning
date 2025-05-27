import h5py
import time
import os
import numpy as np

def test_read_speed(file_path, num_trials=3):
    """测试HDF5文件读取速度"""
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    # 测试完整读取
    full_read_times = []
    for _ in range(num_trials):
        start_time = time.time()
        with h5py.File(file_path, 'r') as f:
            # 读取所有数据集
            def read_all(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data = obj[...]
            
            f.visititems(read_all)
        
        full_read_times.append(time.time() - start_time)
    
    # 测试图像读取 (假设存在特定路径的图像数据)
    image_read_times = []
    try:
        with h5py.File(file_path, 'r') as f:
            if 'observations' in f and 'images' in f['observations']:
                # 找到第一个摄像头
                cam_names = list(f['observations']['images'].keys())
                if cam_names:
                    cam_name = cam_names[0]
                    img_dataset = f['observations']['images'][cam_name]
                    total_frames = img_dataset.shape[0]
                    
                    # 选择10帧或全部帧
                    frames_to_read = min(10, total_frames)
                    indices = list(range(0, total_frames, max(1, total_frames // frames_to_read)))[:frames_to_read]
                    
                    # 测试读取这些帧
                    for _ in range(num_trials):
                        start_time = time.time()
                        with h5py.File(file_path, 'r') as f2:
                            for idx in indices:
                                img = f2['observations']['images'][cam_name][idx]
                        image_read_times.append(time.time() - start_time)
    except Exception as e:
        print(f"图像读取测试失败: {e}")
    
    # 计算平均值
    avg_full_read_time = np.mean(full_read_times)
    avg_image_read_time = np.mean(image_read_times) if image_read_times else 0
    
    return {
        'file_path': file_path,
        'file_size_mb': file_size_mb,
        'full_read_time': avg_full_read_time,
        'image_read_time': avg_image_read_time,
        'throughput_mbps': file_size_mb / avg_full_read_time if avg_full_read_time > 0 else 0
    }

# 测试文件路径
file1 = "/home/madoka/python/rlbench_imitation_learning/data/pick_and_lift_small_size/100demos_hdf5_back/episode_0.hdf5"
file2 = "/home/madoka/python/rlbench_imitation_learning/data/pick_and_lift_small_size/100demos_hdf5/episode_0.hdf5"

# 提示用户清除缓存获得更准确结果
print("建议在测试前清除系统缓存以获得准确结果")
print("您可以运行: sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'")
input("准备好后按回车继续...")

# 运行测试
print("\n开始测试HDF5文件读取速度...")
print(f"文件1: {os.path.basename(file1)}")
print(f"文件2: {os.path.basename(file2)}")

# 预热缓存
with h5py.File(file1, 'r') as f:
    pass
with h5py.File(file2, 'r') as f:
    pass

# 正式测试
results1 = test_read_speed(file1)
results2 = test_read_speed(file2)

# 打印结果表格
print("\n" + "="*60)
print(f"{'参数':<20} {'文件1':<20} {'文件2':<20}")
print("-"*60)
print(f"{'文件大小':<20} {results1['file_size_mb']:.2f} MB{'':<10} {results2['file_size_mb']:.2f} MB")
print(f"{'完整读取时间':<20} {results1['full_read_time']:.4f} 秒{'':<6} {results2['full_read_time']:.4f} 秒")
print(f"{'图像读取时间':<20} {results1['image_read_time']:.4f} 秒{'':<6} {results2['image_read_time']:.4f} 秒")
print(f"{'读取吞吐量':<20} {results1['throughput_mbps']:.2f} MB/秒{'':<4} {results2['throughput_mbps']:.2f} MB/秒")
print("="*60)

# 比较结果
size_diff = abs(results1['file_size_mb'] - results2['file_size_mb']) / max(results1['file_size_mb'], results2['file_size_mb']) * 100
smaller_file = "文件1" if results1['file_size_mb'] < results2['file_size_mb'] else "文件2"
print(f"\n文件大小比较: {smaller_file}更小，差异 {size_diff:.2f}%")

time_diff = abs(results1['full_read_time'] - results2['full_read_time']) / max(results1['full_read_time'], results2['full_read_time']) * 100
faster_file = "文件1" if results1['full_read_time'] < results2['full_read_time'] else "文件2"
print(f"读取速度比较: {faster_file}更快，差异 {time_diff:.2f}%")

# 得出结论
if smaller_file == faster_file:
    print(f"\n结论: {smaller_file}既更小又更快，推荐使用该文件格式。")
else:
    print(f"\n结论: {smaller_file}更小，但{faster_file}读取更快。根据您的需求权衡选择。")