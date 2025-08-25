import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
from matplotlib import font_manager
from tqdm import tqdm

def viz_static_frames(actions, static_frames, static_frame_segs, save_path, diversity_info=None, min_continuous_static_frames_threshold=5):
    """
    Visualize static frames and segments on the action sequence timeline.
    Adds statistics for static frames and long static segments.
    
    Args:
        actions: List of action frames
        static_frames: List of all static frame indices
        static_frame_segs: List of static segments (dicts with start/end/length)
        save_path: Path to save the visualization (should end with .jpg)
        diversity_info: Optional diversity information for segment annotations
        min_continuous_static_frames_threshold: Minimum length of a static segment to be considered "long".
    """
    if not actions:
        return

    # Initialize num_diversity_segments to avoid UnboundLocalError
    num_diversity_segments = 0
    if diversity_info and 'distribute' in diversity_info:
        num_diversity_segments = len(diversity_info['distribute'])

    # --- Font Configuration for Chinese Support ---
    try:
        font_path = '/x2robot_v2/wjm/prj/x2robot_dataset/scripts/SimHei.ttf'  # 替换为你的字体文件路径
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = 'SimHei'
        else:
            print(f"警告: 字体文件不存在于 {font_path}，将使用默认字体。中文可能无法正常显示。")
            plt.rcParams['font.family'] = 'sans-serif'
    except Exception as e:
        print(f"加载字体时出错: {e}，将使用默认字体。中文可能无法正常显示。")
        plt.rcParams['font.family'] = 'sans-serif'
        
    plt.rcParams['axes.unicode_minus'] = False # 解决保存图片时负号显示为方块的问题
    # ---------------------------------------------

    fig, ax = plt.subplots(figsize=(15, 6.5)) # 增加高度以适应更多内容
    
    # Draw the main timeline
    total_frames = len(actions)
    ax.plot([0, total_frames], [0, 0], 'k-', linewidth=2)
    
    # 1. Mark all static frames (light red dots)
    if static_frames:
        ax.scatter(static_frames, [0]*len(static_frames), 
                  color='lightcoral', s=30, label='静态帧', zorder=3)
    
    # 2. Highlight static segments (deep red bars)
    for seg in static_frame_segs:
        rect = patches.Rectangle(
            (seg['start'], -0.2), seg['length'], 0.4,
            linewidth=1, edgecolor='darkred', facecolor='darkred', alpha=0.7)
        ax.add_patch(rect)
        # Add segment label
        ax.text(seg['start'] + seg['length']/2, 0.3, 
                f"{seg['length']}帧", 
                ha='center', va='center', color='darkred', fontsize=9)
    
    # 3. Add diversity segments if provided (blue markers and labels)
    if diversity_info and 'distribute' in diversity_info:
        # 使用一个列表来管理每个文本标签的垂直位置，避免重叠
        for i, (seg_range, instruction) in enumerate(diversity_info['distribute'].items()):
            start, end = map(int, seg_range.split())
            
            # 确定一个不与之前文本重叠的 y 位置
            current_y = -0.5 - i * 0.6 
            
            # Draw segment range
            ax.plot([start, end], [current_y, current_y], 'b-', linewidth=2, alpha=0.7)
            ax.scatter([start, end], [current_y, current_y], color='blue', s=50, zorder=3)
            
            # Add instruction text (with Chinese support)
            ax.text((start + end)/2, current_y - 0.15, 
                    instruction, 
                    ha='center', va='top', color='blue', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            
            # Add vertical line to timeline
            ax.plot([start, start], [0, current_y], 'b--', alpha=0.3)
            ax.plot([end, end], [0, current_y], 'b--', alpha=0.3)
    
    # --- Calculate and Add New Statistics ---
    num_static_frames = len(static_frames)
    
    # Calculate total frames in long static segments
    long_static_segments_total_frames = 0
    num_long_static_segments = 0
    if static_frame_segs:
        for seg in static_frame_segs:
            if seg['length'] >= min_continuous_static_frames_threshold:
                long_static_segments_total_frames += seg['length']
                num_long_static_segments += 1

    # Format the statistics strings
    static_frame_ratio_str = f"静态帧: {num_static_frames} / {total_frames} ({num_static_frames/total_frames:.1%})"
    
    long_segment_ratio_str = ""
    if num_long_static_segments > 0:
        long_segment_ratio_str = f"长连续静止段 (≥{min_continuous_static_frames_threshold}帧): {long_static_segments_total_frames} / {total_frames} ({long_static_segments_total_frames/total_frames:.1%})"
    else:
        long_segment_ratio_str = f"长连续静止段 (≥{min_continuous_static_frames_threshold}帧): 0 / {total_frames} (0.0%)"

    # Add statistics text to the plot
    # We'll place them in the upper left corner for visibility
    stat_text = f"{static_frame_ratio_str}\n{long_segment_ratio_str}"
    
    # Calculate appropriate y_pos for statistics based on number of diversity segments
    # This ensures statistics don't overlap with diversity segment text if they are numerous
    stat_y_pos = 0.95 - num_diversity_segments * 0.06  # Start near top, adjust based on diversity segments
    if stat_y_pos < 0.5: # Ensure it's not too low if there are many diversity segments
        stat_y_pos = 0.5
        
    ax.text(0.01, stat_y_pos, stat_text, 
            transform=ax.transAxes, # Use axes coordinates for positioning
            fontsize=10,
            verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5)) # Add a background box for better visibility

    # Configure plot appearance
    ax.set_xlim(0, total_frames)
    
    # Dynamically adjust ylim to accommodate statistics text and diversity segments
    # Calculate required vertical space for diversity segments
    diversity_space_needed = 0
    if diversity_info and 'distribute' in diversity_info:
        diversity_space_needed = num_diversity_segments * 0.6 + 0.5 # Base offset + segments * spacing
    
    # Set the bottom limit based on diversity segments and a small buffer
    ax.set_ylim(-diversity_space_needed, 1.0) 
    
    ax.set_yticks([])
    ax.set_xlabel('帧序号')
    ax.set_title('静态帧可视化')
    
    # Add legend with Chinese labels
    handles = [
        patches.Patch(facecolor='lightcoral', label='静态帧'),
        patches.Patch(facecolor='darkred', alpha=0.7, label='静态区间'),
    ]
    if diversity_info and 'distribute' in diversity_info:
        handles.append(patches.Patch(facecolor='blue', alpha=0.3, label='动作区间'))
    
    legend = ax.legend(handles=handles, loc='upper right')
    # Ensure the legend uses the correct font
    for text in legend.get_texts():
        text.set_fontproperties(font_manager.FontProperties(fname=font_path) if os.path.exists(font_path) else None)
    
    plt.tight_layout()
    
    # Ensure save_path ends with .jpg
    if not save_path.lower().endswith('.jpg'):
        save_path = os.path.splitext(save_path)[0] + '.jpg'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='jpg')
    plt.close()


def json_2_action_list(json_path):
    try:
        with open(json_path, "r") as f:
            actions = json.load(f)
    except:
        raise ValueError("Json file {} open failed!".format(json_path))
    return actions['data']

def is_static_frame(prev_frame, curr_frame, threshold=0.005):
        """
        检测两帧之间是否为静止帧
        
        Args:
            prev_frame: 前一帧数据
            curr_frame: 当前帧数据
            threshold: 变化阈值
            
        Returns:
            bool: 如果所有关键字段变化都小于阈值则返回True，否则返回False
        """
        # 检查每个关键字段
        for key in ["follow_left_position", "follow_left_rotation", "follow_left_gripper", "follow_right_position", "follow_right_rotation", "follow_right_gripper", "car_pose", "lifting_mechanism_position", "head_rotation"]:
            # 确保关键字段存在于两帧中
            if key not in prev_frame or key not in curr_frame:
                continue
            if key in ["follow_left_position", "follow_right_position", "follow_left_rotation", "follow_right_rotation", "car_pose", "head_rotation"]:
                # 对于位置和旋转，分别检查每个坐标分量
                for j in range(len(prev_frame[key])):
                    curr_val = prev_frame[key][j]
                    next_val = curr_frame[key][j]
                    change = abs(next_val - curr_val)
                    if change > threshold:
                        return False
            else:
                # 对于标量值
                curr_val = prev_frame[key]
                next_val = curr_frame[key]
                change = abs(next_val - curr_val)
                if change > threshold:
                    return False
        return True


def get_static_frame_segs(actions, threshold=0.005, min_seg_length=10):
    """
    获取静止帧信息
    
    Args:
        actions: 动作列表（每个元素是一帧的动作数据）
        threshold: 静止判定阈值
        min_seg_length: 最小静止段长度
        
    Returns:
        tuple: (static_frames, static_frame_segs)
            static_frames: 所有静止帧的序号列表
            static_frame_segs: 连续静止段列表，每个元素是{
                'start': 起始帧,
                'end': 结束帧,
                'length': 段长度
            }
    """
    static_frames = []
    static_frame_segs = []
    
    if not actions or len(actions) < 2:
        return static_frames, static_frame_segs
    
    # 1. 找出所有静止帧
    for i in range(1, len(actions)):
        if is_static_frame(actions[i-1], actions[i], threshold):
            static_frames.append(i)
    
    # 2. 如果没有静止帧，直接返回空列表
    if not static_frames:
        return static_frames, static_frame_segs
    
    # 3. 找出连续的静止段
    current_seg_start = static_frames[0]
    current_seg_length = 1
    
    for i in range(1, len(static_frames)):
        if static_frames[i] == static_frames[i-1] + 1:
            current_seg_length += 1
        else:
            # 当前段结束，检查长度
            if current_seg_length >= min_seg_length:
                static_frame_segs.append({
                    'start': current_seg_start,
                    'end': current_seg_start + current_seg_length - 1,
                    'length': current_seg_length
                })
            # 开始新段
            current_seg_start = static_frames[i]
            current_seg_length = 1
    
    # 处理最后一个段
    if current_seg_length >= min_seg_length:
        static_frame_segs.append({
            'start': current_seg_start,
            'end': current_seg_start + current_seg_length - 1,
            'length': current_seg_length
        })
    
    return static_frames, static_frame_segs

class StaticFrameChecker:
    def __init__(self, threshold=0.005, min_seg_length=10):
        self.threshold = threshold
        self.min_seg_length = min_seg_length

    def batch_check(self, topic_path_list, json_save_path, viz_save_folder=None):
        sample_2_static_frames = {}
        for topic_path in topic_path_list:
            if not os.path.exists(topic_path):
                raise ValueError("Topic {} not exists!".format(topic))

            print(f"Processing {topic_path}...")
            for sample in tqdm(os.listdir(topic_path)):
                sample_path = os.path.join(topic_path, sample)
                video_path = os.path.join(sample_path, "faceImg.mp4")
                if not os.path.exists(video_path):
                    continue
                action_path = os.path.join(sample_path, f"{sample}.json")
                if not os.path.exists(action_path):
                    raise ValueError("Action file {} not exists!".format(action_path))
                
                diversity_info = None
                diversity_path = os.path.join(sample_path, "diversity.json")
                if os.path.exists(diversity_path):
                    try:
                        with open(diversity_path, "r") as f:
                            diversity_info = json.load(f)
                    except:
                        print(f"Diversity file {diversity_path} open failed!")

                actions = json_2_action_list(action_path)
                
                static_frames, static_frame_segs = get_static_frame_segs(actions, threshold=0.005, min_seg_length=10)

                if viz_save_folder is not None:
                    save_path = os.path.join(viz_save_folder, f"{sample}.jpg")
                    viz_static_frames(actions, static_frames, static_frame_segs, save_path=save_path, diversity_info=diversity_info)

                sample_2_static_frames[sample] = {
                    'static_frames': static_frames,
                    'static_frame_segs': static_frame_segs
                }

        with open(json_save_path, "w") as f:
            json.dump(sample_2_static_frames, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':


    # 这里填写你想要检测静止帧的所有topic的路径
    topic_path_list = [
        '/x2robot_data/zhengwei/10103/20250707-day-navigate_and_pick_waste-dual-turtle',
    ]


    # 创建StaticFrameChecker，threshold代表相邻两帧的变化阈值，min_seg_length代表连续静止段的最小长度
    static_frame_checker = StaticFrameChecker(threshold=0.05, min_seg_length=5)


    # json_save_path用于保存检测到的静止帧信息，viz_save_folder用于保存静止帧可视化结果（按照每个episode来可视化）
    static_frame_checker.batch_check(
        topic_path_list, 
        json_save_path='/x2robot_v2/wjm/prj/outputs/static_frames.json',
        viz_save_folder='/x2robot_v2/wjm/prj/outputs/imgs'
    )