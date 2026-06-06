# 批量处理入口
import argparse

def make_parse():
    parser = argparse.ArgumentParser(description='MOT数据预处理（VideoLLAMA3专用版）')
    parser.add_argument('--dataset_entry', type=str, default="G:/UAVBenchmark", help='数据集总输入路径')
    parser.add_argument('--dataset_name',type=str,default='DynUAVI',help='数据集名称.可选：DynUAVI、VisDrone、UAVDT')
    parser.add_argument('--split_name',type=str,default='test',help="数据子集的名称")
    parser.add_argument('--video_name',type=str, default='009', help='视频名称')

    parser.add_argument('--output_dir', type=str, default="G:/language-conditional_MOT", help='输出总目录')
    parser.add_argument('--no_video', action='store_true', help='不生成可视化视频')
    parser.add_argument('--no_save_frames', action='store_true', help='不生成关键帧上的bbox可视化')
    parser.add_argument('--no_export_json', action='store_true', help='不导出关键帧的bbox坐标')
    parser.add_argument('--no_export_videollama3_input', action='store_true', help='不导出videollama3模型的json输入字典')
    
    parser.add_argument('--add_key_frames', type=str, default='',
                        help='补充关键帧列表，逗号分隔，如: 771,1204,1289,2028')
    
    args = parser.parse_args()
    return args