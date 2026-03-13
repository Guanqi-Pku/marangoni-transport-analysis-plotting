import os
import shutil
from pathlib import Path
import logging

# =================配置区域 (Configuration) =================

# 1. 目标文件夹路径 (设置为 '.' 表示当前脚本所在目录)
TARGET_DIRECTORY = 'D:/Research/microfluid_pgl_meg/0.5Delta+7.5PGL_mf'

# 2. 映射字典：定义 {文件后缀: 目标子文件夹名}
# 可根据需要无限扩展，例如添加 '.ch03': 'Infrared'
CHANNEL_MAPPING = {
    'ch00.tif': 'meg',    
    'ch01.tif': 'pgl',  
    'ch02.tif': 'bf'  
    # '.tif': 'Original_Images',   # 也可以处理普通图片
}

# 3. 是否开启演示模式 (True: 会先自动创建一些假文件供测试; False: 处理实际文件)
DEMO_MODE = False

# =================代码实现区域 (Implementation) =================

class ImageOrganizer:
    def __init__(self, target_dir, mapping_rules):
        """
        初始化整理器
        :param target_dir: 包含混乱图像的根目录
        :param mapping_rules: 后缀名与文件夹的映射字典
        """
        self.target_dir = Path(target_dir)
        self.mapping_rules = mapping_rules
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)

    def generate_demo_files(self):
        """仅用于演示：生成一些假文件"""
        self.logger.info("正在生成测试文件...")
        demo_files = [
            "sample_A.ch00", "sample_A.ch01", "sample_A.ch02",
            "sample_B.ch00", "sample_B.ch01",
            "test_image.ch02", "ignored_file.txt"
        ]
        for fname in demo_files:
            (self.target_dir / fname).touch()
        self.logger.info(f"已在 {self.target_dir} 生成测试文件。")

    def organize(self):
        """执行核心分类逻辑"""
        if not self.target_dir.exists():
            self.logger.error(f"目标目录不存在: {self.target_dir}")
            return

        self.logger.info(f"开始整理目录: {self.target_dir.resolve()}")
        
        # 统计计数器
        stats = {k: 0 for k in self.mapping_rules.values()}
        stats['Unmoved'] = 0

        # 遍历目录下所有文件 (不递归进入子文件夹，防止重复移动)
        # 如果需要递归，将 self.target_dir.iterdir() 改为 self.target_dir.rglob('*')
        for file_path in self.target_dir.iterdir():
            if file_path.is_dir():
                continue

            moved = False
            # 检查文件是否匹配映射规则
            for suffix, folder_name in self.mapping_rules.items():
                # 使用 endswith 匹配，这样既可以匹配 .ch00 也可以匹配 .tif.ch00
                if file_path.name.endswith(suffix):
                    dest_folder = self.target_dir / folder_name
                    self._move_file(file_path, dest_folder)
                    stats[folder_name] += 1
                    moved = True
                    break # 匹配到一个规则后即停止，避免多重匹配
            
            if not moved:
                stats['Unmoved'] += 1

        self.print_summary(stats)

    def _move_file(self, source_path, dest_folder):
        """移动单个文件的辅助方法"""
        try:
            # 确保目标子文件夹存在
            dest_folder.mkdir(exist_ok=True)
            
            destination_path = dest_folder / source_path.name
            
            # 移动文件
            shutil.move(str(source_path), str(destination_path))
            self.logger.info(f"移动: {source_path.name} -> {dest_folder.name}/")
            
        except Exception as e:
            self.logger.error(f"移动 {source_path.name} 失败: {e}")

    def print_summary(self, stats):
        print("\n" + "="*30)
        print("处理完成统计报告")
        print("="*30)
        for folder, count in stats.items():
            print(f"{folder}: {count} 个文件")
        print("="*30)

if __name__ == "__main__":
    # 实例化整理器
    organizer = ImageOrganizer(TARGET_DIRECTORY, CHANNEL_MAPPING)

    # 如果是演示模式，先生成假文件
    if DEMO_MODE:
        print(">>> 演示模式已开启 <<<")
        organizer.generate_demo_files()
    
    # 运行整理
    organizer.organize()
    
    if DEMO_MODE:
        print("\n提示: 这是演示运行。将代码中的 DEMO_MODE 改为 False 即可处理真实文件。")