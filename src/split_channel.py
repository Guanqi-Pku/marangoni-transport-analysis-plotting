import os
import shutil
from pathlib import Path
import logging

# ================= Configuration Area =================

# 1. Target folder path (set to '.' to use the folder containing this script)
TARGET_DIRECTORY = 'D:/Research/microfluid_pgl_meg/0.5Delta+7.5PGL_mf'

# 2. Mapping dictionary: define {file extension: target subfolder name}
# Extend as needed, e.g., add '.ch03': 'Infrared'
CHANNEL_MAPPING = {
    'ch00.tif': 'meg',    
    'ch01.tif': 'pgl',  
    'ch02.tif': 'bf'  
    # '.tif': 'Original_Images',   # regular image files can also be processed
}

# 3. Whether to enable demo mode (True: automatically create dummy files for testing first; False: process actual files)
DEMO_MODE = False

# ================= Implementation Area =================

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
        
        # Counters
        stats = {k: 0 for k in self.mapping_rules.values()}
        stats['Unmoved'] = 0

        # Iterate over all files in the directory (do not recurse into subfolders to avoid duplicate moves)
        # To recurse, change self.target_dir.iterdir() to self.target_dir.rglob('*')
        for file_path in self.target_dir.iterdir():
            if file_path.is_dir():
                continue

            moved = False
            # Check whether the file matches a mapping rule
            for suffix, folder_name in self.mapping_rules.items():
                # Use endswith for matching, so both .ch00 and .tif.ch00 are supported
                if file_path.name.endswith(suffix):
                    dest_folder = self.target_dir / folder_name
                    self._move_file(file_path, dest_folder)
                    stats[folder_name] += 1
                    moved = True
                    break # Stop after the first matching rule to avoid multiple matches
            
            if not moved:
                stats['Unmoved'] += 1

        self.print_summary(stats)

    def _move_file(self, source_path, dest_folder):
        """移动单个文件的辅助方法"""
        try:
            # Ensure the target subfolder exists
            dest_folder.mkdir(exist_ok=True)
            
            destination_path = dest_folder / source_path.name
            
            # Move file
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
    # Instantiate the organizer
    organizer = ImageOrganizer(TARGET_DIRECTORY, CHANNEL_MAPPING)

    # If in demo mode, generate dummy files first
    if DEMO_MODE:
        print(">>> 演示模式已开启 <<<")
        organizer.generate_demo_files()
    
    # Run the organizer
    organizer.organize()
    
    if DEMO_MODE:
        print("\n提示: 这是演示运行。将代码中的 DEMO_MODE 改为 False 即可处理真实文件。")