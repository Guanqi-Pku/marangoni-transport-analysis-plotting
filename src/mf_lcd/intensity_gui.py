import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from matplotlib.figure import Figure
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# --- 1. 全局绘图样式设置 ---
def set_publication_style():
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    mpl.rcParams['font.size'] = 11
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.titlesize'] = 13
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 9
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['axes.linewidth'] = 1.0
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False # 注意：右轴生成时会重新开启 right spine
    mpl.rcParams['figure.autolayout'] = True
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['svg.fonttype'] = 'none'

set_publication_style()

# --- 2. 主GUI应用程序 ---
class FluoAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FluoTS Analyzer Pro V5 - Green Channel & Norm Control")
        self.root.geometry("1480x920")

        # --- 数据状态 ---
        self.folder_path = ""
        self.file_list = []
        self.current_frame_idx = 0
        
        # 缓存图像：img_display 用于显示(RGB)，img_analysis 用于计算(Gray/Green)
        self.img_display = None 
        self.img_analysis = None
        
        # ROI 状态: (x, y, w, h)
        self.roi_signal = None 
        self.roi_norm = None   
        
        self.roi_edit_mode = tk.StringVar(value="none") 

        # --- 布局 ---
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        self.control_panel = ttk.Frame(self.paned_window, width=400, padding="10")
        self.paned_window.add(self.control_panel, weight=0)

        self.right_panel = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_panel, weight=3)

        self.notebook = ttk.Notebook(self.right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_preview = ttk.Frame(self.notebook)
        self.tab_result = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_preview, text="1. Image Preview & ROI Setup")
        self.notebook.add(self.tab_result, text="2. Analysis Result")

        self.setup_control_panel()
        self.setup_preview_tab()
        self.setup_result_tab()

    def setup_control_panel(self):
        frame = self.control_panel
        
        # --- 1. 数据加载 ---
        ttk.Label(frame, text="1. Data Source", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        ttk.Button(frame, text="Load Image Folder", command=self.load_folder).pack(fill=tk.X)
        self.lbl_status = ttk.Label(frame, text="No data loaded", foreground="gray", wraplength=350)
        self.lbl_status.pack(fill=tk.X, pady=2)
        ttk.Separator(frame).pack(fill=tk.X, pady=10)

        # --- 2. ROI 设置 ---
        ttk.Label(frame, text="2. ROI Selection", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        
        frm_roi_tools = ttk.Frame(frame)
        frm_roi_tools.pack(fill=tk.X)
        
        self.rb_view = ttk.Radiobutton(frm_roi_tools, text="Nav", variable=self.roi_edit_mode, value="none", command=self.update_roi_selector_state)
        self.rb_view.pack(side=tk.LEFT)
        self.rb_sig = ttk.Radiobutton(frm_roi_tools, text="Signal (Red)", variable=self.roi_edit_mode, value="signal", command=self.update_roi_selector_state)
        self.rb_sig.pack(side=tk.LEFT, padx=10)
        self.rb_norm = ttk.Radiobutton(frm_roi_tools, text="BG (Blue)", variable=self.roi_edit_mode, value="norm", command=self.update_roi_selector_state)
        self.rb_norm.pack(side=tk.LEFT)
        
        self.lbl_roi_info = ttk.Label(frame, text="Sig: None | Bg: None", font=("Arial", 8))
        self.lbl_roi_info.pack(anchor="w", pady=5)
        ttk.Separator(frame).pack(fill=tk.X, pady=10)

        # --- 3. 阈值预览 ---
        ttk.Label(frame, text="3. Threshold (for Density)", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        self.check_preview_mask = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Show Threshold Mask (Magenta)", variable=self.check_preview_mask, command=self.update_preview_plot).pack(anchor="w")
        
        self.thresh_type = tk.StringVar(value="Adaptive")
        frm_thresh = ttk.Frame(frame)
        frm_thresh.pack(fill=tk.X)
        ttk.Radiobutton(frm_thresh, text="Global", variable=self.thresh_type, value="Global", command=self.update_preview_plot).pack(side=tk.LEFT)
        ttk.Radiobutton(frm_thresh, text="Adaptive", variable=self.thresh_type, value="Adaptive", command=self.update_preview_plot).pack(side=tk.LEFT, padx=10)
        
        self.slider_thresh = tk.Scale(frame, from_=3, to=255, orient=tk.HORIZONTAL, label="Thresh Value / Block Size", command=lambda v: self.update_preview_plot())
        self.slider_thresh.set(11)
        self.slider_thresh.pack(fill=tk.X)
        ttk.Separator(frame).pack(fill=tk.X, pady=10)

        # --- 4. 物理参数 & 归一化选项 ---
        ttk.Label(frame, text="4. Parameters & Normalization", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        frm_phys = ttk.Frame(frame)
        frm_phys.pack(fill=tk.X)
        
        # 改进点2：归一化控制
        self.use_normalization = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Normalize to BG ROI", variable=self.use_normalization).pack(anchor="w", pady=2)
        
        ttk.Label(frm_phys, text="Pixel(µm):").grid(row=0, column=0, sticky="w")
        self.entry_pixel_size = ttk.Entry(frm_phys, width=8)
        self.entry_pixel_size.insert(0, "0.16") 
        self.entry_pixel_size.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(frm_phys, text="TimeGap(s):").grid(row=1, column=0, sticky="w")
        self.entry_time_gap = ttk.Entry(frm_phys, width=8)
        self.entry_time_gap.insert(0, "15")
        self.entry_time_gap.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(frm_phys, text="MaxTime(s):").grid(row=2, column=0, sticky="w")
        self.entry_max_time = ttk.Entry(frm_phys, width=8)
        self.entry_max_time.insert(0, "120")
        self.entry_max_time.grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Separator(frame).pack(fill=tk.X, pady=10)

        # --- 5. 绘图设置 ---
        ttk.Label(frame, text="5. Plot & Smoothing", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        
        frm_smooth = ttk.Frame(frame)
        frm_smooth.pack(fill=tk.X, pady=2)
        ttk.Label(frm_smooth, text="Method:").pack(side=tk.LEFT)
        self.smooth_method_var = tk.StringVar(value="Savgol")
        self.combo_smooth = ttk.Combobox(frm_smooth, textvariable=self.smooth_method_var, 
                                         values=["None", "Savgol", "Moving Avg", "Gaussian"], width=10, state="readonly")
        self.combo_smooth.pack(side=tk.LEFT, padx=5)
        
        frm_param = ttk.Frame(frame)
        frm_param.pack(fill=tk.X, pady=2)
        ttk.Label(frm_param, text="Win/Sigma:").pack(side=tk.LEFT)
        self.entry_smooth_param = ttk.Entry(frm_param, width=6)
        self.entry_smooth_param.insert(0, "51") 
        self.entry_smooth_param.pack(side=tk.LEFT, padx=5)
        
        frm_cmap = ttk.Frame(frame)
        frm_cmap.pack(fill=tk.X, pady=2)
        ttk.Label(frm_cmap, text="Colormap:").pack(side=tk.LEFT)
        self.cmap_var = tk.StringVar(value="viridis")
        self.combo_cmap = ttk.Combobox(frm_cmap, textvariable=self.cmap_var, values=["viridis", "magma", "plasma", "inferno", "coolwarm", "jet"], width=10)
        self.combo_cmap.pack(side=tk.LEFT, padx=5)

        ttk.Separator(frame).pack(fill=tk.X, pady=10)

        # --- 6. 高级显示 ---
        ttk.Label(frame, text="6. Advanced Visuals", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        
        frm_adv = ttk.Frame(frame)
        frm_adv.pack(fill=tk.X)
        self.check_error_bar = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm_adv, text="Error Band", variable=self.check_error_bar).grid(row=0, column=0, sticky="w")
        
        ttk.Label(frm_adv, text="Dens Axis Max:").grid(row=0, column=1, sticky="e", padx=(10,5))
        self.entry_right_ymax = ttk.Entry(frm_adv, width=5)
        self.entry_right_ymax.insert(0, "1.2") 
        self.entry_right_ymax.grid(row=0, column=2, sticky="w")

        ttk.Label(frm_adv, text="Plot Interval:").grid(row=1, column=0, sticky="w", pady=5)
        self.entry_plot_gap = ttk.Entry(frm_adv, width=5)
        self.entry_plot_gap.insert(0, "1")
        self.entry_plot_gap.grid(row=1, column=1, sticky="w", pady=5)

        self.btn_run = ttk.Button(frame, text="Analyze & Plot Results", command=self.run_analysis)
        self.btn_run.pack(fill=tk.X, pady=15)
        
        self.btn_export = ttk.Button(frame, text="Export SVG", command=self.export_plot)
        self.btn_export.pack(fill=tk.X)

        self.progress = ttk.Progressbar(frame, orient="horizontal", mode="determinate")
        self.progress.pack(fill=tk.X, pady=10)

    def setup_preview_tab(self):
        nav_frame = ttk.Frame(self.tab_preview)
        nav_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        ttk.Button(nav_frame, text="<< Prev", command=self.prev_frame).pack(side=tk.LEFT)
        self.lbl_frame_idx = ttk.Label(nav_frame, text="Frame: 0 / 0")
        self.lbl_frame_idx.pack(side=tk.LEFT, padx=10)
        ttk.Button(nav_frame, text="Next >>", command=self.next_frame).pack(side=tk.LEFT)
        
        self.fig_preview = Figure(figsize=(5, 5), dpi=100)
        self.ax_preview = self.fig_preview.add_subplot(111)
        self.ax_preview.axis('off')
        
        self.canvas_preview = FigureCanvasTkAgg(self.fig_preview, master=self.tab_preview)
        self.canvas_preview.draw()
        toolbar = NavigationToolbar2Tk(self.canvas_preview, self.tab_preview)
        toolbar.update()
        self.canvas_preview.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.rs = RectangleSelector(
            self.ax_preview, self.on_select_roi,
            useblit=True, button=[1], minspanx=5, minspany=5,
            spancoords='pixels', interactive=True,
            props=dict(facecolor='none', edgecolor='red', alpha=0.5, fill=False)
        )
        self.rs.set_active(False)

    def setup_result_tab(self):
        self.fig_result = Figure(figsize=(6, 5), dpi=100)
        self.ax_result = self.fig_result.add_subplot(111)
        # 初始化时没有右轴，动态生成
        self.canvas_result = FigureCanvasTkAgg(self.fig_result, master=self.tab_result)
        self.canvas_result.draw()
        toolbar = NavigationToolbar2Tk(self.canvas_result, self.tab_result)
        toolbar.update()
        self.canvas_result.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # --- 核心逻辑 ---
    
    def load_folder(self):
        path = filedialog.askdirectory()
        if not path: return
        exts = ('.tif', '.tiff', '.png', '.jpg', '.bmp')
        self.file_list = sorted([f for f in os.listdir(path) if f.lower().endswith(exts)])
        if not self.file_list:
            messagebox.showerror("Error", "No image files found!")
            return
        self.folder_path = path
        self.lbl_status.config(text=f"Loaded: {len(self.file_list)} images")
        self.current_frame_idx = 0
        self.load_current_image()

    def load_current_image(self):
        if not self.file_list: return
        fname = self.file_list[self.current_frame_idx]
        path = os.path.join(self.folder_path, fname)
        
        # 读取原始数据
        img_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        if img_raw is None:
            print(f"Failed to load: {path}")
            return

        # 改进点1：智能处理 8bit vs RGB Green Channel
        # 准备显示用图像 (img_display) 和分析用图像 (img_analysis)
        
        if len(img_raw.shape) == 2:
            # 8-bit / Grayscale
            self.img_analysis = img_raw
            # 显示用的转RGB
            self.img_display = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)
        
        elif len(img_raw.shape) == 3:
            # Color Image
            # OpenCV 默认读入是 BGR
            # 提取 Green 通道 (Index 1) 用于分析，效果通常优于转灰度
            self.img_analysis = img_raw[:, :, 1] 
            
            # 显示用的转 RGB
            self.img_display = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            
        else:
            messagebox.showerror("Error", "Unsupported image format")
            return

        # 确保 analysis 是 uint8 (为了 adaptiveThreshold)
        if self.img_analysis.dtype != np.uint8:
            self.img_analysis = cv2.normalize(self.img_analysis, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        self.update_preview_plot()
        self.lbl_frame_idx.config(text=f"Frame: {self.current_frame_idx + 1} / {len(self.file_list)}")

    def update_preview_plot(self):
        if self.img_display is None: return
        self.ax_preview.clear()
        self.ax_preview.axis('off')
        
        display_img = self.img_display.copy()
        
        # 预览阈值 Mask
        if self.check_preview_mask.get() and self.img_analysis is not None:
            mask = self.get_threshold_mask(self.img_analysis)
            
            # 创建洋红色覆盖层
            zeros = np.zeros_like(mask)
            mask_magenta = np.dstack((mask, zeros, mask)) 
            
            # 混合显示
            # 将原图变暗，Mask 变亮
            display_gray = cv2.cvtColor(self.img_analysis, cv2.COLOR_GRAY2RGB)
            display_img = cv2.addWeighted(display_gray, 0.6, mask_magenta, 0.4, 0)
            
        self.ax_preview.imshow(display_img)
        
        if self.roi_signal:
            x, y, w, h = self.roi_signal
            self.ax_preview.add_patch(Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none'))
            self.ax_preview.text(x, y-5, "Signal", color='red', fontsize=9, fontweight='bold')
        if self.roi_norm:
            x, y, w, h = self.roi_norm
            self.ax_preview.add_patch(Rectangle((x, y), w, h, linewidth=2, edgecolor='cyan', facecolor='none'))
            self.ax_preview.text(x, y-5, "BG", color='cyan', fontsize=9, fontweight='bold')
            
        self.canvas_preview.draw()

    def get_threshold_mask(self, gray_img):
        val = int(self.slider_thresh.get())
        mode = self.thresh_type.get()
        if mode == "Global":
            _, mask = cv2.threshold(gray_img, val, 255, cv2.THRESH_BINARY)
        else:
            bs = val if val % 2 == 1 else val + 1
            if bs < 3: bs = 3
            mask = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, 2)
        return mask

    def next_frame(self):
        if self.current_frame_idx < len(self.file_list) - 1:
            self.current_frame_idx += 1
            self.load_current_image()

    def prev_frame(self):
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.load_current_image()

    def update_roi_selector_state(self):
        mode = self.roi_edit_mode.get()
        if mode == 'none':
            self.rs.set_active(False)
        else:
            self.rs.set_active(True)
            color = 'red' if mode == 'signal' else 'cyan'
            self.rs.rectprops = dict(facecolor='none', edgecolor=color, alpha=0.8, fill=False, linewidth=2)

    def on_select_roi(self, eclick, erelease):
        x1, y1 = int(round(eclick.xdata)), int(round(eclick.ydata))
        x2, y2 = int(round(erelease.xdata)), int(round(erelease.ydata))
        
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        x = min(x1, x2)
        y = min(y1, y2)
        
        if self.img_analysis is not None:
            H, W = self.img_analysis.shape[:2]
            x = max(0, min(x, W-1))
            y = max(0, min(y, H-1))
            w = min(w, W-x)
            h = min(h, H-y)
            
        roi = (x, y, w, h)
        
        mode = self.roi_edit_mode.get()
        if mode == 'signal':
            self.roi_signal = roi
        elif mode == 'norm':
            self.roi_norm = roi
        self.lbl_roi_info.config(text=f"Sig: {self.roi_signal} | Bg: {self.roi_norm}")
        self.update_preview_plot()

    def apply_smoothing(self, data, method, param):
        if method == "None":
            return data
        
        n = len(data)
        if n < 3: return data
            
        try:
            if method == "Savgol":
                window = int(param)
                if window >= n: window = n - 1 if (n - 1) % 2 == 1 else n - 2
                if window < 5: window = 5 
                if window % 2 == 0: window += 1
                
                poly = 3
                if window <= poly: poly = window - 1
                return savgol_filter(data, window, poly)
            
            elif method == "Moving Avg":
                window = int(param)
                if window >= n: window = n - 1
                if window < 1: window = 1
                kernel = np.ones(window) / window
                return np.convolve(data, kernel, mode='same')
            
            elif method == "Gaussian":
                sigma = float(param)
                return gaussian_filter1d(data, sigma)
                
        except Exception as e:
            print(f"Smoothing error: {e}")
            return data 
            
        return data

    def run_analysis(self):
        if not self.file_list or not self.roi_signal:
            messagebox.showwarning("Warning", "Please select Signal ROI first!")
            return
            
        try:
            self.notebook.select(self.tab_result)
            
            time_gap = float(self.entry_time_gap.get())
            max_time = float(self.entry_max_time.get())
            pixel_size = float(self.entry_pixel_size.get())
            plot_gap = int(self.entry_plot_gap.get())
            
            smooth_method = self.smooth_method_var.get()
            smooth_param = float(self.entry_smooth_param.get())
            
            # 改进点2：获取归一化开关状态
            do_normalize = self.use_normalization.get()
            
            x, y, w, h = self.roi_signal
            
            processed_data = []
            total = len(self.file_list)
            self.progress["maximum"] = total
            
            for i, fname in enumerate(self.file_list):
                t_now = i * time_gap
                if t_now > max_time: break
                
                path = os.path.join(self.folder_path, fname)
                img_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                
                # 处理当前帧（同样的逻辑：取绿色通道或灰度）
                if len(img_raw.shape) == 3:
                    current_slice = img_raw[:, :, 1]
                else:
                    current_slice = img_raw
                
                # --- Signal ROI Stats ---
                roi_slice = current_slice[y:y+h, x:x+w]
                
                # 均值和标准差
                profile = np.mean(roi_slice, axis=0) 
                std_dev = np.std(roi_slice, axis=0)
                
                # --- Normalization Logic ---
                norm_factor = 1.0
                if do_normalize and self.roi_norm:
                    nx, ny, nw, nh = self.roi_norm
                    bg_roi = current_slice[ny:ny+nh, nx:nx+nw]
                    bg_mean = np.mean(bg_roi)
                    if bg_mean > 0:
                        norm_factor = bg_mean
                
                # 计算最终曲线数据
                profile_final = profile / norm_factor
                std_final = std_dev / norm_factor
                
                # --- Density (Threshold) ---
                # 确保用于阈值计算的是 uint8
                if roi_slice.dtype != np.uint8:
                    roi_slice_8u = cv2.normalize(roi_slice, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                else:
                    roi_slice_8u = roi_slice
                    
                mask = self.get_threshold_mask(roi_slice_8u)
                density = np.mean(mask / 255.0, axis=0)
                
                processed_data.append({
                    'time': t_now,
                    'profile': profile_final,
                    'std': std_final,
                    'density': density
                })
                
                if i % 10 == 0:
                    self.progress["value"] = i
                    self.root.update_idletasks()
            
            self.progress["value"] = total
            self.plot_results(processed_data, plot_gap, smooth_method, smooth_param, pixel_size, w)
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def plot_results(self, data, plot_gap, smooth_method, smooth_param, pixel_size, roi_width_px):
        # 改进点3：清理坐标轴，防止 ax2 叠加
        self.ax_result.clear()
        
        # 移除现有的 twinx 轴 (如果存在)
        for axis in self.fig_result.axes:
            if axis != self.ax_result:
                self.fig_result.delaxes(axis)
                
        # 重新创建右轴
        ax2 = self.ax_result.twinx()
        ax2.spines['top'].set_visible(False)
        
        cmap_name = self.cmap_var.get()
        try:
            cmap = plt.get_cmap(cmap_name)
        except:
            cmap = plt.get_cmap('viridis')

        indices = np.linspace(0, 1, len(data))
        x_um = np.arange(roi_width_px) * pixel_size
        
        for i, item in enumerate(data):
            if i % plot_gap != 0: continue
            
            t = item['time']
            # 改进点4：一致的平滑逻辑
            # 对所有数据（Profile, Std, Density）应用相同的平滑参数
            prof_smooth = self.apply_smoothing(item['profile'], smooth_method, smooth_param)
            std_smooth = self.apply_smoothing(item['std'], smooth_method, smooth_param)
            dens_smooth = self.apply_smoothing(item['density'], smooth_method, smooth_param)
            
            color = cmap(indices[i])
            
            # --- 改进点3：Error bar 优化 ---
            # 1. 降低 alpha (0.1)
            # 2. linewidth=0 (去除边框)
            # 3. zorder=1 (确保在最底层)
            if self.check_error_bar.get():
                self.ax_result.fill_between(x_um, 
                                            prof_smooth - std_smooth, 
                                            prof_smooth + std_smooth, 
                                            color=color, alpha=0.1, linewidth=0, zorder=1)
            
            # 主曲线：加粗，高 zorder 确保浮在上面
            self.ax_result.plot(x_um, prof_smooth, color=color, label=f"{t:.0f}s", linewidth=2.5, alpha=0.9, zorder=10)
            
            # 密度曲线：虚线，细一点
            ax2.plot(x_um, dens_smooth, color=color, linestyle='--', linewidth=1.5, alpha=0.4, zorder=5)

        self.ax_result.set_xlabel("Position (µm)", fontsize=13)
        
        # 根据是否归一化设置 Label
        if self.use_normalization.get() and self.roi_norm:
            self.ax_result.set_ylabel("Normalized Intensity (A.U.)", fontsize=13)
        else:
            self.ax_result.set_ylabel("Raw Intensity (Pixel Value)", fontsize=13)
            
        self.ax_result.set_ylim(bottom=0)
        self.ax_result.set_xlim(left=0, right=max(x_um))
        
        # 右轴设置
        try:
            rmax = float(self.entry_right_ymax.get())
        except:
            rmax = 1.1
        ax2.set_ylabel("Condensate Density (Frac.)", fontsize=13, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.set_ylim(0, rmax)
        
        # 图例优化
        handles, labels = self.ax_result.get_legend_handles_labels()
        if len(handles) > 0:
            if len(handles) > 12:
                step = len(handles) // 10 + 1
                self.ax_result.legend(handles[::step], labels[::step], bbox_to_anchor=(1.15, 1), loc='upper left', frameon=False, title="Time (s)")
            else:
                self.ax_result.legend(bbox_to_anchor=(1.15, 1), loc='upper left', frameon=False, title="Time (s)")
            
        self.ax_result.set_title("Fluorescence Profile Evolution", fontsize=14)
        self.fig_result.tight_layout()
        self.canvas_result.draw()

    def export_plot(self):
        fpath = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("SVG", "*.svg"), ("PDF", "*.pdf"), ("PNG", "*.png")])
        if fpath:
            self.fig_result.savefig(fpath, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Saved", f"Plot saved to {fpath}")

if __name__ == "__main__":
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    app = FluoAnalysisApp(root)
    root.mainloop()