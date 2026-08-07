import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.optimize import curve_fit
import os
import glob
import platform

# --- Global Configuration: Font Settings ---
system_name = platform.system()
if system_name == "Windows":
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
elif system_name == "Darwin": # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC']
else: # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False 

# --- Scientific Computation and Fitting Functions ---

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def log_normal(x, a, mu, sigma):
    safe_x = np.where(x > 0, x, 1e-10) 
    return (a / (safe_x * sigma * np.sqrt(2 * np.pi))) * np.exp(-(np.log(safe_x) - mu)**2 / (2 * sigma**2))

# --- Main Application Class ---

class ParticleAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FluoParticleAnalyzer - 荧光颗粒尺寸分析工具")
        self.root.geometry("1400x1000") 

        # --- Data State ---
        self.image_paths = []
        self.current_index = 0
        self.current_image = None
        self.gray_image = None
        self.detected_particles = [] 
        self.detected_contours = []
        self.pixel_ratio = 0.414  
        self.time_interval = 5.0  
        self.current_coverage = 0.0 
        
        # --- Tracking State ---
        self.tracks = {} # {track_id: {frame_idx: {'x':, 'y':, 'd':, ...}}}
        self.active_track_id = 1
        self.is_tracking_mode = False
        self.contour_centers = [] 
        
        # --- View Interaction State ---
        self.view_limits = None 
        self.pan_start = None # Drag start point (x, y) in pixel coordinates
        self.pan_ax_limits = None # Axis limits when dragging starts

        # --- Interface Layout ---
        self._init_ui()
        
    def _init_ui(self):
        # --- 1. Left Scrollable Container Setup ---
        left_container = ttk.Frame(self.root)
        left_container.pack(side=tk.LEFT, fill=tk.Y)

        self.left_canvas = tk.Canvas(left_container, width=340) 
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=self.left_canvas.yview)
        
        self.scrollable_frame = ttk.Frame(self.left_canvas, padding="10")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))
        )

        self.left_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.left_canvas.configure(yscrollcommand=scrollbar.set)

        self.left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.left_canvas.bind_all("<MouseWheel>", self._on_ui_scroll) # Windows
        self.left_canvas.bind_all("<Button-4>", self._on_ui_scroll)   # Linux
        self.left_canvas.bind_all("<Button-5>", self._on_ui_scroll)   # Linux

        # --- Add Controls to scrollable_frame ---
        control_frame = self.scrollable_frame 

        # File Operations
        file_group = ttk.LabelFrame(control_frame, text="文件操作", padding="5")
        file_group.pack(fill=tk.X, pady=5)
        ttk.Button(file_group, text="加载单张图像", command=self.load_single_image).pack(fill=tk.X, pady=2)
        ttk.Button(file_group, text="加载文件夹序列", command=self.load_folder).pack(fill=tk.X, pady=2)
        self.file_label = ttk.Label(file_group, text="未加载文件", wraplength=280)
        self.file_label.pack(pady=5)

        # Navigation
        nav_group = ttk.LabelFrame(control_frame, text="播放控制", padding="5")
        nav_group.pack(fill=tk.X, pady=5)
        
        btn_nav_frame = ttk.Frame(nav_group)
        btn_nav_frame.pack(fill=tk.X)
        ttk.Button(btn_nav_frame, text="< 上一帧", command=self.prev_image).pack(side=tk.LEFT, expand=True)
        ttk.Button(btn_nav_frame, text="下一帧 >", command=self.next_image).pack(side=tk.RIGHT, expand=True)
        
        self.frame_slider = tk.Scale(nav_group, from_=0, to=0, orient=tk.HORIZONTAL, command=self.on_slider_change)
        self.frame_slider.pack(fill=tk.X, pady=5)
        self.frame_info_label = ttk.Label(nav_group, text="Frame: 0/0")
        self.frame_info_label.pack()

        # Trajectory Tracking
        track_group = ttk.LabelFrame(control_frame, text="轨迹追踪 (手动)", padding="5")
        track_group.pack(fill=tk.X, pady=5)
        
        self.track_mode_btn = ttk.Button(track_group, text="开启追踪模式", command=self.toggle_tracking_mode)
        self.track_mode_btn.pack(fill=tk.X, pady=5)
        
        ttk.Label(track_group, text="轨迹列表 (点击切换):").pack(anchor="w")
        
        columns = ("ID", "Points")
        self.track_tree = ttk.Treeview(track_group, columns=columns, show="headings", height=6)
        self.track_tree.heading("ID", text="Track ID")
        self.track_tree.heading("Points", text="点数")
        self.track_tree.column("ID", width=60, anchor="center")
        self.track_tree.column("Points", width=60, anchor="center")
        self.track_tree.pack(fill=tk.X, pady=2)
        
        self.track_tree.bind("<<TreeviewSelect>>", self.on_track_tree_select)

        action_frame = ttk.Frame(track_group)
        action_frame.pack(fill=tk.X, pady=5)
        ttk.Button(action_frame, text="新建轨迹", command=self.add_new_track).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(action_frame, text="删除选中轨迹", command=self.delete_entire_track).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=1)
        
        ttk.Button(track_group, text="删除当前帧的点", command=self.delete_current_point).pack(fill=tk.X, pady=2)
        
        ttk.Separator(track_group, orient='horizontal').pack(fill=tk.X, pady=5)
        ttk.Button(track_group, text="导出轨迹数据 (CSV)", command=self.export_tracks).pack(fill=tk.X, pady=2)
        ttk.Button(track_group, text="导出轨迹矢量图 (PDF/SVG)", command=self.export_tracks_vector_graphic).pack(fill=tk.X, pady=2)

        # Parameter Settings
        param_group = ttk.LabelFrame(control_frame, text="检测参数", padding="5")
        param_group.pack(fill=tk.X, pady=5)

        ttk.Label(param_group, text="像素比 (um/px):").pack(anchor="w")
        self.ratio_entry = ttk.Entry(param_group)
        self.ratio_entry.insert(0, "0.414")
        self.ratio_entry.pack(fill=tk.X)
        self.ratio_entry.bind("<Return>", self.update_analysis)

        ttk.Label(param_group, text="时间间隔 (s):").pack(anchor="w")
        self.time_entry = ttk.Entry(param_group)
        self.time_entry.insert(0, "5.0")
        self.time_entry.pack(fill=tk.X)

        # Detection Controls
        detect_group = ttk.LabelFrame(control_frame, text="检测控制", padding="5")
        detect_group.pack(fill=tk.X, pady=5)

        ttk.Label(detect_group, text="检测算法:").pack(anchor="w")
        self.algo_var = tk.StringVar(value="Global Threshold")
        algo_combo = ttk.Combobox(detect_group, textvariable=self.algo_var, state="readonly")
        # Algorithm Update Options
        algo_combo['values'] = ("Global Threshold", "Top-Hat (Uneven Bg)")
        algo_combo.pack(fill=tk.X)
        algo_combo.bind("<<ComboboxSelected>>", self.update_analysis)

        ttk.Label(detect_group, text="亮度阈值 (0-255):").pack(anchor="w")
        self.thresh_scale = ttk.Scale(detect_group, from_=0, to=255, orient=tk.HORIZONTAL, command=lambda v: self.update_analysis())
        self.thresh_scale.set(50)
        self.thresh_scale.pack(fill=tk.X)

        ttk.Label(detect_group, text="最小直径 (um):").pack(anchor="w")
        self.min_size_entry = ttk.Entry(detect_group)
        self.min_size_entry.insert(0, "0.5")
        self.min_size_entry.pack(fill=tk.X)
        self.min_size_entry.bind("<Return>", self.update_analysis)

        ttk.Label(detect_group, text="最大直径 (um):").pack(anchor="w")
        self.max_size_entry = ttk.Entry(detect_group)
        self.max_size_entry.insert(0, "50.0")
        self.max_size_entry.pack(fill=tk.X)
        self.max_size_entry.bind("<Return>", self.update_analysis)

        # Fitting Controls
        fit_group = ttk.LabelFrame(control_frame, text="分布拟合", padding="5")
        fit_group.pack(fill=tk.X, pady=5)
        ttk.Label(fit_group, text="拟合函数:").pack(anchor="w")
        self.fit_var = tk.StringVar(value="None")
        fit_combo = ttk.Combobox(fit_group, textvariable=self.fit_var, state="readonly")
        fit_combo['values'] = ("None", "Gaussian", "Log-Normal")
        fit_combo.pack(fill=tk.X)
        fit_combo.bind("<<ComboboxSelected>>", self.update_analysis)

        # Batch Export Button
        ttk.Button(control_frame, text="批量导出统计数据 (CSV)", command=self.export_data).pack(fill=tk.X, pady=20)

        # --- 2. Right Plotting Area ---
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(12, 12), dpi=100)
        gs = self.fig.add_gridspec(2, 1, height_ratios=[4, 1])
        
        self.ax_img = self.fig.add_subplot(gs[0])
        self.ax_hist = self.fig.add_subplot(gs[1])
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Bind plot interaction events
        self.canvas.mpl_connect('scroll_event', self.on_graph_scroll)       # Scroll-wheel zoom
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)     # Click / start dragging
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release) # End dragging
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_drag)     # Dragging
        
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _on_ui_scroll(self, event):
        """处理左侧UI面板的滚轮滚动"""
        if platform.system() == 'Windows':
            self.left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        elif platform.system() == 'Linux':
            if event.num == 4:
                self.left_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.left_canvas.yview_scroll(1, "units")
        else: # macOS
             self.left_canvas.yview_scroll(int(-1*event.delta), "units")

    # --- View Interaction Logic (Dragging and Zooming) ---

    def on_graph_scroll(self, event):
        """处理图像区域的滚轮缩放"""
        if event.inaxes != self.ax_img: return

        # Get current axis limits (x0, x1), (y0, y1)
        # Fix: get the limit endpoints directly and preserve direction to prevent the y-axis from flipping due to sorting
        ax = self.ax_img
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim() 

        # Determine zoom scale
        base_scale = 1.1
        if event.button == 'up':
            # Zoom in
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            # Zoom out
            scale_factor = base_scale
        else:
            return

        # Current mouse position
        cur_x = event.xdata
        cur_y = event.ydata
        if cur_x is None or cur_y is None: return

        # Calculate current view span (including direction sign)
        w = x1 - x0
        h = y1 - y0

        # Calculate the mouse's relative position within the current view
        rel_x = (cur_x - x0) / w
        rel_y = (cur_y - y0) / h

        # Calculate the new span
        new_w = w * scale_factor
        new_h = h * scale_factor

        # Calculate the new starting point from the relative position, ensuring the zoom center stays at the mouse position
        new_x0 = cur_x - new_w * rel_x
        new_x1 = new_x0 + new_w
        
        new_y0 = cur_y - new_h * rel_y
        new_y1 = new_y0 + new_h

        # Set new limits
        ax.set_xlim([new_x0, new_x1])
        ax.set_ylim([new_y0, new_y1])
        
        self.view_limits = (ax.get_xlim(), ax.get_ylim())
        self.canvas.draw_idle() 

    def on_mouse_press(self, event):
        """处理鼠标按下：左键选点，右键拖动"""
        if event.inaxes != self.ax_img: return

        if event.button == 1: # Left click: select a point for trajectory tracking
            self.handle_track_click(event)
        elif event.button == 3: # Right click: start panning the view
            self.pan_start = (event.x, event.y)
            self.pan_ax_limits = (self.ax_img.get_xlim(), self.ax_img.get_ylim())

    def on_mouse_drag(self, event):
        """处理鼠标移动：右键拖动视野"""
        if self.pan_start and event.inaxes == self.ax_img:
            dx_pix = event.x - self.pan_start[0]
            dy_pix = event.y - self.pan_start[1]
            
            start_xlim, start_ylim = self.pan_ax_limits
            
            # Get the current axis pixel dimensions on screen
            bbox = self.ax_img.bbox
            width_pix = bbox.width
            height_pix = bbox.height
            
            if width_pix == 0 or height_pix == 0: return

            # Convert pixel differences to data-coordinate differences
            # Note: if the mouse is dragged right (dx>0), the view should move left (decrease xlim), and vice versa
            dx_data = (start_xlim[1] - start_xlim[0]) / width_pix * dx_pix
            dy_data = (start_ylim[1] - start_ylim[0]) / height_pix * dy_pix
            
            new_xlim = (start_xlim[0] - dx_data, start_xlim[1] - dx_data)
            new_ylim = (start_ylim[0] - dy_data, start_ylim[1] - dy_data)
            
            self.ax_img.set_xlim(new_xlim)
            self.ax_img.set_ylim(new_ylim)
            self.view_limits = (new_xlim, new_ylim)
            self.canvas.draw_idle()

    def on_mouse_release(self, event):
        """处理鼠标释放"""
        if event.button == 3:
            self.pan_start = None
            self.pan_ax_limits = None

    # --- File Processing ---

    def load_single_image(self):
        path = filedialog.askopenfilename(filetypes=[("TIF Images", "*.tif"), ("All Files", "*.*")])
        if path:
            self.image_paths = [path]
            self.current_index = 0
            self.view_limits = None 
            self.tracks = {} 
            self.active_track_id = 1
            self.update_track_list()
            self.update_slider_range()
            self.load_image_data()

    def load_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            files = sorted(glob.glob(os.path.join(folder, "*.tif")))
            if not files:
                messagebox.showerror("错误", "该文件夹中未找到TIF图像。")
                return
            self.image_paths = files
            self.current_index = 0
            self.view_limits = None 
            self.tracks = {} 
            self.active_track_id = 1
            self.update_track_list()
            self.update_slider_range()
            self.load_image_data()

    def update_slider_range(self):
        count = len(self.image_paths)
        if count > 0:
            self.frame_slider.config(to=count-1)
        else:
            self.frame_slider.config(to=0)

    def on_slider_change(self, value):
        idx = int(value)
        if idx != self.current_index and 0 <= idx < len(self.image_paths):
            self.current_index = idx
            self.load_image_data(preserve_view=True)

    def load_image_data(self, preserve_view=False):
        if not self.image_paths:
            return
            
        path = self.image_paths[self.current_index]
        self.file_label.config(text=f"文件: {os.path.basename(path)}")
        self.frame_info_label.config(text=f"Frame: {self.current_index + 1}/{len(self.image_paths)}")
        self.frame_slider.set(self.current_index)
        
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None: return

        if len(img.shape) == 2:
            self.gray_image = img
            if self.gray_image.dtype == 'uint16':
                self.gray_image = cv2.normalize(self.gray_image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            self.current_image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2RGB)
        else:
            self.current_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.update_analysis(preserve_view=preserve_view)

    def prev_image(self):
        if self.image_paths and self.current_index > 0:
            self.current_index -= 1
            self.load_image_data(preserve_view=True)

    def next_image(self):
        if self.image_paths and self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.load_image_data(preserve_view=True)

    # --- Trajectory Tracking Logic ---

    def toggle_tracking_mode(self):
        self.is_tracking_mode = not self.is_tracking_mode
        if self.is_tracking_mode:
            self.track_mode_btn.config(text="停止追踪模式 (Tracking ON)", style="Accent.TButton")
        else:
            self.track_mode_btn.config(text="开启追踪模式", style="TButton")
        self.update_analysis(preserve_view=True)

    def update_track_list(self):
        """更新 Treeview 内容"""
        for item in self.track_tree.get_children():
            self.track_tree.delete(item)
            
        track_ids = sorted(list(self.tracks.keys()))
        
        if self.active_track_id not in track_ids:
             track_ids.append(self.active_track_id)
             track_ids = sorted(list(set(track_ids)))

        for tid in track_ids:
            points_count = len(self.tracks.get(tid, {}))
            self.track_tree.insert("", "end", iid=str(tid), values=(tid, points_count))
            
        if str(self.active_track_id) in self.track_tree.get_children():
            self.track_tree.selection_set(str(self.active_track_id))
            self.track_tree.see(str(self.active_track_id))

    def on_track_tree_select(self, event):
        selected_items = self.track_tree.selection()
        if selected_items:
            try:
                new_id = int(selected_items[0])
                self.active_track_id = new_id
                self.update_analysis(preserve_view=True)
            except:
                pass

    def add_new_track(self):
        if not self.tracks:
            new_id = 1
        else:
            new_id = max(self.tracks.keys()) + 1
        
        self.active_track_id = new_id
        self.update_track_list()
        self.update_analysis(preserve_view=True)

    def delete_entire_track(self):
        selected_items = self.track_tree.selection()
        if not selected_items:
            messagebox.showinfo("提示", "请先在列表中选中要删除的轨迹")
            return
            
        tid_to_delete = int(selected_items[0])
        
        confirm = messagebox.askyesno("确认删除", f"确定要删除整条轨迹 (Track {tid_to_delete}) 吗？")
        if confirm:
            if tid_to_delete in self.tracks:
                del self.tracks[tid_to_delete]
            
            remains = sorted(list(self.tracks.keys()))
            if remains:
                self.active_track_id = remains[0]
            else:
                self.active_track_id = 1
                
            self.update_track_list()
            self.update_analysis(preserve_view=True)

    def delete_current_point(self):
        if self.active_track_id in self.tracks:
            if self.current_index in self.tracks[self.active_track_id]:
                del self.tracks[self.active_track_id][self.current_index]
                self.update_track_list() 
                self.update_analysis(preserve_view=True)

    def handle_track_click(self, event):
        """处理轨迹追踪的点选"""
        if not self.is_tracking_mode:
            return
        
        click_x, click_y = event.xdata, event.ydata
        if click_x is None or click_y is None: return
        
        if not self.contour_centers:
            return

        centers = np.array(self.contour_centers)
        distances = np.sqrt((centers[:, 0] - click_x)**2 + (centers[:, 1] - click_y)**2)
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]

        if min_dist < 20: 
            diameter_um = self.detected_particles[min_idx]
            
            center_x_px = centers[min_idx][0]
            center_y_px = centers[min_idx][1]
            
            if self.active_track_id not in self.tracks:
                self.tracks[self.active_track_id] = {}
            
            ratio = self.pixel_ratio
            time_point = self.current_index * self.time_interval
            
            self.tracks[self.active_track_id][self.current_index] = {
                "x_px": center_x_px,
                "y_px": center_y_px,
                "x_um": center_x_px * ratio,
                "y_um": center_y_px * ratio,
                "diameter_um": diameter_um,
                "area_um2": np.pi * (diameter_um/2)**2,
                "time_s": time_point
            }
            
            self.update_track_list()
            self.update_analysis(preserve_view=True)

    def export_tracks(self):
        if not self.tracks:
            messagebox.showinfo("提示", "没有记录任何轨迹数据。")
            return
            
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if not save_path:
            return

        rows = []
        for tid, frames in self.tracks.items():
            sorted_frames = sorted(frames.keys())
            for fid in sorted_frames:
                data = frames[fid]
                rows.append({
                    "Track_ID": tid,
                    "Frame_Index": fid,
                    "Time_s": data['time_s'],
                    "X_um": data['x_um'],
                    "Y_um": data['y_um'],
                    "Diameter_um": data['diameter_um'],
                    "Area_um2": data['area_um2']
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        messagebox.showinfo("成功", f"轨迹数据已导出至: {save_path}")

    def export_tracks_vector_graphic(self):
        if not self.tracks:
            messagebox.showinfo("提示", "没有记录任何轨迹数据。")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".pdf", 
            filetypes=[("PDF File", "*.pdf"), ("SVG File", "*.svg")]
        )
        if not save_path:
            return

        if self.current_image is not None:
            h, w = self.current_image.shape[:2]
        else:
            h, w = 1000, 1000

        fig_temp = plt.Figure(figsize=(10, 10 * h / w), dpi=100)
        ax_temp = fig_temp.add_subplot(111)

        colors = ['red', 'blue', 'orange', 'magenta', 'cyan', 'yellow', 'green', 'purple']

        for tid, frames in self.tracks.items():
            color = colors[tid % len(colors)]
            sorted_fids = sorted(frames.keys())
            px_coords = [(frames[fid]['x_px'], frames[fid]['y_px']) for fid in sorted_fids]
            
            if len(px_coords) > 1:
                px_coords = np.array(px_coords)
                ax_temp.plot(px_coords[:, 0], px_coords[:, 1], '-', color=color, linewidth=1.5, alpha=0.8)
                
                ax_temp.plot(px_coords[0, 0], px_coords[0, 1], 'o', color=color, markersize=4)
                mid_idx = len(px_coords) // 2
                ax_temp.text(px_coords[mid_idx, 0], px_coords[mid_idx, 1], f"T{tid}", color=color, fontsize=10, fontweight='bold')
            elif len(px_coords) == 1:
                 ax_temp.plot(px_coords[0][0], px_coords[0][1], 'o', color=color, markersize=4)
                 ax_temp.text(px_coords[0][0], px_coords[0][1], f"T{tid}", color=color, fontsize=10, fontweight='bold')

        ax_temp.set_xlim(0, w)
        ax_temp.set_ylim(h, 0) 
        
        ax_temp.axis('off')
        
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            canvas = FigureCanvasAgg(fig_temp)
            fig_temp.savefig(save_path, transparent=True, bbox_inches='tight', pad_inches=0)
            messagebox.showinfo("成功", f"轨迹矢量图已导出至: {save_path}")
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {str(e)}")

    # --- Core Analysis Logic ---

    def detect_particles(self, img_gray, ratio, min_um, max_um, threshold_val):
        blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
        
        # 1. Algorithm Selection
        if self.algo_var.get() == "Global Threshold":
            _, binary = cv2.threshold(blurred, int(threshold_val), 255, cv2.THRESH_BINARY)
            
        elif self.algo_var.get() == "Top-Hat (Uneven Bg)":
            # 2. Calculate the Appropriate Kernel Size (Top-Hat Kernel Size)
            # Must be larger than the largest particle diameter; otherwise the particle interiors will be "subtracted" and become hollow
            # Default: convert max_um to pixels and multiply by 1.5
            max_px = max_um / ratio
            ksize = int(max_px * 1.5) 
            if ksize % 2 == 0: ksize += 1 # Ensure it is odd
            if ksize < 3: ksize = 3
            
            # 3. Morphological Top-Hat Transform: original image - opened image (background)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
            
            # 4. Apply a threshold to the processed "flat" background image
            # The threshold slider now controls how much brighter the particles are than the background
            _, binary = cv2.threshold(tophat, int(threshold_val), 255, cv2.THRESH_BINARY)
        else:
            # Fallback
            _, binary = cv2.threshold(blurred, int(threshold_val), 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_diameters = []
        valid_contours = []
        valid_centers = []
        
        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            if area_px == 0: continue
            
            diameter_px = 2 * np.sqrt(area_px / np.pi)
            diameter_um = diameter_px * ratio
            
            if min_um <= diameter_um <= max_um:
                valid_diameters.append(diameter_um)
                valid_contours.append(cnt)
                
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0
                valid_centers.append((cX, cY))
                
        return np.array(valid_diameters), valid_contours, valid_centers, binary

    def update_analysis(self, event=None, preserve_view=False):
        if self.gray_image is None:
            return

        # 1. Try to retrieve the previous view limits
        current_xlim = None
        current_ylim = None
        
        if preserve_view:
            # Prefer the current live view of the image (the user may have just dragged it without updating view_limits yet)
            if self.ax_img.has_data():
                 current_xlim = self.ax_img.get_xlim()
                 current_ylim = self.ax_img.get_ylim()
            # Otherwise, use the cached view_limits
            elif self.view_limits is not None:
                 current_xlim, current_ylim = self.view_limits

        try:
            ratio = float(self.ratio_entry.get())
            min_size = float(self.min_size_entry.get())
            max_size = float(self.max_size_entry.get())
            thresh_val = self.thresh_scale.get()
            self.pixel_ratio = ratio
        except ValueError:
            return 

        diameters, contours, centers, binary_img = self.detect_particles(
            self.gray_image, ratio, min_size, max_size, thresh_val
        )
        self.detected_particles = diameters
        self.detected_contours = contours
        self.contour_centers = centers

        h, w = self.gray_image.shape[:2]
        total_area_um2 = h * w * (ratio ** 2)
        particles_area_um2 = np.sum([np.pi * (d/2)**2 for d in diameters])
        if total_area_um2 > 0:
            self.current_coverage = particles_area_um2 / total_area_um2
        else:
            self.current_coverage = 0.0

        # --- Plotting ---
        self.ax_img.clear()
        self.ax_hist.clear()

        display_img = self.current_image.copy()
        
        contour_color = (0, 255, 0) if not self.is_tracking_mode else (0, 255, 255) 
        cv2.drawContours(display_img, contours, -1, contour_color, 1)
        
        self.ax_img.imshow(display_img)
        
        if self.tracks:
            colors = ['red', 'blue', 'orange', 'magenta', 'cyan', 'yellow', 'green', 'purple']
            for tid, frames in self.tracks.items():
                color = colors[tid % len(colors)]
                
                sorted_fids = sorted(frames.keys())
                px_coords = [(frames[fid]['x_px'], frames[fid]['y_px']) for fid in sorted_fids]
                
                if len(px_coords) > 1:
                    px_coords = np.array(px_coords)
                    self.ax_img.plot(px_coords[:, 0], px_coords[:, 1], '-', color=color, linewidth=1.5, alpha=0.7)
                
                if self.current_index in frames:
                    curr_pt = frames[self.current_index]
                    marker = '*' if tid == self.active_track_id else 'o'
                    size = 12 if tid == self.active_track_id else 6
                    self.ax_img.plot(curr_pt['x_px'], curr_pt['y_px'], marker, color=color, markersize=size, markeredgecolor='white')
                    self.ax_img.text(curr_pt['x_px']+5, curr_pt['y_px']+5, f"T{tid}", color=color, fontsize=9, fontweight='bold')

        title_suffix = f" | 追踪模式: {'ON' if self.is_tracking_mode else 'OFF'}"
        self.ax_img.set_title(f"检测: {len(diameters)} 颗粒 | 覆盖率: {self.current_coverage:.2%}{title_suffix}")

        # --- Restore View ---
        # Must be set explicitly; otherwise imshow will reset it
        if current_xlim is not None and current_ylim is not None:
            self.ax_img.set_xlim(current_xlim)
            self.ax_img.set_ylim(current_ylim)
            self.view_limits = (current_xlim, current_ylim)

        # --- Histogram ---
        if len(diameters) > 0:
            n, bins, patches = self.ax_hist.hist(diameters, bins=20, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='数据分布')
            self.ax_hist.set_xlabel('直径 (μm)')
            self.ax_hist.set_ylabel('概率密度')

            fit_type = self.fit_var.get()
            if fit_type != "None" and len(diameters) > 5:
                try:
                    x_fit = np.linspace(min(bins), max(bins), 100)
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    
                    if fit_type == "Gaussian":
                        popt, _ = curve_fit(gaussian, bin_centers, n, p0=[1, np.mean(diameters), np.std(diameters)])
                        y_fit = gaussian(x_fit, *popt)
                        label_str = f'高斯: $\mu$={popt[1]:.2f}, $\sigma$={popt[2]:.2f}'
                    elif fit_type == "Log-Normal":
                        mu_init = np.log(np.mean(diameters))
                        sigma_init = 0.5
                        popt, _ = curve_fit(log_normal, bin_centers, n, p0=[1, mu_init, sigma_init], maxfev=5000)
                        y_fit = log_normal(x_fit, *popt)
                        label_str = f'对数正态: $\mu$={popt[1]:.2f}, $\sigma$={popt[2]:.2f}'
                    
                    self.ax_hist.plot(x_fit, y_fit, 'r-', linewidth=2, label=label_str)
                except Exception as e:
                    pass
            self.ax_hist.legend()
        else:
            self.ax_hist.set_xlabel('直径 (μm)')
            self.ax_hist.set_ylabel('概率密度')
            self.ax_hist.text(0.5, 0.5, "未检测到颗粒", ha='center', va='center', transform=self.ax_hist.transAxes)

        self.canvas.draw_idle() # Use draw_idle to improve responsiveness

    # --- Batch Export Function ---

    def export_data(self):
        if not self.image_paths:
            messagebox.showwarning("警告", "没有可处理的图像")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")])
        if not save_path:
            return

        progress_win = tk.Toplevel(self.root)
        progress_win.title("处理中")
        progress_label = ttk.Label(progress_win, text="正在批量分析所有图像，请稍候...")
        progress_label.pack(padx=20, pady=20)
        self.root.update()

        try:
            ratio = float(self.ratio_entry.get())
            min_size = float(self.min_size_entry.get())
            max_size = float(self.max_size_entry.get())
            thresh_val = self.thresh_scale.get()
            dt = float(self.time_entry.get())

            all_data = []

            for i, path in enumerate(self.image_paths):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None: continue
                
                h, w = img.shape[:2]
                total_area_um2 = h * w * (ratio ** 2)

                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if img.dtype == 'uint16':
                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                
                diameters, _, _, _ = self.detect_particles(img, ratio, min_size, max_size, thresh_val)
                
                particles_area_sum = np.sum([np.pi * (d/2)**2 for d in diameters])
                coverage_ratio = particles_area_sum / total_area_um2 if total_area_um2 > 0 else 0
                time_point = i * dt
                
                if len(diameters) == 0:
                    all_data.append({
                        "Filename": os.path.basename(path),
                        "Frame_Index": i,
                        "Time_s": time_point,
                        "Diameter_um": np.nan,
                        "Area_um2": 0,
                        "Coverage_Ratio": 0
                    })
                else:
                    for d in diameters:
                        all_data.append({
                            "Filename": os.path.basename(path),
                            "Frame_Index": i,
                            "Time_s": time_point,
                            "Diameter_um": d,
                            "Area_um2": np.pi * (d/2)**2,
                            "Coverage_Ratio": coverage_ratio
                        })

            df = pd.DataFrame(all_data)
            if save_path.endswith('.xlsx'):
                df.to_excel(save_path, index=False)
            else:
                df.to_csv(save_path, index=False)
            
            progress_win.destroy()
            messagebox.showinfo("成功", f"数据已导出至: {save_path}")

        except Exception as e:
            progress_win.destroy()
            messagebox.showerror("错误", f"导出过程中出错: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ParticleAnalyzerApp(root)
    root.mainloop()