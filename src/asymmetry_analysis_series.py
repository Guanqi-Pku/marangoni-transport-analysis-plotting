import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Rectangle, Circle
from skimage import io, measure, filters
from scipy.signal import correlate2d
from scipy.ndimage import center_of_mass
import pandas as pd
import os

class TimeSeriesAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Time-Series Asymmetry Tracker (4D ROI)")
        self.root.geometry("1400x950")

        # --- Data State ---
        self.stack_rgb = None     # (T, H, W, 3)
        self.stack_magenta = None # (T, H, W)
        self.stack_green = None   # (T, H, W)
        self.n_frames = 0
        self.current_frame = 0
        
        # Tracking Data: Dictionary mapping frame_index -> {'bbox': [r,c,h,w], 'results': dict}
        self.frame_data = {} 
        
        # --- Interaction State ---
        self.mode = "SELECT"      # SELECT, DRAW
        self.interaction_state = None
        self.drag_start = None
        self.last_mouse_pos = None

        self.setup_ui()

    def setup_ui(self):
        # 1. Left Control Panel
        self.panel_control = tk.Frame(self.root, width=380, bg="#f5f5f5", padx=15, pady=15)
        self.panel_control.pack(side=tk.LEFT, fill=tk.Y)
        
        # 2. Right Canvas Area
        self.panel_right = tk.Frame(self.root)
        self.panel_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.create_matplotlib_canvas()
        self.create_controls()

    def create_controls(self):
        # Title
        tk.Label(self.panel_control, text="Time-Series Tracker", font=("Arial", 18, "bold"), bg="#f5f5f5").pack(pady=(0, 20))

        # --- Step 1: Input ---
        frame_input = tk.LabelFrame(self.panel_control, text="1. Input & Calibration", bg="#f5f5f5", font=("Arial", 10, "bold"))
        frame_input.pack(fill=tk.X, pady=8, ipady=3)
        
        btn_load = tk.Button(frame_input, text="Load TIF Stack", command=self.load_image, bg="#e0e0e0", font=("Arial", 10), height=2)
        btn_load.pack(fill=tk.X, padx=10, pady=8)
        
        frame_cal = tk.Frame(frame_input, bg="#f5f5f5")
        frame_cal.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(frame_cal, text="Pixel Size (µm):", bg="#f5f5f5").pack(side=tk.LEFT)
        self.entry_px = tk.Entry(frame_cal, width=8)
        self.entry_px.insert(0, "0.44")
        self.entry_px.pack(side=tk.RIGHT)
        
        tk.Label(frame_cal, text="Interval (s):", bg="#f5f5f5").pack(side=tk.LEFT, padx=(15,0))
        self.entry_dt = tk.Entry(frame_cal, width=6)
        self.entry_dt.insert(0, "5.0")
        self.entry_dt.pack(side=tk.RIGHT)

        # --- Step 2: ROI & Tracking ---
        # Increased size and padding for better usability
        frame_roi = tk.LabelFrame(self.panel_control, text="2. ROI & Tracking", bg="#f5f5f5", font=("Arial", 11, "bold"), fg="blue")
        frame_roi.pack(fill=tk.X, pady=15, ipady=5)

        # Padding Adjustment
        frame_pad = tk.Frame(frame_roi, bg="#f5f5f5")
        frame_pad.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(frame_pad, text="ROI Padding (px):", bg="#f5f5f5", font=("Arial", 9)).pack(side=tk.LEFT)
        self.scale_pad = tk.Scale(frame_pad, from_=0, to=50, orient=tk.HORIZONTAL, bg="#f5f5f5", length=150)
        self.scale_pad.set(10)
        self.scale_pad.pack(side=tk.RIGHT)

        self.btn_draw = tk.Button(frame_roi, text="Draw ROI (Current Frame)", command=self.toggle_draw_mode, bg="white", font=("Arial", 10), height=2)
        self.btn_draw.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(frame_roi, text="Auto Tracking:", bg="#f5f5f5", font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=(10,5))
        
        # Bigger Buttons for Tracking
        btn_auto = tk.Button(frame_roi, text="Auto Detect\n(Current Frame)", command=self.auto_detect_current, bg="#add8e6", font=("Arial", 10), height=2)
        btn_auto.pack(fill=tk.X, padx=10, pady=2)
        
        btn_track = tk.Button(frame_roi, text="Track Forward >>\n(All Frames)", command=self.track_forward, bg="#ffd700", font=("Arial", 10, "bold"), height=2)
        btn_track.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(frame_roi, text="(Set ROI on Frame 0 -> Click Track Forward)", bg="#f5f5f5", fg="gray", font=("Arial", 8)).pack(pady=2)

        # --- Step 3: Analysis ---
        frame_algo = tk.LabelFrame(self.panel_control, text="3. Algorithm", bg="#f5f5f5", font=("Arial", 10, "bold"))
        frame_algo.pack(fill=tk.X, pady=8)
        
        self.algo_var = tk.StringVar(value="CCF")
        tk.Radiobutton(frame_algo, text="Cross Correlation (CCF)", variable=self.algo_var, value="CCF", bg="#f5f5f5", anchor="w").pack(fill=tk.X, padx=10)
        tk.Radiobutton(frame_algo, text="Center of Mass (COM)", variable=self.algo_var, value="COM", bg="#f5f5f5", anchor="w").pack(fill=tk.X, padx=10)

        btn_calc = tk.Button(frame_algo, text="Calculate Vectors (All Frames)", command=self.run_analysis, bg="#90ee90", font=("Arial", 10, "bold"), height=2)
        btn_calc.pack(fill=tk.X, padx=10, pady=8)

        # --- Step 4: Export ---
        frame_export = tk.LabelFrame(self.panel_control, text="4. Output", bg="#f5f5f5", font=("Arial", 10, "bold"))
        frame_export.pack(fill=tk.X, pady=8)
        
        self.var_composite = tk.BooleanVar(value=True)
        chk_comp = tk.Checkbutton(frame_export, text="Generate Time-Projection\n(One composite image)", variable=self.var_composite, bg="#f5f5f5", justify=tk.LEFT)
        chk_comp.pack(anchor="w", padx=10, pady=5)
        
        self.btn_save = tk.Button(frame_export, text="Save Data & Images", command=self.save_results, state=tk.DISABLED, font=("Arial", 10), height=2)
        self.btn_save.pack(fill=tk.X, padx=10, pady=8)

        # --- Status ---
        self.lbl_status = tk.Label(self.panel_control, text="Ready.", bd=1, relief=tk.SUNKEN, anchor=tk.W, font=("Arial", 9))
        self.lbl_status.pack(side=tk.BOTTOM, fill=tk.X)

    def create_matplotlib_canvas(self):
        # Frame container for canvas + slider
        frame_vis = tk.Frame(self.panel_right)
        frame_vis.pack(fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.fig.patch.set_facecolor('#f0f0f0')
        self.ax.axis('off')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_vis)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, frame_vis)
        self.toolbar.update()

        # Time Slider
        self.frame_slider = tk.Scale(frame_vis, from_=0, to=0, orient=tk.HORIZONTAL, label="Time Point (Frame)", 
                                     command=self.on_slider_change, length=600)
        self.frame_slider.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=10)

        # Events
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.canvas.mpl_connect('button_release_event', self.on_release)

    # --- Core Logic ---

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("TIFF Stack", "*.tif *.tiff")])
        if not path: return
        
        try:
            # Load Stack
            img = io.imread(path)
            # Normalize shape to (T, H, W, C)
            if img.ndim == 3: # Maybe (T, H, W) grayscale or (H, W, C) single
                if img.shape[2] == 3: # Single RGB
                    img = img[np.newaxis, ...]
                else: # Grayscale stack
                    # Convert to Pseudo-RGB for consistency
                    img_stack = []
                    for i in range(img.shape[0]):
                        sl = img[i]
                        img_stack.append(np.stack([sl, sl, sl], axis=-1))
                    img = np.array(img_stack)
            
            self.stack_rgb = img
            self.n_frames = img.shape[0]
            
            # Split Channels (Assume Magenta=R+B, Green=G)
            self.stack_magenta = (self.stack_rgb[..., 0].astype(float) + self.stack_rgb[..., 2].astype(float)) / 2.0
            self.stack_green = self.stack_rgb[..., 1].astype(float)
            
            # Reset Data
            self.frame_data = {}
            self.current_frame = 0
            
            # UI Update
            self.frame_slider.config(to=self.n_frames - 1, label=f"Time Point (Frame 0 - {self.n_frames-1})")
            self.frame_slider.set(0)
            self.display_current_frame()
            self.lbl_status.config(text=f"Loaded: {self.n_frames} frames.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {e}")

    def on_slider_change(self, val):
        self.current_frame = int(val)
        self.display_current_frame()

    def display_current_frame(self):
        if self.stack_rgb is None: return
        
        self.ax.clear()
        self.ax.axis('off')
        
        # Display Image
        self.ax.imshow(self.stack_rgb[self.current_frame])
        self.ax.set_title(f"Frame: {self.current_frame + 1} / {self.n_frames}")

        # Draw ROI for CURRENT frame
        if self.current_frame in self.frame_data:
            data = self.frame_data[self.current_frame]
            r, c, h, w = data['bbox']
            
            # Box
            rect = Rectangle((c, r), w, h, linewidth=2, edgecolor='lime', facecolor='none')
            self.ax.add_patch(rect)
            
            # Vector (if calculated)
            if 'vector' in data:
                dy, dx = data['vector']
                cy, cx = data['center']
                scale = 10.0 # Visual scale
                self.ax.arrow(cx, cy, dx*scale, dy*scale, color='cyan', width=0.5, head_width=2, length_includes_head=True)
                self.ax.plot(cx, cy, 'm.', markersize=4)

        # Draw Trajectory (History up to current frame)
        frame_indices = sorted([f for f in self.frame_data.keys() if f <= self.current_frame])
        if len(frame_indices) > 1:
            centers_y = [self.frame_data[f]['center'][0] for f in frame_indices if 'center' in self.frame_data[f]]
            centers_x = [self.frame_data[f]['center'][1] for f in frame_indices if 'center' in self.frame_data[f]]
            if centers_x:
                self.ax.plot(centers_x, centers_y, 'w--', linewidth=1, alpha=0.7)
                self.ax.plot(centers_x[-1], centers_y[-1], 'wo', markersize=3) # Current dot

        self.canvas.draw()

    # --- Interaction (Draw/Edit ROI) ---

    def toggle_draw_mode(self):
        if self.mode == "SELECT":
            self.mode = "DRAW"
            self.btn_draw.config(bg="yellow", text="Drawing Mode...")
        else:
            self.mode = "SELECT"
            self.btn_draw.config(bg="white", text="Draw ROI")

    def on_press(self, event):
        if event.inaxes != self.ax: return
        self.drag_start = (event.xdata, event.ydata)
        self.last_mouse_pos = (event.xdata, event.ydata)
        
        if self.mode == "SELECT" and self.current_frame in self.frame_data:
            # Check if clicked inside existing ROI to move
            r, c, h, w = self.frame_data[self.current_frame]['bbox']
            if c <= event.xdata <= c+w and r <= event.ydata <= r+h:
                self.interaction_state = "MOVING"

    def on_drag(self, event):
        if not event.inaxes: return
        
        if self.mode == "DRAW" and self.drag_start:
            # Visual feedback for drawing
            self.display_current_frame()
            x0, y0 = self.drag_start
            w = event.xdata - x0
            h = event.ydata - y0
            rect = Rectangle((x0, y0), w, h, linewidth=1, edgecolor='yellow', linestyle='--')
            self.ax.add_patch(rect)
            self.canvas.draw()

        elif self.interaction_state == "MOVING":
            dx = event.xdata - self.last_mouse_pos[0]
            dy = event.ydata - self.last_mouse_pos[1]
            self.last_mouse_pos = (event.xdata, event.ydata)
            
            bbox = self.frame_data[self.current_frame]['bbox']
            bbox[0] += dy
            bbox[1] += dx
            # Delete results if moved
            if 'vector' in self.frame_data[self.current_frame]:
                del self.frame_data[self.current_frame]['vector']
            self.display_current_frame()

    def on_release(self, event):
        if self.mode == "DRAW" and self.drag_start and event.xdata:
            x0, y0 = self.drag_start
            x1, y1 = event.xdata, event.ydata
            
            r = int(min(y0, y1))
            c = int(min(x0, x1))
            h = int(abs(y1 - y0))
            w = int(abs(x1 - x0))
            
            if h > 5 and w > 5:
                # Save ROI for this frame
                self.frame_data[self.current_frame] = {
                    'bbox': [r, c, h, w]
                }
                self.mode = "SELECT"
                self.btn_draw.config(bg="white", text="Draw ROI")
                self.display_current_frame()

        self.interaction_state = None
        self.drag_start = None

    # --- Tracking & Analysis ---

    def auto_detect_current(self):
        """Simple Otsu threshold + Largest Object for current frame"""
        if self.stack_magenta is None: return
        
        img = self.stack_magenta[self.current_frame]
        val = filters.threshold_otsu(img)
        binary = img > val
        labeled = measure.label(binary)
        props = measure.regionprops(labeled)
        
        if not props: return
        
        # Pick largest
        largest = max(props, key=lambda x: x.area)
        minr, minc, maxr, maxc = largest.bbox
        
        # Add padding (User Adjustable)
        pad = self.scale_pad.get()
        
        r = max(0, minr - pad)
        c = max(0, minc - pad)
        h = (maxr - minr) + 2*pad
        w = (maxc - minc) + 2*pad
        
        self.frame_data[self.current_frame] = {'bbox': [r, c, h, w]}
        self.display_current_frame()

    def track_forward(self):
        """Track the object from current_frame to end"""
        if self.current_frame not in self.frame_data:
            messagebox.showwarning("Warning", "Define ROI on current frame first.")
            return

        start_f = self.current_frame
        curr_bbox = self.frame_data[start_f]['bbox'] # [r, c, h, w]
        img_h, img_w = self.stack_magenta.shape[1], self.stack_magenta.shape[2]
        pad = self.scale_pad.get() # User Adjustable Padding
        
        padding_search = pad + 10 # Look a bit outside previous box
        
        for t in range(start_f + 1, self.n_frames):
            prev_r, prev_c, prev_h, prev_w = map(int, curr_bbox)
            
            # Define search region (previous ROI + margin)
            sr_r = max(0, prev_r - padding_search)
            sr_c = max(0, prev_c - padding_search)
            sr_rh = min(img_h, prev_r + prev_h + padding_search)
            sr_cw = min(img_w, prev_c + prev_w + padding_search)
            
            roi_img = self.stack_magenta[t, sr_r:sr_rh, sr_c:sr_cw]
            
            if roi_img.size == 0: break
            
            # Simple Centroid in neighborhood
            try:
                thresh = filters.threshold_otsu(roi_img)
                binary = roi_img > thresh
                labeled = measure.label(binary)
                props = measure.regionprops(labeled)
                
                if props:
                    # Find object closest to center of search window
                    center_sr = (roi_img.shape[0]/2, roi_img.shape[1]/2)
                    best_prop = min(props, key=lambda p: (p.centroid[0]-center_sr[0])**2 + (p.centroid[1]-center_sr[1])**2)
                    
                    minr, minc, maxr, maxc = best_prop.bbox
                    
                    # New global coords
                    new_r = sr_r + minr - pad
                    new_c = sr_c + minc - pad
                    new_h = (maxr - minr) + 2*pad
                    new_w = (maxc - minc) + 2*pad
                    
                    # Safety Bounds
                    new_r = max(0, min(new_r, img_h - 1))
                    new_c = max(0, min(new_c, img_w - 1))
                    
                    curr_bbox = [new_r, new_c, new_h, new_w]
                    self.frame_data[t] = {'bbox': curr_bbox}
                else:
                    # Keep previous if lost
                    self.frame_data[t] = {'bbox': curr_bbox}
            except:
                 self.frame_data[t] = {'bbox': curr_bbox}

        self.lbl_status.config(text=f"Tracking complete from frame {start_f} to {self.n_frames-1}")
        self.run_analysis() # Auto run calc after track

    def run_analysis(self):
        """Calculate vector for all frames with ROI"""
        px_size = float(self.entry_px.get())
        algo = self.algo_var.get()
        
        count = 0
        for t, data in self.frame_data.items():
            r, c, h, w = map(int, data['bbox'])
            
            # Boundary checks
            img_h, img_w = self.stack_magenta.shape[1:]
            if r >= img_h or c >= img_w: continue
            
            crop_m = self.stack_magenta[t, r:r+h, c:c+w]
            crop_g = self.stack_green[t, r:r+h, c:c+w]
            
            if crop_m.size == 0: continue
            
            # 1. Magenta Center (Structure)
            try:
                cy_loc, cx_loc = center_of_mass(crop_m)
            except:
                cy_loc, cx_loc = h/2, w/2
                
            cy_glob = r + cy_loc
            cx_glob = c + cx_loc
            data['center'] = (cy_glob, cx_glob)

            # --- NEW: Radius of Gyration (Rg) for Magenta ---
            y_grid, x_grid = np.indices(crop_m.shape)
            sq_dist = (y_grid - cy_loc)**2 + (x_grid - cx_loc)**2
            sum_intensity = np.sum(crop_m)
            if sum_intensity > 0:
                Rg_px = np.sqrt(np.sum(crop_m * sq_dist) / sum_intensity)
            else:
                Rg_px = 0
            
            data['rg_um'] = Rg_px * px_size
            # ------------------------------------------------

            # 2. Green Vector
            dy, dx = 0.0, 0.0
            
            if algo == "COM":
                if np.sum(crop_g) > 0:
                    g_cy, g_cx = center_of_mass(crop_g)
                    dy = g_cy - cy_loc
                    dx = g_cx - cx_loc
            
            elif algo == "CCF":
                cm_z = crop_m - crop_m.mean()
                cg_z = crop_g - crop_g.mean()
                # Use faster fft convolution
                corr = correlate2d(cg_z, cm_z, boundary='symm', mode='same')
                y_peak, x_peak = np.unravel_index(np.argmax(corr), corr.shape)
                
                mid_y, mid_x = corr.shape[0]//2, corr.shape[1]//2
                dy = y_peak - mid_y
                dx = x_peak - mid_x
            
            mag_um = np.sqrt(dx**2 + dy**2) * px_size
            data['vector'] = (dy, dx)
            data['mag_um'] = mag_um
            count += 1
            
        self.btn_save.config(state=tk.NORMAL)
        self.lbl_status.config(text=f"Calculated vectors for {count} frames.")
        self.display_current_frame()

    def save_results(self):
        save_dir = filedialog.askdirectory(title="Select Folder to Save Results")
        if not save_dir: return

        frames = sorted(self.frame_data.keys())
        rows = []
        dt = float(self.entry_dt.get())
        px = float(self.entry_px.get())

        # Create subfolder for images
        img_dir = os.path.join(save_dir, "roi_images")
        os.makedirs(img_dir, exist_ok=True)

        for i, t in enumerate(frames):
            d = self.frame_data[t]
            if 'vector' not in d: continue
            
            # --- 1. Data Collection ---
            dy, dx = d['vector']
            cy, cx = d['center']
            
            # --- NEW: Angle Calculation relative to Movement ---
            angle_rel = np.nan
            
            # We need the NEXT frame to determine movement direction
            if i < len(frames) - 1:
                next_t = frames[i+1]
                next_d = self.frame_data[next_t]
                if 'center' in next_d:
                    n_cy, n_cx = next_d['center']
                    # Movement Vector: Next - Current
                    move_y = n_cy - cy
                    move_x = n_cx - cx
                    
                    # Angles in degrees (-180 to 180)
                    angle_move = np.degrees(np.arctan2(move_y, move_x))
                    angle_asym = np.degrees(np.arctan2(dy, dx))
                    
                    # Relative Angle: Asymmetry - Movement
                    # Normalize to [-180, 180]
                    diff = angle_asym - angle_move
                    angle_rel = (diff + 180) % 360 - 180
            
            rows.append({
                'Frame': t,
                'Time_s': t * dt,
                'Centroid_X_px': cx,
                'Centroid_Y_px': cy,
                'Vector_X_um': dx * px,
                'Vector_Y_um': dy * px,
                'Magnitude_um': d['mag_um'],
                'Angle_Global_Deg': np.degrees(np.arctan2(dy, dx)),
                'Angle_vs_Movement_Deg': angle_rel,  # <-- Added
                'Rg_Magenta_um': d.get('rg_um', 0)   # <-- Added
            })

            # --- 2. Image Generation (Black Background ROI - Individual) ---
            # Create a dedicated figure for saving to avoid messing up GUI
            fig_save, ax_save = plt.subplots(figsize=(4, 4))
            
            # Black background
            full_h, full_w = self.stack_magenta.shape[1:]
            black_bg = np.zeros((full_h, full_w, 3), dtype=np.uint8)
            
            # Fill only ROI
            r, c, h, w = map(int, d['bbox'])
            # Ensure bounds
            r_end, c_end = min(r+h, full_h), min(c+w, full_w)
            
            # Insert original RGB content into black canvas
            roi_content = self.stack_rgb[t, r:r_end, c:c_end, :]
            
            # Determine placement (handle edge cases)
            black_bg[r:r_end, c:c_end, :] = roi_content

            # Crop the view to the ROI plus a margin
            margin = 20
            view_r_min = max(0, r - margin)
            view_r_max = min(full_h, r_end + margin)
            view_c_min = max(0, c - margin)
            view_c_max = min(full_w, c_end + margin)
            
            crop_view = black_bg[view_r_min:view_r_max, view_c_min:view_c_max]
            
            ax_save.imshow(crop_view)
            ax_save.axis('off')
            
            # Draw Vector (Adjusted for crop)
            rel_cx = cx - view_c_min
            rel_cy = cy - view_r_min
            
            ax_save.arrow(rel_cx, rel_cy, dx*5, dy*5, color='cyan', width=1, head_width=3)
            ax_save.plot(rel_cx, rel_cy, 'm.')
            
            title_txt = f"T={t*dt:.1f}s | Mag={d['mag_um']:.2f}um"
            if not np.isnan(angle_rel):
                title_txt += f"\nRel.Angle={angle_rel:.1f}°"
            ax_save.set_title(title_txt, color='white', fontsize=8)
            
            # Save
            fig_save.savefig(os.path.join(img_dir, f"frame_{t:03d}.png"), facecolor='black', dpi=100)
            plt.close(fig_save)

        # --- 3. Composite Time-Projection Image ---
        if self.var_composite.get() and len(frames) > 0:
            full_h, full_w = self.stack_magenta.shape[1:]
            composite_img = np.zeros((full_h, full_w, 3), dtype=np.uint8)
            
            # Accumulate frames (Max Projection of ROIs)
            for t in frames:
                d = self.frame_data[t]
                r, c, h, w = map(int, d['bbox'])
                r_end, c_end = min(r+h, full_h), min(c+w, full_w)
                
                roi_data = self.stack_rgb[t, r:r_end, c:c_end, :]
                
                # Take Maximum (to overlay bright spots)
                composite_img[r:r_end, c:c_end, :] = np.maximum(
                    composite_img[r:r_end, c:c_end, :], 
                    roi_data
                )
            
            # Plot Composite
            fig_comp, ax_comp = plt.subplots(figsize=(10, 10))
            ax_comp.imshow(composite_img)
            ax_comp.axis('off')
            
            # Draw Full Trajectory
            all_cy = [self.frame_data[t]['center'][0] for t in frames]
            all_cx = [self.frame_data[t]['center'][1] for t in frames]
            ax_comp.plot(all_cx, all_cy, 'w--', linewidth=1, alpha=0.5, label="Trajectory")
            
            # Draw All Vectors
            for t in frames:
                d = self.frame_data[t]
                if 'vector' in d:
                    cy, cx = d['center']
                    dy, dx = d['vector']
                    # Use a slightly more visible color or scale for static image
                    ax_comp.arrow(cx, cy, dx*5, dy*5, color='cyan', width=0.5, head_width=2)
                    ax_comp.plot(cx, cy, 'm.', markersize=2)
            
            ax_comp.set_title("Time-Projection (Max ROI Intensity + Trajectory)", color='white', backgroundcolor='black')
            fig_comp.savefig(os.path.join(save_dir, "time_projection_composite.png"), facecolor='black', dpi=200)
            plt.close(fig_comp)

        # Save CSV
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(save_dir, "tracking_results.csv"), index=False)
        
        messagebox.showinfo("Success", f"Saved CSV and images to:\n{save_dir}\n\nComposite Image Generated: {self.var_composite.get()}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TimeSeriesAnalyzerApp(root)
    root.mainloop()