import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import random
import os

# Ensure required directories exist
os.makedirs('gt_image', exist_ok=True)
os.makedirs('gt_binary_image', exist_ok=True)
os.makedirs('gt_instance_image', exist_ok=True)


class AnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Công cụ chú thích ảnh")
        self.root.geometry("1000x700")

        # Main variables for the tool
        self.drawing = False
        self.mode = 'line'
        self.action_mode = 'label'  # 'label', 'pan', 'edit'
        self.edit_sub_mode = None  # 'delete', 'add'
        self.ix, self.iy = -1, -1
        self.annotations = []  # Now a list of dictionaries
        self.current_object = []
        self.curve_points = []
        self.current_color = (0, 0, 255)
        self.thickness = 2
        self.img = None
        self.binary_img = None
        self.gray_lane_img = None
        self.image_path_list = []
        self.index = 0
        self.img_path = None if not self.image_path_list else self.image_path_list[self.index]
        self.text = f"0/0"
        self.check_save = [False]*len(self.image_path_list)
        self.image_label = {}
        self.selected_object_index = None

        # Variables for zoom and pan
        self.zoom_scale = 1.0
        self.pan_x = 0
        self.pan_y = 0

        # Setup UI
        self.setup_ui()

        # Bind keyboard shortcuts
        self.bind_shortcuts()

    def setup_ui(self):
        # Toolbar
        toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        tk.Button(toolbar, text="Mở ảnh", command=self.open_folder_and_get_image_paths).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text="Lưu ảnh", command=self.save_images).pack(side=tk.LEFT, padx=2, pady=2)
        self.mode_button = tk.Button(toolbar, text="Chế độ vẽ: line", command=self.toggle_mode)
        self.mode_button.pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text="Hoàn thành đối tượng", command=self.finish_current_object).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text="Xóa gần nhất", command=self.undo_last_annotation).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text="Xóa tất cả", command=self.clear_all_annotations).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text="Tạo Train/Valid/Test", command=self.create_train_valid_test_files).pack(side=tk.LEFT, padx=2, pady=2)
        self.next_button = tk.Button(self.root, text="Next", command=self.next_image)
        self.next_button.pack(side=tk.RIGHT)

        self.prev_button = tk.Button(self.root, text="Previous", command=self.prev_image)
        self.prev_button.pack(side=tk.LEFT)
        # Label để hiển thị số lượng ảnh
        self.image_count_label = tk.Label(self.root, text=self.text, font=("Arial", 14))
        self.image_count_label.pack(side=tk.BOTTOM)

        # Checkbox để kiểm tra đã lưu
        self.save_check_var = tk.BooleanVar()
        self.save_check_box = tk.Checkbutton(self.root, text="Saved", variable=self.save_check_var, command=self.check_click_button)
        self.save_check_box.pack(side=tk.BOTTOM, anchor=tk.SE)

        toolbar.pack(side=tk.TOP, fill=tk.X)

        # Thickness slider
        thickness_slider = tk.Scale(self.root, from_=1, to=30, orient=tk.HORIZONTAL, label="Độ dày nét vẽ",
                                    command=self.update_thickness)
        thickness_slider.set(self.thickness)
        thickness_slider.pack(side=tk.TOP, fill=tk.X)

        # Canvas for image display
        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack()
        self.img_display = self.canvas.create_image(0, 0, anchor=tk.NW)

        # Mouse event bindings for the canvas
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Motion>", self.on_mouse_motion)

        self.action_mode_button = tk.Button(toolbar, text="Chế độ: Label", command=self.toggle_action_mode)
        self.action_mode_button.pack(side=tk.LEFT, padx=2, pady=2)

        # Nút chuyển đổi chế độ chỉnh sửa
        self.edit_mode_button = tk.Button(toolbar, text="Chỉnh sửa: None", command=self.toggle_edit_sub_mode, state=tk.DISABLED)
        self.edit_mode_button.pack(side=tk.LEFT, padx=2, pady=2)

    def bind_shortcuts(self):
        # Bind keyboard shortcuts to functions
        self.root.bind('<Control-o>', self.open_folder_and_get_image_paths)
        self.root.bind('<Control-s>', self.save_images)
        self.root.bind('m', self.toggle_mode)
        self.root.bind('<Tab>', self.toggle_action_mode)
        self.root.bind('<Return>', self.finish_current_object)
        self.root.bind('z', self.undo_last_annotation)
        self.root.bind('c', self.clear_all_annotations)
        self.root.bind('t', self.create_train_valid_test_files)
        self.root.bind('n', self.next_image)
        self.root.bind('p', self.prev_image)
        self.root.bind('d', self.set_edit_sub_mode_delete)
        self.root.bind('a', self.set_edit_sub_mode_add)

    def set_edit_sub_mode_delete(self, event=None):
        if self.action_mode == 'edit':
            self.edit_sub_mode = 'delete'
            self.edit_mode_button.config(text="Chỉnh sửa: Xóa đối tượng")
            self.selected_object_index = None
            self.update_display()

    def set_edit_sub_mode_add(self, event=None):
        if self.action_mode == 'edit':
            self.edit_sub_mode = 'add'
            self.edit_mode_button.config(text="Chỉnh sửa: Thêm vào đối tượng")
            self.selected_object_index = None
            self.update_display()

    def toggle_edit_sub_mode(self, event=None):
        if self.action_mode == 'edit':
            if self.edit_sub_mode == 'delete':
                self.edit_sub_mode = 'add'
            else:
                self.edit_sub_mode = 'delete'
            self.edit_mode_button.config(text=f"Chỉnh sửa: {'Xóa đối tượng' if self.edit_sub_mode == 'delete' else 'Thêm vào đối tượng'}")
            self.selected_object_index = None
            self.update_display()

    def create_instance_image(self, binary_image):
        # Tạo một instance image với cùng kích thước như binary_image
        instance_image = np.zeros_like(binary_image)

        # Tìm các thành phần liên thông
        num_labels, labels_im = cv2.connectedComponents(binary_image.astype(np.uint8))

        # Gán nhãn cho từng thành phần
        for label in range(1, num_labels):
            instance_image[labels_im == label] = label  # Gán nhãn cho các pixel tương ứng

        return instance_image

    def open_folder_and_get_image_paths(self, event=None):
        # Hiển thị hộp thoại để chọn thư mục
        folder_path = filedialog.askdirectory()

        if folder_path:
            # Lọc các tệp hình ảnh có phần mở rộng phù hợp
            valid_extensions = ('.png', '.jpg', '.jpeg')
            self.image_path_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path)
                           if file.lower().endswith(valid_extensions)]

        self.text = f"{self.index+1} / {len(self.image_path_list)}"
        self.image_count_label.config(text=self.text)

        self.img_path = self.image_path_list[self.index]
        self.check_save = [False]*len(self.image_path_list)

        if self.img_path:
            self.img = cv2.imread(self.img_path)
            if self.img is not None:
                self.img = cv2.resize(self.img, (800, 600))
                # Khởi tạo canvas nhị phân và canvas phân biệt lane cùng kích thước
                self.binary_img = np.zeros((600, 800), dtype=np.uint8)
                self.gray_lane_img = np.zeros((600, 800), dtype=np.uint8)
                self.update_display()

    def check_click_button(self):
        if not self.save_check_var.get():
            self.check_save[self.index] = False
        else:
            self.check_save[self.index] = True

    def next_image(self, event=None):
        if self.index < len(self.image_path_list) - 1:
            self.index += 1
            self.img_path = self.image_path_list[self.index]
            if not self.check_save[self.index]:
                self.save_check_var.set(False)
            else:
                self.save_check_var.set(True)
            self.img = cv2.imread(self.img_path)
            self.img = cv2.resize(self.img, (800, 600))
            self.binary_img = np.zeros((600, 800), dtype=np.uint8)
            self.gray_lane_img = np.zeros((600, 800), dtype=np.uint8)
            self.annotations.clear()
            self.current_object.clear()
            # Reset zoom and pan
            self.zoom_scale = 1.0
            self.pan_x = 0
            self.pan_y = 0
            self.update_display()
        else:
            messagebox.showerror("Lỗi", "Không thể mở ảnh. Vui lòng thử lại.")
        self.text = f"{self.index+1} / {len(self.image_path_list)}"
        self.image_count_label.config(text=self.text)

    def prev_image(self, event=None):
        if self.index > 0:
            self.index -= 1
            self.img_path = self.image_path_list[self.index]
            if not self.check_save[self.index]:
                self.save_check_var.set(False)
            else:
                self.save_check_var.set(True)
            self.annotations.clear()
            self.current_object.clear()
            self.img = cv2.imread(self.img_path)
            self.img = cv2.resize(self.img, (800, 600))
            self.binary_img = np.zeros((600, 800), dtype=np.uint8)
            self.gray_lane_img = np.zeros((600, 800), dtype=np.uint8)
            # Reset zoom and pan
            self.zoom_scale = 1.0
            self.pan_x = 0
            self.pan_y = 0
            self.update_display()
        else:
            messagebox.showerror("Lỗi", "Không thể mở ảnh. Vui lòng thử lại.")
        self.text = f"{self.index+1} / {len(self.image_path_list)}"
        self.image_count_label.config(text=self.text)

    def save_images(self, event=None):
        while True:
            base_name = os.path.splitext(os.path.basename(self.image_path_list[self.index]))[0]

            # Định nghĩa các đường dẫn để lưu ảnh
            image_path = f"gt_image/{base_name}.png"
            binary_path = f"gt_binary_image/{base_name}.png"
            gray_lane_path = f"gt_instance_image/{base_name}.png"

            # Kiểm tra xem bất kỳ tệp nào đã tồn tại
            if os.path.exists(image_path) or os.path.exists(binary_path) or os.path.exists(gray_lane_path):
                # Hỏi người dùng nhập tên mới
                retry = messagebox.askyesno("Đã lưu",
                                            f"File '{base_name}' đã tồn tại. Ghi đè?")
                if not retry:
                    return
            self.image_label[self.index] = self.img
                # Save the annotated original image
            img = cv2.imread(self.img_path)
            img = cv2.resize(img, (800, 600))
            if self.img is not None:
                cv2.imwrite(image_path, img)

            # Save the binary image
            if self.binary_img is not None:
                cv2.imwrite(binary_path, self.binary_img)

                # Save the instance (gray lane) image
            # self.gray_lane_img = self.create_instance_image(self.binary_img)
            if self.gray_lane_img is not None:
                cv2.imwrite(gray_lane_path, self.gray_lane_img)

            messagebox.showinfo("Lưu ảnh", f"Đã lưu ảnh dưới tên:\n{image_path}\n{binary_path}\n{gray_lane_path}")
            self.check_save[self.index] = True
            self.save_check_var.set(True)
            break

    def create_train_valid_test_files(self, event=None):
        # Ask for base path
        base_path = simpledialog.askstring("Nhập đường dẫn",
                                           "Nhập đường dẫn cơ sở cho các tệp (ví dụ: dataset/dataset_tusimple/training):")
        if not base_path:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập đường dẫn cơ sở.")
            return

        # Get n_sample for the test set
        n_sample = simpledialog.askinteger("Nhập n_sample", "Số lượng ảnh trong tập val:")
        if not n_sample:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập số lượng n_sample.")
            return

        # Get list of image base names from 'gt_image' folder
        img_files = sorted([f.split('.')[0] for f in os.listdir('gt_image') if f.endswith('.png')])
        print(img_files)

        if len(img_files) < n_sample:
            messagebox.showerror("Lỗi", "Số lượng ảnh trong thư mục ít hơn số lượng yêu cầu cho test.")
            return

        # Split data
        train_files = img_files

        # Write to files
        self.write_to_txt('train.txt', train_files, base_path)

        # Random selection for test set
        test_files = random.sample(img_files, n_sample)
        self.write_to_txt('val.txt', test_files, base_path)

        messagebox.showinfo("Hoàn tất", "Các tệp train.txt, val.txt đã được tạo.")

    def write_to_txt(self, filename, files_list, base_path):
        with open(filename, 'w') as f:
            for file_base in files_list:
                image_path = f"{base_path}/gt_image/{file_base}.png"
                binary_path = f"{base_path}/gt_binary_image/{file_base}.png"
                instance_path = f"{base_path}/gt_instance_image/{file_base}.png"
                f.write(f"{image_path} {binary_path} {instance_path}\n")

    def random_color(self):
        return tuple(random.randint(0, 255) for _ in range(3))

    def random_gray(self):
        gray_value = random.randint(50, 200)  # Chọn một giá trị xám ngẫu nhiên (giữa 50 và 200 để dễ phân biệt)
        return gray_value

    def open_image(self, event=None):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.img_path = file_path
            self.img = cv2.imread(file_path)
            if self.img is not None:
                self.img = cv2.resize(self.img, (800, 600))
                # Khởi tạo canvas nhị phân và canvas phân biệt lane cùng kích thước
                self.binary_img = np.zeros((600, 800), dtype=np.uint8)
                self.gray_lane_img = np.zeros((600, 800), dtype=np.uint8)
                self.update_display()
            else:
                messagebox.showerror("Lỗi", "Không thể mở ảnh. Vui lòng thử lại.")

    def update_display(self):
        if self.img is not None:
            # Thay đổi kích thước ảnh theo zoom_scale
            new_width = int(800 * self.zoom_scale)
            new_height = int(600 * self.zoom_scale)
            img_resized = cv2.resize(self.img, (new_width, new_height))
            # Tạo bản sao để vẽ các chú thích
            img_display = img_resized.copy()
            # Vẽ các đối tượng
            for idx, obj in enumerate(self.annotations):
                # Nếu đối tượng đang được chọn, vẽ với màu đặc biệt
                if idx == self.selected_object_index:
                    color = (0, 255, 255)  # Màu vàng
                else:
                    color = None
                self.draw_object_on_image(obj, img_display, color)
            # Chuyển đổi ảnh từ BGR sang RGB
            img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
            # Tạo ảnh PIL từ mảng NumPy
            img_pil = Image.fromarray(img_rgb)
            # Tạo đối tượng PhotoImage từ ảnh PIL
            img_tk = ImageTk.PhotoImage(image=img_pil)
            # Cập nhật hình ảnh trên canvas
            self.canvas.itemconfig(self.img_display, image=img_tk)
            self.canvas.image = img_tk
            # Di chuyển ảnh theo pan_x và pan_y
            self.canvas.coords(self.img_display, self.pan_x, self.pan_y)
        else:
            messagebox.showwarning("Warning", "No image to display!")  # Thông báo nếu không có ảnh

    def update_display_with_cursor(self, x, y):
        if self.img is None:
            return
        # Chuyển đổi tọa độ canvas sang tọa độ ảnh
        img_x = (x - self.pan_x) / self.zoom_scale
        img_y = (y - self.pan_y) / self.zoom_scale
        img_x = int(img_x)
        img_y = int(img_y)
        if self.index in self.image_label:
            img_temp = self.image_label[self.index].copy()  # Lấy ảnh từ từ điển
        else:
            img_temp = self.img.copy()
        cv2.circle(img_temp, (img_x, img_y), self.thickness // 2, self.current_color, -1)
        self.update_display_image(img_temp)

    def on_mouse_motion(self, event):
        if self.action_mode == 'label' and not self.drawing:
            self.update_display_with_cursor(event.x, event.y)
        elif self.action_mode == 'edit' and self.selected_object_index is not None and self.edit_sub_mode == 'add':
            self.update_display()
        else:
            pass

    def on_mouse_down(self, event):
        if self.action_mode == 'label':
            self.draw_annotation(event)
        elif self.action_mode == 'pan':
            self.start_pan(event)
        elif self.action_mode == 'edit':
            if self.edit_sub_mode == 'add':
                if self.selected_object_index is not None:
                    self.draw_annotation(event)
                else:
                    self.select_object(event)
            elif self.edit_sub_mode == 'delete':
                self.select_object(event)
            else:
                messagebox.showinfo("Thông báo", "Vui lòng chọn chế độ chỉnh sửa.")

    def on_mouse_move(self, event):
        if self.action_mode == 'label':
            self.draw_annotation(event)
        elif self.action_mode == 'pan':
            self.pan_image(event)
        elif self.action_mode == 'edit':
            if self.edit_sub_mode == 'add':
                if self.selected_object_index is not None:
                    self.draw_annotation(event)
            elif self.edit_sub_mode == 'delete':
                pass
            else:
                pass

    def on_mouse_up(self, event):
        if self.action_mode == 'label':
            self.draw_annotation(event)
        elif self.action_mode == 'pan':
            pass  # Không cần làm gì khi nhả chuột trong chế độ pan
        elif self.action_mode == 'edit':
            if self.edit_sub_mode == 'add':
                if self.selected_object_index is not None:
                    self.draw_annotation(event)
            elif self.edit_sub_mode == 'delete':
                pass
            else:
                pass

    def zoom(self, event):
        # Lấy vị trí chuột trên canvas
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        # Lấy tọa độ ảnh trước khi zoom
        image_x = (x - self.pan_x) / self.zoom_scale
        image_y = (y - self.pan_y) / self.zoom_scale
        # Điều chỉnh zoom_scale
        if event.delta > 0:
            self.zoom_scale *= 1.1
        else:
            self.zoom_scale /= 1.1
        self.zoom_scale = max(min(self.zoom_scale, 10), 0.1)
        # Điều chỉnh pan để giữ điểm dưới con trỏ chuột cố định
        self.pan_x = x - image_x * self.zoom_scale
        self.pan_y = y - image_y * self.zoom_scale
        self.update_display()

    def start_pan(self, event):
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def pan_image(self, event):
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.pan_x += dx
        self.pan_y += dy
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.update_display()

    # def draw_annotation(self, event):
        # Chuyển đổi tọa độ canvas sang tọa độ ảnh
        x = (event.x - self.pan_x) / self.zoom_scale
        y = (event.y - self.pan_y) / self.zoom_scale
        x = int(x)
        y = int(y)
        if self.img is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng mở một ảnh trước khi vẽ.")
            return

        # Determine the target object to add annotations to
        if self.action_mode == 'label':
            target_object = self.current_object
            target_gray_value = None
        elif self.action_mode == 'edit' and self.edit_sub_mode == 'add' and self.selected_object_index is not None:
            obj = self.annotations[self.selected_object_index]
            target_object = obj['annotations']
            target_gray_value = obj['gray_value']
        else:
            return  # Do nothing if no target object

        if event.type == tk.EventType.ButtonPress:
            self.drawing = True
            self.ix, self.iy = x, y
            if self.mode == 'curve':
                self.curve_points = [(self.ix, self.iy)]

        elif event.type == tk.EventType.Motion:
            if self.drawing:
                img_temp = self.img.copy()
                # Draw existing annotations
                self.redraw_annotations(img_temp)
                # Draw current drawing
                if self.mode == 'line':
                    cv2.line(img_temp, (self.ix, self.iy), (x, y), self.current_color, self.thickness)
                elif self.mode == 'curve':
                    self.curve_points.append((x, y))
                    if len(self.curve_points) > 1:
                        cv2.polylines(img_temp, [np.array(self.curve_points)], isClosed=False, color=self.current_color,
                                      thickness=self.thickness)
                self.update_display_image(img_temp)
            else:
                self.update_display_with_cursor(event.x, event.y)

        elif event.type == tk.EventType.ButtonRelease:
            self.drawing = False
            if self.mode == 'line':
                # Add line annotation to target_object
                target_object.append(('line', (self.ix, self.iy), (x, y), self.current_color, self.thickness))
                # Update images accordingly
                cv2.line(self.img, (self.ix, self.iy), (x, y), self.current_color, self.thickness)
                cv2.line(self.binary_img, (self.ix, self.iy), (x, y), 255, self.thickness)
                if target_gray_value is not None:
                    cv2.line(self.gray_lane_img, (self.ix, self.iy), (x, y), target_gray_value, self.thickness)
            elif self.mode == 'curve':
                self.curve_points.append((x, y))
                target_object.append(('curve', self.curve_points, self.current_color, self.thickness))
                cv2.polylines(self.img, [np.array(self.curve_points)], isClosed=False, color=self.current_color,
                              thickness=self.thickness)
                cv2.polylines(self.binary_img, [np.array(self.curve_points)], isClosed=False, color=255,
                              thickness=self.thickness)
                if target_gray_value is not None:
                    cv2.polylines(self.gray_lane_img, [np.array(self.curve_points)], isClosed=False, color=target_gray_value,
                                  thickness=self.thickness)
            self.update_display()
            # Reset curve_points
            self.curve_points = []
    def draw_annotation(self, event):
        # Chuyển đổi tọa độ canvas sang tọa độ ảnh
        x = (event.x - self.pan_x) / self.zoom_scale
        y = (event.y - self.pan_y) / self.zoom_scale
        x = int(x)
        y = int(y)
        
        if self.img is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng mở một ảnh trước khi vẽ.")
            return

        if event.type == tk.EventType.ButtonPress:
            self.drawing = True
            self.ix, self.iy = x, y
            if self.mode == 'curve':
                self.curve_points = [(self.ix, self.iy)]

        elif event.type == tk.EventType.Motion and self.drawing:
            # Tạo một bản sao tạm thời của hình ảnh gốc để vẽ
            img_temp = self.img.copy()

            # Vẽ đường thẳng hoặc đường cong lên bản sao tạm thời
            if self.mode == 'line':
                cv2.line(img_temp, (self.ix, self.iy), (x, y), self.current_color, self.thickness)
            elif self.mode == 'curve':
                self.curve_points.append((x, y))
                if len(self.curve_points) > 1:
                    cv2.polylines(img_temp, [np.array(self.curve_points)], isClosed=False, color=self.current_color,
                                thickness=self.thickness)

            # Cập nhật hiển thị trên canvas với bản sao tạm thời
            self.update_display_image(img_temp)

        elif event.type == tk.EventType.ButtonRelease and self.drawing:
            self.drawing = False
            # Vẽ lên ảnh gốc khi thả chuột
            if self.mode == 'line':
                cv2.line(self.img, (self.ix, self.iy), (x, y), self.current_color, self.thickness)
                cv2.line(self.binary_img, (self.ix, self.iy), (x, y), 255, self.thickness)
                self.current_object.append(('line', (self.ix, self.iy), (x, y), self.current_color, self.thickness))
            elif self.mode == 'curve':
                self.curve_points.append((x, y))
                cv2.polylines(self.img, [np.array(self.curve_points)], isClosed=False, color=self.current_color,
                            thickness=self.thickness)
                cv2.polylines(self.binary_img, [np.array(self.curve_points)], isClosed=False, color=255,
                            thickness=self.thickness)
                self.current_object.append(('curve', self.curve_points, self.current_color, self.thickness))
            
            # Cập nhật hiển thị sau khi hoàn thành vẽ
            self.update_display()

    def finish_current_object(self, event=None):
        if self.current_object:
            # Chọn màu xám ngẫu nhiên cho đối tượng hiện tại
            gray_value = self.random_gray()
            # Store the object as a dictionary
            obj = {'annotations': self.current_object, 'gray_value': gray_value}
            self.annotations.append(obj)
            self.current_object = []
            self.current_color = self.random_color()
            self.redraw_annotations()

    def toggle_mode(self, event=None):
        self.mode = 'curve' if self.mode == 'line' else 'line'
        self.mode_button.config(text=f"Chế độ vẽ: {self.mode}")

    def toggle_action_mode(self, event=None):
        if self.action_mode == 'label':
            self.action_mode = 'pan'
            self.action_mode_button.config(text="Chế độ: Pan")
            self.edit_mode_button.config(state=tk.DISABLED)
            self.edit_sub_mode = None
            self.selected_object_index = None
            self.update_display()
        elif self.action_mode == 'pan':
            self.action_mode = 'edit'
            self.action_mode_button.config(text="Chế độ: Chỉnh sửa")
            self.edit_mode_button.config(state=tk.NORMAL)
            self.edit_sub_mode = 'delete'  # Mặc định là 'delete'
            self.edit_mode_button.config(text="Chỉnh sửa: Xóa đối tượng")
        else:
            self.action_mode = 'label'
            self.action_mode_button.config(text="Chế độ: Label")
            self.edit_mode_button.config(state=tk.DISABLED)
            self.edit_sub_mode = None
            self.selected_object_index = None
            self.update_display()

    def update_display_image(self, temp_img):
        # Thay đổi kích thước ảnh tạm thời theo zoom_scale
        new_width = int(800 * self.zoom_scale)
        new_height = int(600 * self.zoom_scale)
        img_resized = cv2.resize(temp_img, (new_width, new_height))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.canvas.itemconfig(self.img_display, image=img_tk)
        self.canvas.image = img_tk
        # Di chuyển ảnh theo pan_x và pan_y
        self.canvas.coords(self.img_display, self.pan_x, self.pan_y)

    def undo_last_annotation(self, event=None):
        if self.current_object:
            self.current_object.pop()
            self.redraw_annotations()
        elif self.annotations:
            if self.action_mode == 'edit' and self.edit_sub_mode == 'add' and self.selected_object_index is not None:
                obj = self.annotations[self.selected_object_index]
                if obj['annotations']:
                    obj['annotations'].pop()
                    self.redraw_annotations()
                    self.update_display()
                else:
                    messagebox.showinfo("Thông báo", "Không còn chú thích để xóa.")
            else:
                self.annotations.pop()
                self.redraw_annotations()
        else:
            messagebox.showinfo("Thông báo", "Không còn chú thích để xóa.")

    def clear_all_annotations(self, event=None):
        if self.img is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng mở một ảnh trước khi xóa.")
            return

        if messagebox.askyesno("Xác nhận", "Bạn có chắc chắn muốn xóa tất cả các chú thích không?"):
            self.annotations = []
            self.current_object = []
            self.save_check_var.set(False)
            if self.index in self.image_label:
                del self.image_label[self.index]
            self.binary_img = np.zeros((600, 800), dtype=np.uint8)  # Xóa canvas nhị phân
            self.gray_lane_img = np.zeros((600, 800), dtype=np.uint8)  # Xóa canvas phân biệt lane
            self.redraw_annotations()

    def redraw_annotations(self, img=None):
        if self.img_path:
            if img is None:
                self.img = cv2.imread(self.img_path)
                if self.img is not None:
                    self.img = cv2.resize(self.img, (800, 600))
                    self.binary_img = np.zeros((600, 800), dtype=np.uint8)
                    self.gray_lane_img = np.zeros((600, 800), dtype=np.uint8)
                else:
                    messagebox.showerror("Lỗi", "Không thể tải lại ảnh gốc.")
                    return
            else:
                # Use provided img (e.g., for temporary display)
                self.img = img

            for obj in self.annotations:
                gray_value = obj['gray_value']
                for part in obj['annotations']:
                    if part[0] == 'line':
                        _, (x1, y1), (x2, y2), color, thickness = part
                        cv2.line(self.img, (x1, y1), (x2, y2), color, thickness)
                        cv2.line(self.binary_img, (x1, y1), (x2, y2), 255, thickness)
                        cv2.line(self.gray_lane_img, (x1, y1), (x2, y2), gray_value, thickness)
                    elif part[0] == 'curve':
                        _, points, color, thickness = part
                        cv2.polylines(self.img, [np.array(points)], isClosed=False, color=color, thickness=thickness)
                        cv2.polylines(self.binary_img, [np.array(points)], isClosed=False, color=255, thickness=thickness)
                        cv2.polylines(self.gray_lane_img, [np.array(points)], isClosed=False, color=gray_value, thickness=thickness)
            self.update_display()

    def update_thickness(self, value):
        self.thickness = int(value)

    def select_object(self, event):
        x = (event.x - self.pan_x) / self.zoom_scale
        y = (event.y - self.pan_y) / self.zoom_scale
        x = int(x)
        y = int(y)
        # Duyệt qua các đối tượng để kiểm tra xem điểm (x, y) có gần đối tượng nào không
        for idx, obj in enumerate(self.annotations):
            if self.is_point_near_object((x, y), obj['annotations']):
                if self.edit_sub_mode == 'delete':
                    self.selected_object_index = idx
                    self.delete_selected_object()
                elif self.edit_sub_mode == 'add':
                    self.selected_object_index = idx
                    self.update_display()
                return
        # Nếu không tìm thấy, hủy chọn
        self.selected_object_index = None
        self.update_display()

    def is_point_near_object(self, point, annotations, threshold=5):
        x, y = point
        for part in annotations:
            if part[0] == 'line':
                _, (x1, y1), (x2, y2), _, thickness = part
                dist = self.point_to_line_distance((x1, y1), (x2, y2), (x, y))
                if dist <= threshold + thickness:
                    return True
            elif part[0] == 'curve':
                points, _, thickness = part[1], part[2], part[3]
                for i in range(len(points) - 1):
                    dist = self.point_to_line_distance(points[i], points[i+1], (x, y))
                    if dist <= threshold + thickness:
                        return True
        return False

    def point_to_line_distance(self, p1, p2, p3):
        # Tính khoảng cách từ điểm p3 đến đoạn thẳng p1-p2
        x0, y0 = p3
        x1, y1 = p1
        x2, y2 = p2
        num = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
        den = np.hypot(y2 - y1, x2 - x1)
        if den == 0:
            return np.hypot(x0 - x1, y0 - y1)
        return num / den

    def delete_selected_object(self, event=None):
        if self.selected_object_index is not None:
            del self.annotations[self.selected_object_index]
            self.selected_object_index = None
            self.redraw_annotations()
            self.update_display()

    def draw_object_on_image(self, obj, img, override_color=None):
        for part in obj['annotations']:
            if part[0] == 'line':
                _, (x1, y1), (x2, y2), color, thickness = part
                if override_color:
                    color = override_color
                # Tính toán tọa độ theo zoom_scale
                x1 = int(x1 * self.zoom_scale)
                y1 = int(y1 * self.zoom_scale)
                x2 = int(x2 * self.zoom_scale)
                y2 = int(y2 * self.zoom_scale)
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            elif part[0] == 'curve':
                _, points, color, thickness = part
                if override_color:
                    color = override_color
                scaled_points = [(int(x * self.zoom_scale), int(y * self.zoom_scale)) for x, y in points]
                cv2.polylines(img, [np.array(scaled_points)], isClosed=False, color=color, thickness=thickness)


# Khởi chạy ứng dụng
root = tk.Tk()
app = AnnotationTool(root)
root.mainloop()
