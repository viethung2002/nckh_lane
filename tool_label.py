
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
        self.ix, self.iy = -1, -1
        self.annotations = []
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

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        # Toolbar
        toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        tk.Button(toolbar, text="Mở ảnh", command=self.open_folder_and_get_image_paths).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text="Lưu ảnh", command=self.save_images).pack(side=tk.LEFT, padx=2, pady=2)
        self.mode_button = tk.Button(toolbar, text="Chế độ: line", command=self.toggle_mode)
        self.mode_button.pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text="Hoàn thành đối tượng", command=self.finish_current_object).pack(side=tk.LEFT, padx=2,
                                                                                                 pady=2)
        tk.Button(toolbar, text="Xóa gần nhất", command=self.undo_last_annotation).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text="Xóa tất cả", command=self.clear_all_annotations).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text="Tạo Train/Valid/Test", command=self.create_train_valid_test_files).pack(side=tk.LEFT,
                                                                                                         padx=2, pady=2)
        self.next_button = tk.Button(root, text="Next",command=self.next_image)
        self.next_button.pack(side=tk.RIGHT)

        self.prev_button = tk.Button(root, text="Previous",command=self.prev_image)
        self.prev_button.pack(side=tk.LEFT)
        # Label để hiển thị số lượng ảnh
        self.image_count_label = tk.Label(root, text=self.text, font=("Arial", 14))
        self.image_count_label.pack(side=tk.BOTTOM)

        # Checkbox để kiểm tra đã lưu
        self.save_check_var = tk.BooleanVar()
        self.save_check_box = tk.Checkbutton(root, text="Saved", variable=self.save_check_var, command=self.check_click_button)
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
        self.canvas.bind("<ButtonPress-1>", self.draw_annotation)
        self.canvas.bind("<B1-Motion>", self.draw_annotation)
        self.canvas.bind("<ButtonRelease-1>", self.draw_annotation)
        self.canvas.bind("<Motion>", lambda event: self.update_display_with_cursor(event.x, event.y))

    def open_folder_and_get_image_paths(self):
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
    def next_image(self):
        if self.index < len(self.image_path_list) - 1:
            self.index += 1
            self.img_path = self.image_path_list[self.index]
            if not self.check_save[self.index]:
                self.save_check_var.set(False)
            else:
                self.save_check_var.set(True)
            if self.img_path:
                self.img = cv2.imread(self.img_path)
                if self.img is not None:
                    self.img = cv2.resize(self.img, (800, 600))
                    # Khởi tạo canvas nhị phân và canvas phân biệt lane cùng kích thước
                    self.binary_img = np.zeros((600, 800), dtype=np.uint8)
                    self.gray_lane_img = np.zeros((600, 800), dtype=np.uint8)
                    self.update_display()
        else:
            messagebox.showerror("Lỗi", "Không thể mở ảnh. Vui lòng thử lại.")
        self.text = f"{self.index+1} / {len(self.image_path_list)}"
        self.image_count_label.config(text=self.text)


    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.img_path = self.image_path_list[self.index]
            if not self.check_save[self.index]:
                self.save_check_var.set(False)
            else:
                self.save_check_var.set(True)
            if self.img_path:
                self.img = cv2.imread(self.img_path)
                if self.img is not None:
                    self.img = cv2.resize(self.img, (800, 600))
                    # Khởi tạo canvas nhị phân và canvas phân biệt lane cùng kích thước
                    self.binary_img = np.zeros((600, 800), dtype=np.uint8)
                    self.gray_lane_img = np.zeros((600, 800), dtype=np.uint8)
                    self.update_display()
        else:
            messagebox.showerror("Lỗi", "Không thể mở ảnh. Vui lòng thử lại.")
        self.text = f"{self.index+1} / {len(self.image_path_list)}"
        self.image_count_label.config(text=self.text)

    def save_images(self):
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
                # Save the annotated original image
            self.img = cv2.imread(self.img_path)
            self.img = cv2.resize(self.img, (800, 600))
            if self.img is not None:
                cv2.imwrite(image_path, self.img)

            # Save the binary image
            if self.binary_img is not None:
                cv2.imwrite(binary_path, self.binary_img)

                # Save the instance (gray lane) image
            if self.gray_lane_img is not None:
                cv2.imwrite(gray_lane_path, self.gray_lane_img)

            messagebox.showinfo("Lưu ảnh", f"Đã lưu ảnh dưới tên:\n{image_path}\n{binary_path}\n{gray_lane_path}")
            self.check_save[self.index] = True
            self.save_check_var.set(True)
            break

    def create_train_valid_test_files(self):
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
                image_path = os.path.join(base_path, "gt_image", f"{file_base}.png")
                binary_path = os.path.join(base_path, "gt_binary_image", f"{file_base}.png")
                instance_path = os.path.join(base_path, "gt_instance_image", f"{file_base}.png")
                f.write(f"{image_path} {binary_path} {instance_path}\n")

    def random_color(self):
        return tuple(random.randint(0, 255) for _ in range(3))

    def random_gray(self):
        gray_value = random.randint(50, 200)  # Chọn một giá trị xám ngẫu nhiên (giữa 50 và 200 để dễ phân biệt)
        return gray_value

    def open_image(self):
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
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.canvas.itemconfig(self.img_display, image=img_tk)
            self.canvas.image = img_tk

    def update_display_with_cursor(self, x, y):
        if self.img is None:
            return
        img_temp = self.img.copy()
        cv2.circle(img_temp, (x, y), self.thickness // 2, self.current_color, -1)
        self.update_display_image(img_temp)

    def draw_annotation(self, event):
        x, y = event.x, event.y
        if self.img is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng mở một ảnh trước khi vẽ.")
            return

        if event.type == tk.EventType.ButtonPress:
            self.drawing = True
            self.ix, self.iy = x, y
            if self.mode == 'curve':
                self.curve_points = [(self.ix, self.iy)]

        elif event.type == tk.EventType.Motion:
            if self.drawing:
                img_temp = self.img.copy()
                if self.mode == 'line':
                    cv2.line(img_temp, (self.ix, self.iy), (x, y), self.current_color, self.thickness)
                elif self.mode == 'curve':
                    self.curve_points.append((x, y))
                    if len(self.curve_points) > 1:
                        cv2.polylines(img_temp, [np.array(self.curve_points)], isClosed=False, color=self.current_color,
                                      thickness=self.thickness)
                self.update_display_image(img_temp)
            else:
                self.update_display_with_cursor(x, y)

        elif event.type == tk.EventType.ButtonRelease:
            self.drawing = False
            if self.mode == 'line':
                cv2.line(self.img, (self.ix, self.iy), (x, y), self.current_color, self.thickness)
                cv2.line(self.binary_img, (self.ix, self.iy), (x, y), 255, self.thickness)  # Vẽ lên ảnh nhị phân
                self.current_object.append(('line', (self.ix, self.iy), (x, y), self.current_color, self.thickness))
            elif self.mode == 'curve':
                self.curve_points.append((x, y))
                cv2.polylines(self.img, [np.array(self.curve_points)], isClosed=False, color=self.current_color,
                              thickness=self.thickness)
                cv2.polylines(self.binary_img, [np.array(self.curve_points)], isClosed=False, color=255,
                              thickness=self.thickness)  # Vẽ lên ảnh nhị phân
                self.current_object.append(('curve', self.curve_points, self.current_color, self.thickness))
            self.update_display()

    def finish_current_object(self):
        if self.current_object:
            # Chọn màu xám ngẫu nhiên cho đối tượng hiện tại
            gray_value = self.random_gray()
            for part in self.current_object:
                if part[0] == 'line':
                    _, (x1, y1), (x2, y2), _, thickness = part
                    cv2.line(self.gray_lane_img, (x1, y1), (x2, y2), gray_value, thickness)
                elif part[0] == 'curve':
                    _, points, _, thickness = part
                    cv2.polylines(self.gray_lane_img, [np.array(points)], isClosed=False, color=gray_value,
                                  thickness=thickness)

            self.annotations.append(self.current_object)
            self.current_object = []
            self.current_color = self.random_color()
            self.redraw_annotations()

    def toggle_mode(self):
        self.mode = 'curve' if self.mode == 'line' else 'line'
        self.mode_button.config(text=f"Chế độ: {self.mode}")

    def update_display_image(self, temp_img):
        img_rgb = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.canvas.itemconfig(self.img_display, image=img_tk)
        self.canvas.image = img_tk

    def undo_last_annotation(self):
        if self.current_object:
            self.current_object.pop()
        elif self.annotations:
            self.annotations.pop()
        else:
            messagebox.showinfo("Thông báo", "Không còn chú thích để xóa.")
        self.redraw_annotations()

    def clear_all_annotations(self):
        if self.img is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng mở một ảnh trước khi xóa.")
            return

        if messagebox.askyesno("Xác nhận", "Bạn có chắc chắn muốn xóa tất cả các chú thích không?"):
            self.annotations = []
            self.current_object = []
            self.binary_img = np.zeros((600, 800), dtype=np.uint8)  # Xóa canvas nhị phân
            self.gray_lane_img = np.zeros((600, 800), dtype=np.uint8)  # Xóa canvas phân biệt lane
            self.redraw_annotations()

    def redraw_annotations(self):
        if self.img_path:
            self.img = cv2.imread(self.img_path)
            if self.img is not None:
                self.img = cv2.resize(self.img, (800, 600))
                self.binary_img = np.zeros((600, 800), dtype=np.uint8)
                self.gray_lane_img = np.zeros((600, 800), dtype=np.uint8)
                for obj in self.annotations:
                    gray_value = self.random_gray()
                    for part in obj:
                        if part[0] == 'line':
                            _, (x1, y1), (x2, y2), color, thickness = part
                            cv2.line(self.img, (x1, y1), (x2, y2), color, thickness)
                            cv2.line(self.binary_img, (x1, y1), (x2, y2), 255, thickness)
                            cv2.line(self.gray_lane_img, (x1, y1), (x2, y2), gray_value, thickness)
                        elif part[0] == 'curve':
                            _, points, color, thickness = part
                            cv2.polylines(self.img, [np.array(points)], isClosed=False, color=color,
                                          thickness=thickness)
                            cv2.polylines(self.binary_img, [np.array(points)], isClosed=False, color=255,
                                          thickness=thickness)
                            cv2.polylines(self.gray_lane_img, [np.array(points)], isClosed=False, color=gray_value,
                                          thickness=thickness)
                self.update_display()
            else:
                messagebox.showerror("Lỗi", "Không thể tải lại ảnh gốc.")

    def update_thickness(self, value):
        self.thickness = int(value)


# Khởi chạy ứng dụng
root = tk.Tk()
app = AnnotationTool(root)
root.mainloop()
