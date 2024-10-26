import torch
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from model.lanenet.Hnet import HNet

class LaneNetPostProcessor:
    def __init__(self, hnet_model_path, eps=0.35, min_samples=300, min_area_threshold=100, device='cuda'):
        """
        Khởi tạo lớp hậu xử lý LaneNet với các tham số cho DBSCAN và ngưỡng diện tích tối thiểu.
        :param hnet_model_path: Đường dẫn tới mô hình HNet đã huấn luyện
        :param eps: Tham số khoảng cách cho DBSCAN
        :param min_samples: Số lượng mẫu tối thiểu để tạo thành một cụm trong DBSCAN
        :param min_area_threshold: Ngưỡng diện tích tối thiểu để giữ lại thành phần kết nối
        :param device: Thiết bị để chạy mô hình HNet ('cpu' hoặc 'cuda')
        """
        self.eps = eps
        self.min_samples = min_samples
        self.min_area_threshold = min_area_threshold
        self.device = device

        # Khởi tạo mô hình HNet và tải trọng số
        self.hnet = HNet().to(device)
        hnet_state_dict = torch.load(hnet_model_path, map_location=device)
        self.hnet.load_state_dict(hnet_state_dict)  # Tải state dictionary
        self.hnet.eval()

        # Định nghĩa bảng màu để trực quan hóa
        self.color_map = [
            np.array([255, 0, 0]),    # Đỏ
            np.array([0, 255, 0]),    # Xanh lá
            np.array([0, 0, 255]),    # Xanh dương
            np.array([125, 125, 0]),  # Vàng
            np.array([0, 125, 125]),  # Xanh lục lam
            np.array([125, 0, 125]),  # Tím
            np.array([50, 100, 50]),  # Xanh lá đậm
            np.array([100, 50, 100])  # Tím đậm
        ]

    def _morphological_process(self, binary_img, kernel_size=5):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=1)
        return closing

    def _connect_components_analysis(self, binary_img):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
        return num_labels, labels, stats, centroids

    def _dbscan_cluster(self, embedding_feats):
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        features = StandardScaler().fit_transform(embedding_feats)
        db.fit(features)
        db_labels = db.labels_
        unique_labels = np.unique(db_labels)
        return db_labels, unique_labels

    def _apply_hnet(self, image):
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            hnet_output = self.hnet(image_tensor)
        return hnet_output.squeeze().cpu().numpy()

    def postprocess(self, binary_seg_pred, instance_seg_logits, source_image=None):
        """
        Hàm chính để xử lý sau khi mô hình LaneNet đưa ra dự đoán.
        :param binary_seg_pred: Ảnh nhị phân dự đoán từ mô hình (đầu ra segmentation nhị phân)
        :param instance_seg_logits: Đầu ra logits phân đoạn instance từ mô hình
        :param source_image: Ảnh gốc để hiển thị kết quả (nếu có)
        :return: Ảnh nhị phân sau xử lý, các thông số fitting đường làn và ảnh gốc đã được vẽ lên
        """
        # Chuyển đổi đầu ra segmentation nhị phân thành ảnh nhị phân thực tế
        binary_seg_pred = binary_seg_pred.squeeze().cpu().numpy() * 255
        binary_seg_pred = binary_seg_pred.astype(np.uint8)

        # Phép toán hình thái học để làm mịn
        binary_seg_pred = self._morphological_process(binary_seg_pred)

        # Phân tích các thành phần liên thông
        num_labels, labels, stats, _ = self._connect_components_analysis(binary_seg_pred)

        # Loại bỏ các thành phần nhỏ
        for i in range(num_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.min_area_threshold:
                binary_seg_pred[labels == i] = 0

        # Lấy các đặc trưng nhúng từ output phân đoạn instance
        instance_seg_logits = instance_seg_logits.squeeze().cpu().numpy()
        idx = np.where(binary_seg_pred == 255)
        embedding_feats = instance_seg_logits[:, idx[0], idx[1]].T

        # Phân cụm các đặc trưng nhúng để phân biệt các làn đường
        db_labels, unique_labels = self._dbscan_cluster(embedding_feats)

        # Tạo mask để trực quan hóa các lane khác nhau
        mask = np.zeros((binary_seg_pred.shape[0], binary_seg_pred.shape[1], 3), dtype=np.uint8)
        
        print("Unique labels:", unique_labels)  # Thêm dòng này để kiểm tra các label

        for label in unique_labels:
            if label == -1:
                continue  # Bỏ qua các điểm nhiễu
            # Đảm bảo rằng label là một số nguyên
            try:
                color = self.color_map[int(label) % len(self.color_map)]
            except IndexError:
                print(f"Label {label} vượt quá kích thước của color_map. Sử dụng màu mặc định.")
                color = np.array([255, 255, 255])  # Màu trắng như màu mặc định
            # Gán màu cho các pixel tương ứng
            mask[idx[0][db_labels == label], idx[1][db_labels == label]] = color

        # Overlay mask lên ảnh gốc nếu có
        if source_image is not None:
            overlay = cv2.addWeighted(source_image, 0.6, mask, 0.4, 0)
            return mask, overlay
        else:
            return mask
