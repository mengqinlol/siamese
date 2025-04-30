import sys
import torch
import numpy as np
import cv2
from PIL import Image

import os
import random
import sys
import threading
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QScrollArea, QScrollBar
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from networkx import enumerate_all_cliques
import torch
from tqdm import tqdm

from dataset import SiameseDataset
from minN import MinN
from model import SiameseNetwork
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from segment_anything import sam_model_registry, SamPredictor, build_sam_vit_b

from lora import LoRA_sam

M = 10000
N = 10

class LabelButton(QPushButton):
    selected = pyqtSignal(str)
    def __init__(self,  parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event):
        self.selected.emit(self.text())

class ScrollableButtonBar(QWidget):
    label_selected = pyqtSignal(str)
    def __init__(self, parent, buttons=None):
        super().__init__(parent)
        self.buttons = buttons if buttons else []
        self.initUI()
        self.update_buttons()  # 初始创建按钮

    def initUI(self):
        main_layout = QVBoxLayout(self)
        
        # 创建滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # 创建按钮容器
        self.button_widget = QWidget()
        self.button_layout = QHBoxLayout(self.button_widget)
        self.button_layout.setSpacing(10)
        self.button_layout.setContentsMargins(10, 0, 10, 0)
        
        # 设置滚动区域
        self.scroll_area.setWidget(self.button_widget)
        
        # 创建自定义滚动条
        self.h_scroll = QScrollBar(Qt.Horizontal)
        self.h_scroll.setFixedHeight(15)
        
        # 同步滚动条与滚动区域
        self.scroll_bar = self.scroll_area.horizontalScrollBar()
        self.scroll_bar.rangeChanged.connect(self.h_scroll.setRange)
        self.scroll_bar.valueChanged.connect(self.h_scroll.setValue)
        self.h_scroll.valueChanged.connect(self.scroll_bar.setValue)
        
        # 布局设置
        main_layout.addWidget(self.scroll_area)
        main_layout.addWidget(self.h_scroll)
        self.setLayout(main_layout)

    def update_buttons(self, buttons=None):
        """动态更新按钮列表"""
        if buttons is not None:
            self.buttons = buttons
            
        # 清空现有按钮
        while self.button_layout.count():
            item = self.button_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 添加新按钮
        for text in self.buttons:
            btn = LabelButton(text)
            btn.setStyleSheet("font-size: 32px; border: none; background-color: white; color: black; border-radius: 5px;")
            btn.setFixedWidth(120)
            btn.setFixedHeight(60)
            btn.selected.connect(self.on_button_click)
            self.button_layout.addWidget(btn)
        self.button_layout.addStretch()
        
        # 强制更新布局
        self.button_widget.adjustSize()
        self.scroll_area.updateGeometry()

    def on_button_click(self, text):
        self.label_selected.emit(text)

    def wheelEvent(self, event):
        if event.pixelDelta().y() != 0:
            delta = event.pixelDelta().y()
        else:
            delta = event.angleDelta().y() // 8
        
        new_value = self.scroll_bar.value() - delta * 3
        self.scroll_bar.setValue(new_value)
        event.accept()


class Predictor:
    def __init__(self, model_path = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_path is not None:
            self.load_model(model_path)
        self.trans = transforms.Compose([
            transforms.Resize([220, 220]),
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
        ])
        
    def load_model(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()
        self.model.to(self.device)
        print('Model loaded successfully.')

    def set_labeled_imgs_path(self, path):
        train_data = []
        self.label_set = []
        for label in tqdm(os.listdir(f'{path}')):
            self.label_set.append(label)
            for img_name in os.listdir(f'{path}/{label}'):
                img_path = f'{path}/{label}/{img_name}'
                img = Image.open(img_path).convert('RGB')
                img = self.trans(img)
                train_data.append((img, label))
        trainedDataset = SiameseDataset(img_label_list = train_data, forTrain = False)
        trainedDataloader = DataLoader(trainedDataset, batch_size=32, shuffle=False)

        self.labelSet = []
        label_cnt_dict = {}

        # 使用tqdm显示进度条
        for batch in tqdm(trainedDataloader, desc="Processing batches"):
            # 解包batch数据（假设batch包含(samples, labels)）
            samples, labels = batch
            samples = samples.to(self.device)
            
            # 前向传播（禁用梯度计算）
            with torch.no_grad():
                outputs = self.model(samples)  # 输出形状: [batch_size, embedding_dim]
            
            # 逐个处理batch中的样本
            for i in range(outputs.size(0)):
                output = outputs[i]  # 单个样本的输出向量
                label = labels[i]    # 对应的标签
                
                # 检查是否达到最大样本数限制
                if label_cnt_dict.get(label, 0) >= M:
                    continue
                
                # 添加到labelSet并更新计数
                if label not in label_cnt_dict:
                    self.labelSet.append((output.data, label))  # 移动到CPU防止内存泄漏
                    label_cnt_dict[label] = 1
                else:
                    self.labelSet.append((output.data, label))
                    label_cnt_dict[label] += 1

        print('Standard output setting: Done.')

    def load_image(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = self.trans(img)
        img.to(self.device)
        return img

    def predict(self, img):
        img = img.unsqueeze(0)  # 添加batch维度
        with torch.no_grad():
            outputs = self.model(img)
        outputs = outputs.squeeze(0)  # 移除batch维度
        Minn = MinN(N)  

        for vec, v_label in self.labelSet:
            # print(outputs.shape, vec.shape)
            currDis = (outputs - vec).pow(2).sum()
            Minn.add(currDis, v_label)
        
        curr_label = Minn.get_first_label()
        pred_list = Minn.get_sorted_list()

        return curr_label, pred_list


class ImageLabel(QLabel):
    croped_img_generated = pyqtSignal(tuple)
    croped_img_cleared = pyqtSignal()
    def __init__(self, image_path = None):
        super().__init__()
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)

        self.input_points = []
        self.input_labels = []
        self.latest_cropped_pil = None
        self.bboxes = []
        self.masks = []
        self.labels = []

        model_type = "vit_h"
        checkpoint_path = "sam_vit_h_4b8939.pth"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam = build_sam_vit_b("sam_vit_b.pth")
        self.sam_lora = LoRA_sam(self.sam, 512)
        self.sam_lora.load_lora_parameters(f"./lora_rank{self.sam_lora.rank}_2.safetensors")
        self.sam_lora.sam.to(device)
        self.original_pixmap = None
        self.scaled_pixmap = None
        if image_path:
            self.image_path = image_path
            self.set_image(image_path)

    def set_image(self, image_path):
        print("set image"+ image_path)
        self.image_path = image_path
        self.original_image_bgr = cv2.imread(image_path)
        self.original_image = cv2.cvtColor(self.original_image_bgr, cv2.COLOR_BGR2RGB)
        self.original_pixmap = self.convert_cv_to_pixmap(self.original_image)
        self.scaled_pixmap = self.original_pixmap

        # self.predictor = SamPredictor(self.sam)
        self.predictor = SamPredictor(self.sam_lora.sam)
        self.predictor.set_image(self.original_image)
        self.update_scaled_pixmap()

    def convert_cv_to_pixmap(self, image):
        image = np.ascontiguousarray(image, dtype=np.uint8)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image)

    def resizeEvent(self, event):
        self.update_scaled_pixmap()
        super().resizeEvent(event)

    def update_scaled_pixmap(self):
        if self.original_pixmap:
            self.scaled_pixmap = self.original_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(self.scaled_pixmap)

    def is_within_image_bounds(self, x, y, image_size):
        return 0 <= x <= image_size.width() and 0 <= y <= image_size.height()

    def set_latest_pixmap(self):
        print(self.best_mask, self.bbox)
        pixmap_size = self.scaled_pixmap.size()
        overlay = self.original_image.copy()
        overlay = cv2.resize(overlay, (pixmap_size.width(), pixmap_size.height()))
        if self.best_mask is not None:
            best_mask = self.best_mask
            # 绘制掩码和bbox
            resized_mask = cv2.resize(best_mask.astype(np.uint8) * 255,
                                        (pixmap_size.width(), pixmap_size.height()),
                                        interpolation=cv2.INTER_NEAREST)
            overlay[resized_mask > 0] = overlay[resized_mask > 0] * 0.5 + np.array([225, 0, 0]) * 0.5

        if self.bbox is not None:
            bbox = self.bbox
            scale_x = pixmap_size.width() / self.original_image.shape[1]
            scale_y = pixmap_size.height() / self.original_image.shape[0]
            x1, y1, x2, y2 = [int(coord * scale) for coord, scale in zip(bbox, [scale_x, scale_y, scale_x, scale_y])]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

        for i in range(len(self.masks)):
            maskk = self.masks[i]
            resized_mask = cv2.resize(maskk.astype(np.uint8) * 255,
                                    (pixmap_size.width(), pixmap_size.height()),
                                    interpolation=cv2.INTER_NEAREST)

            overlay[resized_mask > 0] = overlay[resized_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5

            bboxx = self.bboxes[i]
            scale_x = pixmap_size.width() / self.original_image.shape[1]
            scale_y = pixmap_size.height() / self.original_image.shape[0]
            x1, y1, x2, y2 = [int(coord * scale) for coord, scale in zip(bboxx, [scale_x, scale_y, scale_x, scale_y])]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 225, 0), 2)

            label = self.labels[i]
            cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        display_pixmap = self.convert_cv_to_pixmap(overlay)
        self.setPixmap(display_pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.input_points.clear()
            self.input_labels.clear()
            self.latest_cropped_pil = None
            self.best_mask = None
            self.bbox = None
            self.set_latest_pixmap()
            return

        if self.pixmap() is None:
            return

        click_point = event.pos()
        label_size = self.size()
        pixmap_size = self.scaled_pixmap.size()

        x_offset = (label_size.width() - pixmap_size.width()) / 2
        y_offset = (label_size.height() - pixmap_size.height()) / 2

        x_in_image = click_point.x() - x_offset
        y_in_image = click_point.y() - y_offset

        if self.is_within_image_bounds(x_in_image, y_in_image, pixmap_size):
            rel_x = x_in_image / pixmap_size.width()
            rel_y = y_in_image / pixmap_size.height()

            orig_x = rel_x * self.original_image.shape[1]
            orig_y = rel_y * self.original_image.shape[0]

            print(f"Relative click position on original image: ({orig_x:.2f}, {orig_y:.2f})")
            print(f"Relative click position on scaled image: ({x_in_image:.2f}, {y_in_image:.2f})")

            self.input_points.append([orig_x, orig_y])
            self.input_labels.append(1)

            masks, scores, _ = self.predictor.predict(
                point_coords=np.array(self.input_points),
                point_labels=np.array(self.input_labels),
                multimask_output=True
            )

            best_mask = masks[np.argmax(scores)]
            self.best_mask = best_mask

            # 计算 bounding box
            mask_indices = np.argwhere(best_mask)
            y_min, x_min = mask_indices.min(axis=0)
            y_max, x_max = mask_indices.max(axis=0)
            bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
            self.bbox = bbox
            print(f"Detected BBox: {bbox}")

            # 显示 bbox 图像到 thumbnail_label
            cropped = self.original_image[y_min:y_max, x_min:x_max]
            self.croped_img_generated.emit((int(x_min), int(y_min), int(x_max), int(y_max)))
            # self.parent_window.thumbnail_label.setPixmap(thumb_pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))

            # 保存 PIL 版本
            self.latest_cropped_pil = Image.fromarray(cropped)

            self.set_latest_pixmap()
            

    def next_mask(self, label):
        self.masks.append(self.best_mask)
        self.best_mask = None
        self.bboxes.append(self.bbox)
        self.bbox = None
        self.labels.append(label)
        self.set_latest_pixmap()


class ImageWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer")

        self.predictor = Predictor()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.trans = transforms.Compose([
            transforms.Resize([220, 220]),
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
        ])

        layout = QVBoxLayout()

         # 顶部按钮布局
        top_layout = QVBoxLayout()
        
        # 新增按钮
        self.select_labeled_folder_layout = QHBoxLayout()
        self.select_labeled_folder_btn = QPushButton("选择已标注文件夹", self)
        self.select_labeled_folder_label = QLabel("已标注文件夹路径：", self)
        self.select_labeled_folder_layout.addWidget(self.select_labeled_folder_btn)
        self.select_labeled_folder_layout.addWidget(self.select_labeled_folder_label)

        self.select_unlabeled_folder_layout = QHBoxLayout()
        self.select_unlabeled_folder_btn = QPushButton("选择未标注文件夹", self)  # 新按钮
        self.select_unlabeled_folder_label = QLabel("未标注文件夹路径：", self)
        self.select_unlabeled_folder_layout.addWidget(self.select_unlabeled_folder_btn)
        self.select_unlabeled_folder_layout.addWidget(self.select_unlabeled_folder_label)
        
        self.select_labeled_folder_btn.setEnabled(False)
        self.select_unlabeled_folder_btn.setEnabled(False)

        self.model_choose_layout = QHBoxLayout()
        self.model_choose_button = QPushButton("选择模型", self)
        self.model_choose_label = QLabel("模型文件：", self)
        self.model_choose_layout.addWidget(self.model_choose_button)
        self.model_choose_layout.addWidget(self.model_choose_label)

        # 添加按钮到布局（保持顺序）
        top_layout.addLayout(self.select_labeled_folder_layout)
        top_layout.addLayout(self.select_unlabeled_folder_layout)
        top_layout.addLayout(self.model_choose_layout)
        
        layout.addLayout(top_layout)

        self.select_labeled_folder_btn.clicked.connect(self.select_labeled_folder)
        self.select_unlabeled_folder_btn.clicked.connect(self.select_unlabeled_folder)  # 新连接
        self.model_choose_button.clicked.connect(self.select_model)

        self.image_label = ImageLabel()
        self.image_label.setFixedSize(700, 400)
        self.image_label.croped_img_generated.connect(self.predict_croped_img)
        self.image_label.croped_img_cleared.connect(self.clear_croped_img)
        layout.addWidget(self.image_label)

         # 底部区域
        bottom_layout = QHBoxLayout()
        self.label_shown = QLabel("预测的标签：", self)
        self.label_shown.setStyleSheet("font-size: 32px; font-weight: bold;")
        self.start_button = QPushButton("开始", self)
        self.start_button.clicked.connect(self.on_start)
        
        # 布局设置
        bottom_layout.addWidget(self.label_shown)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.start_button)
        layout.addLayout(bottom_layout)
        tmp = ['1','2', '3', '4', '5', '6', '7', '8', '9', '10']
        self.pred_label_buttons = ScrollableButtonBar(self, tmp)
        self.pred_label_buttons.label_selected.connect(self.on_label_selected)
        self.pred_label_buttons.hide()
        layout.addWidget(self.pred_label_buttons)

        self.setLayout(layout)

        self.cur_bboxes_label = []


    def select_model(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "Model Files (*.pth, *.pt)")
        if model_path:
            self.model_path = model_path
            self.model_choose_label.setText(f"模型文件: {os.path.basename(model_path)}")
            self.predictor.load_model(model_path)
            self.select_labeled_folder_btn.setEnabled(True)
            self.select_unlabeled_folder_btn.setEnabled(True)
    
    # 新增槽函数
    def select_unlabeled_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择未标注文件夹")
        if folder_path:
            self.select_unlabeled_folder_label.setText(f"未标注文件夹: {folder_path}")
            self.start_button.setEnabled(True)
            self.unlabeled_folder = folder_path
            self.unlabeled_images = [f"{folder_path}/{file}" for file in os.listdir(folder_path)]
            # 打乱顺序
            random.shuffle(self.unlabeled_images)
            self.unlabeled_index = -1
            # 这里可以添加加载未标注数据的逻辑

    # 原选择已标注文件夹的槽函数
    class Worker(QObject):
        finished = pyqtSignal()
        
        def __init__(self, predictor, folder_path):
            super().__init__()
            self.predictor = predictor
            self.folder_path = folder_path
        
        def process(self):
            self.predictor.set_labeled_imgs_path(self.folder_path)
            self.finished.emit()

    def select_labeled_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择已标注文件夹")
        self.labeled_folder_path = folder_path
        if folder_path:
            # 更新UI操作保持在主线程
            self.select_labeled_folder_label.setText(f"正在加载: {folder_path}")
            self.select_labeled_folder_label.setStyleSheet("color: orange;")
            self.start_button.setEnabled(True)
            
            # 创建线程和工作者对象
            self.worker_thread = QThread()
            self.worker = self.Worker(self.predictor, folder_path)
            
            # 将工作者移动到子线程
            self.worker.moveToThread(self.worker_thread)
            
            # 连接信号槽
            self.worker_thread.started.connect(self.worker.process)  # 线程启动时调用process
            self.worker.finished.connect(self.worker_thread.quit)     # 完成后退出线程
            self.worker.finished.connect(self.worker.deleteLater)  # 删除工作者
            self.worker_thread.finished.connect(self.worker_thread.deleteLater)  # 删除线程
            self.worker.finished.connect(self.on_select_labeled_folder_finish)  # 完成后调用on_finish
            
            # 启动线程
            self.worker_thread.start()
    
    def on_select_labeled_folder_finish(self):
        # 更新UI操作保持在主线程
        self.select_labeled_folder_label.setText(f"加载完成: {self.labeled_folder_path}")
        self.select_labeled_folder_label.setStyleSheet("color: green;")

    def on_start(self):
        self.label_shown.setText("开始处理...")
        # 这里可以添加处理逻辑
        self.next_image()

    def display_image(self, image_path):
        self.image_label.set_image(image_path)

    def next_image(self, text = ""):
        print(text)
        self.unlabeled_index += 1
        if self.unlabeled_index < len(self.unlabeled_images):
            self.display_image(self.unlabeled_images[self.unlabeled_index])
        else:
            self.label_shown.setText("没有更多图片")
            self.label_shown.setStyleSheet("color: red;")
        self.pred_label_buttons.update_buttons(['1','2', '3', '4', '5', '6', '7', '8', '9', '10'])

    def predict_croped_img(self, bbox_tuple):
        self.predicting_bbox = bbox_tuple
        img_path = self.unlabeled_images[self.unlabeled_index]
        img = Image.open(img_path).convert('RGB')
        bbox_img = img.crop(bbox_tuple)
        bbox_img = self.trans(bbox_img)
        bbox_img.to(self.device)
        label_predicted, pred_list = self.predictor.predict(bbox_img)
        self.label_shown.setText(f"预测结果: {label_predicted}")
        pred_label_list = [item[0] for item in pred_list]
        for label in self.predictor.label_set:
            if label not in pred_label_list:
                pred_label_list.append(label)
        self.pred_label_buttons.update_buttons(pred_label_list)
        self.pred_label_buttons.show()

    def on_label_selected(self, label):
        print(label)
        self.cur_bboxes_label.append((self.predicting_bbox, label))
        self.image_label.next_mask(label)
        self.pred_label_buttons.hide()

    def clear_croped_img(self):
        self.pred_label_buttons.update_buttons(['1','2', '3', '4', '5', '6', '7', '8', '9', '10'])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageWindow()
    window.resize(1000, 600)
    window.show()
    sys.exit(app.exec_())