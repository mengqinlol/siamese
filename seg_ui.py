import os
import random
import sys
import threading
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog
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

IMAGE_SIZE = [220, 220]

M = 10000
N = 10

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QPushButton, QScrollArea, QScrollBar, QVBoxLayout

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QPushButton, QScrollArea, QScrollBar, QVBoxLayout
from PyQt5.QtCore import Qt

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QPushButton, QScrollArea, QScrollBar, QVBoxLayout
from PyQt5.QtCore import Qt

class ScrollableButtonBar(QWidget):
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
            btn = QPushButton(text)
            btn.setStyleSheet("font-size: 32px; border: none; background-color: white; color: black; border-radius: 5px;")
            btn.setFixedWidth(120)
            btn.setFixedHeight(60)
            btn.clicked.connect(lambda checked, t=text: self.parent().next_image(t))
            self.button_layout.addWidget(btn)
        self.button_layout.addStretch()
        
        # 强制更新布局
        self.button_widget.adjustSize()
        self.scroll_area.updateGeometry()

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
            transforms.Resize(IMAGE_SIZE),
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

    def set_target_paths(self, target_paths):
        self.target_paths = target_paths
        self.target_idx = 0
        self.predicted_idx = 0
        self.target_ans = [()] * len(target_paths)

    def empty(self):
        return len(self.target_paths) == self.target_idx

    def load_image(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = self.trans(img)
        img.to(self.device)
        return img
    
    def get_next_target(self):
        if self.target_idx >= self.predicted_idx:
            self.predict_next_10()
            print('predict next 10')
        if self.target_idx % 10 == 1:
            thread = threading.Thread(target=self.predict_next_10)
            thread.start()
        ans = self.target_ans[self.target_idx]
        self.target_idx += 1
        return ans[0], ans[1], self.target_idx - 1
    
    def predict_next_10(self):
        self.model.to(self.device)
        img_list = []
        for i in range(self.predicted_idx, self.predicted_idx + 10):
            img = self.load_image(self.target_paths[i])
            img_list.append((img, "unknown"))
        testDataset = SiameseDataset(img_label_list = img_list, forTrain = False)
        testDataloader = DataLoader(testDataset, batch_size=10, shuffle=False)

        for i, data in enumerate(testDataloader):
            samples, labels = data  
            samples = samples.to(self.device)
            
            outputs = self.model(samples).data 
            
            for j in range(outputs.size(0)):

                output = outputs[j:j+1] 
                Minn = MinN(N)  
                
                for vec, v_label in self.labelSet:
                    currDis = (output - vec).pow(2).sum(1)
                    Minn.add(currDis, v_label)
                
                curr_label = Minn.get_first_label()
                pred_list = Minn.get_sorted_list()
                
                self.target_ans[self.predicted_idx + j] = (curr_label, pred_list)

        self.predicted_idx += 10



class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置窗口标题和大小
        self.setWindowTitle("AI自动标注系统")
        self.setGeometry(100, 100, 800, 600)  # 调整窗口宽度

        self.predictor = Predictor()

        # 创建主布局
        main_layout = QVBoxLayout()

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
        
        main_layout.addLayout(top_layout)

        # 图片展示区域
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(700, 400)  # 调整图片框大小
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        # 底部区域
        bottom_layout = QHBoxLayout()
        self.label_shown = QLabel("预测的标签：", self)
        self.label_shown.setStyleSheet("font-size: 32px; font-weight: bold;")
        self.start_button = QPushButton("开始", self)
        
        # 布局设置
        bottom_layout.addWidget(self.label_shown)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.start_button)
        main_layout.addLayout(bottom_layout)
        tmp = ['1','2', '3', '4', '5', '6', '7', '8', '9', '10']
        self.pred_label_buttons = ScrollableButtonBar(self, tmp)
        main_layout.addWidget(self.pred_label_buttons)

        # 信号连接
        self.select_labeled_folder_btn.clicked.connect(self.select_labeled_folder)
        self.select_unlabeled_folder_btn.clicked.connect(self.select_unlabeled_folder)  # 新连接
        self.model_choose_button.clicked.connect(self.select_model)
        self.start_button.clicked.connect(self.on_start)

        # 初始状态
        self.start_button.setEnabled(False)

        # 设置主布局
        self.setLayout(main_layout)

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
            self.predictor.set_target_paths(self.unlabeled_images)
            self.unlabeled_index = 0
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

    # 保持原有方法不变
    def on_start(self):
        self.label_shown.setText("开始处理...")
        # 这里可以添加处理逻辑
        self.next_image()

    def display_image(self, image_path):
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.label_shown.setText("图片加载失败")
            self.label_shown.setStyleSheet("color: red;")

    def next_image(self, text = ""):
        print(text)
        if not self.predictor.empty():
            label_predicted, pred_list, idx = self.predictor.get_next_target()
            self.display_image(self.unlabeled_images[idx])
            self.label_shown.setText(f"预测结果: {label_predicted}")
            pred_label_list = [item[0] for item in pred_list]
            for label in self.predictor.label_set:
                if label not in pred_label_list:
                    pred_label_list.append(label)
            self.pred_label_buttons.update_buttons(pred_label_list)
        else:
            self.label_shown.setText("没有更多图片")
            self.label_shown.setStyleSheet("color: red;")

    def keyPressEvent(self, a0):
        if a0.key() == Qt.Key_Space:
            self.next_image()
        return super().keyPressEvent(a0)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())