import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QPushButton
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class HeatmapWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Heatmap with Slider")
        self.setGeometry(100, 100, 700, 200)

        self.initUI()

    def initUI(self):
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)

        layout = QVBoxLayout(main_widget)

        # 创建显示热度图的区域
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # 创建滑块并设置范围和初始值
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 1000)
        self.slider.setValue(500)
        layout.addWidget(self.slider)

        next_button = QPushButton("Next")
        next_button.clicked.connect(lambda value: self.updateValues(_update_optimizer_flag=True))
        layout.addWidget(next_button)

        # 连接滑块的valueChanged信号到更新热度图的槽函数
        self.slider.valueChanged.connect(self.update_heatmap)

        # 初始化热度图
        self.update_heatmap()

    def update_heatmap(self):
        # 获取滑块的值
        slider_value = self.slider.value()

        # 创建一个随机的热度图数据
        heatmap_data = np.random.rand(12, 160) * slider_value

        # 绘制热度图
        self.ax.clear()
        self.ax.imshow(heatmap_data, cmap='viridis')
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HeatmapWindow()
    window.show()
    sys.exit(app.exec_())
