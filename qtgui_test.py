import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QGridLayout, QLabel, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QColor


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 创建水平滑块和垂直滑块
        self.horizontal_slider = QSlider(Qt.Horizontal, self)
        self.horizontal_slider.valueChanged.connect(self.on_horizontal_slider_value_changed)
        self.vertical_sliders = []
        for i in range(20):
            slider = QSlider(Qt.Vertical, self)
            slider.setRange(0, 100)
            slider.setDisabled(True)
            self.vertical_sliders.append(slider)

        # 创建热度图区域
        self.heatmap_label = QLabel(self)
        self.heatmap_label.setFixedSize(200, 200)
        self.heatmap_pixmap = QPixmap(200, 200)
        self.heatmap_pixmap.fill(Qt.white)
        self.heatmap_label.setPixmap(self.heatmap_pixmap)

        # 将水平滑块和垂直滑块以及热度图区域添加到布局中
        grid = QGridLayout()
        grid.addWidget(self.heatmap_label, 0, 6, 1, 10)
        for i in range(len(self.vertical_sliders)):
            grid.addWidget(self.vertical_sliders[i], 1, i)
        grid.addWidget(self.horizontal_slider, 2, 0, 1, 20)

        # 创建主窗口，并设置布局
        widget = QWidget(self)
        widget.setLayout(grid)
        self.setCentralWidget(widget)

        # 设置主窗口的大小和标题
        self.resize(600, 400)
        self.setWindowTitle('Slider Example')

    def on_horizontal_slider_value_changed(self, value):
        # 更新垂直滑块的值
        for i in range(len(self.vertical_sliders)):
            self.vertical_sliders[i].setValue(value * (i + 1))

        # 更新热度图区域的颜色
        img = QImage(200, 200, QImage.Format_RGB32)
        for x in range(200):
            for y in range(200):
                r, g, b = self.get_heatmap_color(value, x, y)
                img.setPixelColor(x, y, QColor(int(r), int(g), int(b)))
        self.heatmap_pixmap = QPixmap.fromImage(img)
        self.heatmap_label.setPixmap(self.heatmap_pixmap)

    def get_heatmap_color(self, value, x, y):
        r = 255 * value / 100
        g = 255 * x / 200
        b = 255 * y / 200
        return r, g, b


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
