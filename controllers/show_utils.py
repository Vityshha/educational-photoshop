import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem

from controllers.utils import Utils
import matplotlib.pyplot as plt

class UtilsWithDisplay:

    @staticmethod
    def show_min_max(image: np.ndarray, roi_image: np.ndarray = None):
        result = Utils.get_min_max(image, roi_image)
        if isinstance(result, dict):
            text = "\n".join(f"{k}: {v}" for k, v in result.items())
        else:
            text = f"min: {result[0]}\nmax: {result[1]}"

        lines = text.split('\n')
        line_height = 40
        margin = 20
        canvas_height = line_height * len(lines)
        canvas_width = 400

        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        for i, line in enumerate(lines):
            y = margin + i * line_height
            cv2.putText(canvas, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Min / Max Values", canvas)
        cv2.waitKey(0)

    @staticmethod
    def show_mean_std(image: np.ndarray, roi_image: np.ndarray = None):
        result = Utils.get_mean_std(image, roi_image)
        if isinstance(result, dict):
            text = "\n".join(f"{k}: {v:.2f}" for k, v in result.items())
        else:
            text = f"mean: {result[0]:.2f}\nstd: {result[1]:.2f}"

        lines = text.split('\n')
        line_height = 40
        margin = 20
        canvas_height = line_height * len(lines)
        canvas_width = 400

        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        for i, line in enumerate(lines):
            y = margin + i * line_height
            cv2.putText(canvas, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Mean / Std", canvas)
        cv2.waitKey(0)

    @staticmethod
    def show_histogram(image: np.ndarray, roi_image: np.ndarray = None):
        hist = Utils.get_histogram(image, roi_image)
        plt.figure(figsize=(6, 4))
        if isinstance(hist, dict):
            colors = ['r', 'g', 'b']
            for i, (k, h) in enumerate(hist.items()):
                plt.plot(h, color=colors[i], label=k)
        else:
            plt.plot(hist, color='gray', label='Grayscale')
        plt.title("Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.draw()
        fig = plt.gcf()
        fig.canvas.draw()
        img_np = np.array(fig.canvas.renderer.buffer_rgba())
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        plt.close()

        cv2.imshow("Histogram", img_bgr)
        cv2.waitKey(0)


    @staticmethod
    def show_noise_estimation(image: np.ndarray, roi_image: np.ndarray = None):
        """
        Отображает оценку аддитивного белого шума (СКО высокочастотной составляющей).
        """
        noise_std = Utils.estimate_white_noise_std(image, roi_image)
        text = f"Estimated noise (std): {noise_std:.2f}"

        canvas_height = 100
        canvas_width = 400
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

        cv2.putText(canvas, text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Noise Estimation", canvas)
        cv2.waitKey(0)


    @staticmethod
    def show_correlation_function(parent, image: np.ndarray, roi_image: np.ndarray = None,
                                  title="Корреляционная матрица"):
        """
        Строит и отображает корреляционную функцию как таблицу чисел в QDialog, с цветовым кодированием значений.

        :param parent: Родительское окно (обычно self или None).
        :param image: Входное изображение.
        :param roi_image: Маска ROI (или None).
        :param title: Заголовок окна.
        """
        corr = Utils.estimate_correlation_function(image, roi_image)

        # Нормализация в диапазон [0, 1]
        corr_min = np.min(corr)
        corr_max = np.max(corr)
        corr_norm = (corr - corr_min) / (corr_max - corr_min + 1e-8)

        dialog = QDialog(parent)
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)

        label = QLabel(f"Размер: {corr.shape[0]}×{corr.shape[1]}")
        layout.addWidget(label)

        table = QTableWidget()
        table.setRowCount(corr.shape[0])
        table.setColumnCount(corr.shape[1])

        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                val = corr[i, j]
                norm_val = corr_norm[i, j]

                # Преобразуем значение в цвет от синего к красному через зелёный
                red = int(norm_val * 255)
                green = int((1 - abs(norm_val - 0.5) * 2) * 255)
                blue = int((1 - norm_val) * 255)
                color = QColor(red, green, blue)

                item = QTableWidgetItem(f"{val:.2f}")
                item.setTextAlignment(Qt.AlignCenter)
                item.setBackground(color)
                table.setItem(i, j, item)

        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        layout.addWidget(table)

        dialog.resize(800, 800)
        dialog.exec_()
