import cv2
import numpy as np


class Utils:
    """
    Класс для работы с изображениями, включая загрузку, сохранение, преобразование и масштабирование.
    """

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """
        Загружает изображение по указанному пути и преобразует его из BGR в RGB.

        :param image_path: Путь к изображению.
        :return: Изображение в формате RGB.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Изображение по пути {image_path} не найдено или не может быть загружено.")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def save_image(image_path: str, image: np.ndarray) -> None:
        """
        Сохраняет изображение по указанному пути, преобразуя его из RGB в BGR.

        :param image_path: Путь для сохранения изображения.
        :param image: Изображение в формате RGB.
        """
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    @staticmethod
    def get_rgb(image: np.ndarray, x: int, y: int) -> tuple:
        """
        Возвращает значения RGB для пикселя с координатами (x, y).

        :param image: Изображение в формате RGB.
        :param x: Координата x пикселя.
        :param y: Координата y пикселя.
        :return: Кортеж значений (R, G, B).
        """
        return image[x, y, 0], image[x, y, 1], image[x, y, 2]

    @staticmethod
    def convert_to_24bit_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Преобразует изображение в 24-битное градационное изображение.

        :param image: Входное изображение.
        :return: 24-битное градационное изображение.
        """
        if image.ndim not in [2, 3]:
            raise ValueError("Изображение должно быть 2D или 3D массивом.")

        if image.ndim == 2:
            grayscale = image
        else:
            if image.shape[2] == 4:
                image = image[..., :3]
            if image.shape[2] == 3:
                grayscale = np.dot(image[..., :3], [0.299, 0.587, 0.114])
            else:
                grayscale = image[..., 0]

        grayscale = np.clip(grayscale, 0, 255).astype(np.uint8)
        grayscale_24bit = np.stack([grayscale, grayscale, grayscale], axis=-1)
        return grayscale_24bit

    @staticmethod
    def scale_image_subsampling(image: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Масштабирует изображение с использованием выборки (subsampling).

        :param image: Входное изображение.
        :param scale_factor: Коэффициент масштабирования.
        :return: Масштабированное изображение.
        """
        if scale_factor <= 0:
            raise ValueError("Коэффициент масштабирования должен быть положительным.")

        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)

        scaled_image = image[::int(1 / scale_factor), ::int(1 / scale_factor)]
        return scaled_image

    @staticmethod
    def scale_image_interpolation(image: np.ndarray, scale_factor: float,
                                  interpolation_method: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Масштабирует изображение с использованием интерполяции.

        :param image: Входное изображение.
        :param scale_factor: Коэффициент масштабирования.
        :param interpolation_method: Метод интерполяции (по умолчанию cv2.INTER_LINEAR).
        :return: Масштабированное изображение.
        """
        if scale_factor <= 0:
            raise ValueError("Коэффициент масштабирования должен быть положительным.")

        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)

        scaled_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation_method)
        return scaled_image