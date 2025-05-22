import cv2
import numpy as np
from scipy.signal import correlate2d

from PyQt5.QtGui import  QPainterPath
from PyQt5.QtCore import QRect


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
    def save_image(image_path: str, image: np.ndarray, is_select_mode=False) -> None:
        """
        Сохраняет изображение по указанному пути.
        Если is_select_mode=True, заменяет чёрный фон на прозрачность (alpha=0).

        :param image_path: Путь для сохранения изображения.
        :param image: Изображение в формате RGB.
        :param is_select_mode: Флаг, означает, что нужно сохранить с прозрачным фоном.
        """
        if is_select_mode:
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Для сохранения с прозрачностью нужно RGB-изображение.")

            alpha = np.any(image != 0, axis=2).astype(np.uint8) * 255

            rgba = np.dstack((image, alpha))

            bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)

            if not image_path.lower().endswith(".png"):
                image_path += ".png"

            cv2.imwrite(image_path, bgra)
        else:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_path, bgr)


    @staticmethod
    def ensure_rgb(image: np.ndarray) -> np.ndarray:
        """
        Гарантирует, что изображение в формате RGB.

        :param image: Входное изображение (grayscale, RGB, RGBA).
        :return: RGB изображение (H, W, 3).
        """
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3:
            if image.shape[2] == 1:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:
                return image
        raise ValueError("Unsupported image format.")


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
        if scale_factor <= 0 or scale_factor >= 1:
            raise ValueError("Коэффициент масштабирования должен быть в диапазоне (0, 1) для уменьшения изображения.")

        step = int(1 / scale_factor)
        scaled_image = image[::step, ::step]
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


    @staticmethod
    def contrast_stretching(image: np.ndarray, roi_image: np.ndarray = None) -> np.ndarray:
        """
        Применяет контрастирование (stretching) на всём изображении или только в зоне интереса.

        :param image: RGB или grayscale изображение (np.uint8).
        :param roi_image: Вырезанная область интереса (RGB или Grayscale), такого же размера, как image.
                          Пиксели с ненулевыми значениями считаются зоной интереса.
        :return: Контрастированное изображение.
        """
        if image is None:
            raise ValueError("Изображение не должно быть None.")

        if roi_image is not None:
            if roi_image.shape[:2] != image.shape[:2]:
                raise ValueError("Размер roi_image должен совпадать с изображением.")
            if roi_image.ndim == 3:
                mask = (np.any(roi_image != 0, axis=2)).astype(np.uint8) * 255
            else:
                mask = (roi_image != 0).astype(np.uint8) * 255
        else:
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

        output = image.copy()

        if image.ndim == 3:
            for c in range(3):
                channel = image[:, :, c]
                roi_values = channel[mask == 255]

                if roi_values.size == 0:
                    continue

                m, M = np.min(roi_values), np.max(roi_values)
                if M == m:
                    continue

                stretched = np.clip((channel.astype(np.float32) - m) * (255.0 / (M - m)), 0, 255).astype(np.uint8)
                output[:, :, c][mask == 255] = stretched[mask == 255]
        else:
            channel = image
            roi_values = channel[mask == 255]

            if roi_values.size == 0:
                return image.copy()

            m, M = np.min(roi_values), np.max(roi_values)
            if M == m:
                return image.copy()

            stretched = np.clip((channel.astype(np.float32) - m) * (255.0 / (M - m)), 0, 255).astype(np.uint8)
            output[mask == 255] = stretched[mask == 255]

        return output


    @staticmethod
    def quantization(image: np.ndarray, levels: int) -> np.ndarray:
        """
        Применяет квантование к изображению, уменьшая количество уровней яркости.

        :param image: Входное изображение.
        :param levels: Количество уровней яркости (например, 2 для черно-белого).
        :return: Квантованное изображение.
        """
        if levels < 2 or levels > 256:
            raise ValueError("Количество уровней должно быть в диапазоне [2, 256].")

        if image.ndim == 3:
            channels = cv2.split(image)
            quantized_channels = []
            for channel in channels:
                quantized_channel = np.floor_divide(channel, 256 // levels) * (256 // levels)
                quantized_channels.append(quantized_channel)
            return cv2.merge(quantized_channels)
        else:
            return np.floor_divide(image, 256 // levels) * (256 // levels)


    @staticmethod
    def get_roi_mask_from_region(image: np.ndarray, region: np.ndarray) -> np.ndarray:
        """
        Возвращает бинарную маску (0/255), где непустые пиксели из `region` обозначают зону интереса.
        Поддерживает grayscale и RGB `region`. Размер `region` должен совпадать с `image`.

        :param image: Оригинальное изображение (используется для размера).
        :param region: Вырезанный ROI на черном фоне, такого же размера как image.
        :return: Бинарная маска (0 — фон, 255 — зона интереса).
        """
        if image.shape[:2] != region.shape[:2]:
            raise ValueError("Размер region должен совпадать с изображением.")

        if region.ndim == 3:
            nonzero_mask = np.any(region != 0, axis=2)
        else:
            nonzero_mask = region != 0

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[nonzero_mask] = 255
        return mask


    @staticmethod
    def extract_roi_from_region(image: np.ndarray, region) -> np.ndarray:
        """
        Извлекает область интереса (ROI) из изображения, сохраняя исходный размер.
        Пиксели вне региона обнуляются (становятся чёрными).

        :param image: RGB или grayscale изображение (np.ndarray).
        :param region: QRect или QPainterPath.
        :return: ROI-изображение того же размера, где вне региона — чёрный фон.
        """
        if image is None:
            return None

        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        if isinstance(region, QRect):
            x, y, w, h = region.x(), region.y(), region.width(), region.height()
            mask[y:y + h, x:x + w] = 255

        elif isinstance(region, QPainterPath):
            polygon = region.toFillPolygon().toPolygon()
            pts = np.array([[p.x(), p.y()] for p in polygon], dtype=np.int32)

            if pts.shape[0] >= 3:
                cv2.fillPoly(mask, [pts], color=255)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    mask = np.zeros_like(mask)
                    cv2.drawContours(mask, [largest_contour], -1, color=255, thickness=-1)
            else:
                return np.zeros_like(image)

        else:
            return np.zeros_like(image)

        if image.ndim == 3:
            roi = cv2.bitwise_and(image, image, mask=mask)
        else:
            roi = cv2.bitwise_and(image, image, mask=mask)

        return roi


    @staticmethod
    def get_min_max(image: np.ndarray, roi_image: np.ndarray = None):
        if roi_image is not None:
            if roi_image.shape != image.shape:
                raise ValueError("Размер roi_image должен совпадать с изображением.")
            mask = Utils.get_roi_mask_from_region(image, roi_image)
        else:
            mask = None

        if image.ndim == 3:
            result = {}
            for c in range(3):
                values = image[:, :, c][mask == 255] if mask is not None else image[:, :, c].flatten()
                if values.size > 0:
                    result[f'channel_{c}_min'] = int(np.min(values))
                    result[f'channel_{c}_max'] = int(np.max(values))
            return result
        else:
            values = image[mask == 255] if mask is not None else image.flatten()
            return int(np.min(values)), int(np.max(values))


    @staticmethod
    def get_mean_std(image: np.ndarray, roi_image: np.ndarray = None):
        if roi_image is not None:
            if roi_image.shape != image.shape:
                raise ValueError("Размер roi_image должен совпадать с изображением.")
            mask = Utils.get_roi_mask_from_region(image, roi_image)
        else:
            mask = None

        if image.ndim == 3:
            result = {}
            for c in range(3):
                values = image[:, :, c][mask == 255] if mask is not None else image[:, :, c].flatten()
                if values.size > 0:
                    result[f'channel_{c}_mean'] = float(np.mean(values))
                    result[f'channel_{c}_std'] = float(np.std(values))
            return result
        else:
            values = image[mask == 255] if mask is not None else image.flatten()
            return float(np.mean(values)), float(np.std(values))


    @staticmethod
    def get_histogram(image: np.ndarray, roi_image: np.ndarray = None):
        if roi_image is not None:
            if roi_image.shape != image.shape:
                raise ValueError("Размер roi_image должен совпадать с изображением.")
            mask = Utils.get_roi_mask_from_region(image, roi_image)
        else:
            mask = None

        hist_range = [0, 256]
        hist_size = 256

        if image.ndim == 3:
            histograms = {}
            for c in range(3):
                channel = image[:, :, c]
                hist = cv2.calcHist([channel], [0], mask, [hist_size], hist_range)
                histograms[f'channel_{c}'] = hist.flatten()
            return histograms
        else:
            hist = cv2.calcHist([image], [0], mask, [hist_size], hist_range)
            return hist.flatten()


    @staticmethod
    def smooth_roi(image: np.ndarray, roi_image: np.ndarray, radius: int) -> np.ndarray:
        """
        Сглаживает значения пикселей внутри ROI по квадратной окрестности радиуса r.

        :param image: Входное изображение (np.uint8), RGB или grayscale.
        :param roi_image: Изображение ROI (той же формы), где ненулевые пиксели — зона интереса.
        :param radius: Радиус квадратной окрестности.
        :return: Изображение с применённым сглаживанием в ROI.
        """

        if roi_image is None:
            height, width = image.shape[:2]
            roi_image = np.ones((height, width), dtype=np.uint8) * 255

        if image.shape[:2] != roi_image.shape[:2]:
            raise ValueError("Изображения должны быть одного размера.")

        mask = Utils.get_roi_mask_from_region(image, roi_image)
        kernel_size = 2 * radius + 1
        output = image.copy()

        if image.ndim == 3:
            for c in range(3):
                channel = image[:, :, c]
                smoothed = cv2.blur(channel, (kernel_size, kernel_size))
                output[:, :, c][mask == 255] = smoothed[mask == 255]
        else:
            smoothed = cv2.blur(image, (kernel_size, kernel_size))
            output[mask == 255] = smoothed[mask == 255]

        return output


    @staticmethod
    def estimate_white_noise_std(image: np.ndarray, roi_image: np.ndarray = None) -> float:
        """
        Оценивает уровень аддитивного белого шума как стандартное отклонение высокочастотной составляющей.
        """

        if roi_image is None:
            height, width = image.shape[:2]
            mask = np.ones((height, width), dtype=np.uint8) * 255
        else:
            if roi_image.shape[:2] != image.shape[:2]:
                raise ValueError("Размер roi_image должен совпадать с изображением.")
            mask = Utils.get_roi_mask_from_region(image, roi_image)


        if image.ndim == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image.copy()

        blurred = cv2.GaussianBlur(image_gray, (3, 3), 0)
        noise = cv2.absdiff(image_gray, blurred)

        if mask is not None:
            values = noise[mask == 255]
        else:
            values = noise.flatten()

        return float(np.std(values)) if values.size > 0 else 0.0


    @staticmethod
    def reduce_white_noise(image: np.ndarray, roi_image: np.ndarray = None, strength: int = 5) -> np.ndarray:
        """
        Уменьшает аддитивный белый шум путём сглаживания.
        """

        if roi_image is None:
            height, width = image.shape[:2]
            mask = np.ones((height, width), dtype=np.uint8) * 255
        else:
            if roi_image.shape[:2] != image.shape[:2]:
                raise ValueError("Размер roi_image должен совпадать с изображением.")
            mask = Utils.get_roi_mask_from_region(image, roi_image)

        output = image.copy()
        if image.ndim == 3:
            for c in range(3):
                channel = image[:, :, c]
                denoised = cv2.fastNlMeansDenoising(channel, None, h=strength, templateWindowSize=7,
                                                    searchWindowSize=21)
                if mask is not None:
                    output[:, :, c][mask == 255] = denoised[mask == 255]
                else:
                    output[:, :, c] = denoised
        else:
            denoised = cv2.fastNlMeansDenoising(image, None, h=strength, templateWindowSize=7, searchWindowSize=21)
            if mask is not None:
                output[mask == 255] = denoised[mask == 255]
            else:
                output = denoised

        return output


    @staticmethod
    def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """
        Поворачивает изображение на указанный угол относительно центра, без обрезки,
        заполняя пустые области белым цветом.

        :param image: Входное изображение (np.ndarray).
        :param angle: Угол поворота (в градусах).
        :return: Повернутое изображение.
        """
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        if image.ndim == 3 and image.shape[2] == 3:
            border_color = (255, 255, 255)
        elif image.ndim == 2:
            border_color = 255
        else:
            raise ValueError("Неподдерживаемый формат изображения.")

        rotated = cv2.warpAffine(
            image, M, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_color
        )

        return rotated


    @staticmethod
    def crop_to_roi(select_zone: np.ndarray) -> np.ndarray:
        """
        Обрезает изображение до зоны интереса, исключая черный фон.
        Чёрный фон внутри ограничивающего прямоугольника заменяется на белый.
        :param select_zone: Вырезанная область (RGB или grayscale) на черном фоне.
        :return: Обрезанное изображение с белым фоном.
        """
        if select_zone is None or select_zone.size == 0:
            return select_zone

        if select_zone.ndim == 3:
            mask = np.any(select_zone != 0, axis=2).astype(np.uint8) * 255
        else:
            mask = (select_zone != 0).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            shape = (1, 1, select_zone.shape[2]) if select_zone.ndim == 3 else (1, 1)
            return np.ones(shape, dtype=select_zone.dtype) * 255  # Возвращаем белый пиксель

        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

        cropped = select_zone[y:y + h, x:x + w]
        cropped_mask = mask[y:y + h, x:x + w]

        if select_zone.ndim == 3:
            white_bg = np.ones_like(cropped) * 255
            cropped = np.where(cropped_mask[:, :, None] == 255, cropped, white_bg)
        else:
            white_bg = np.ones_like(cropped) * 255
            cropped = np.where(cropped_mask == 255, cropped, white_bg)

        return cropped


    @staticmethod
    def get_pixel_value(image: np.ndarray, x: int, y: int):
        """
        Возвращает амплитуду пикселя по координатам (x, y).
        :param image: Изображение (grayscale или RGB).
        :param x: Координата X.
        :param y: Координата Y.
        :return: Значение пикселя (int или tuple).
        """
        if not (0 <= y < image.shape[0]) or not (0 <= x < image.shape[1]):
            raise ValueError("Координаты вне границ изображения.")

        if image.ndim == 2:
            return int(image[y, x])
        elif image.ndim == 3:
            return tuple(int(v) for v in image[y, x])
        else:
            raise ValueError("Неподдерживаемый формат изображения.")


    @staticmethod
    def set_pixel_value(image: np.ndarray, x: int, y: int, value):
        """
        Устанавливает амплитуду пикселя по координатам (x, y).
        :param image: Изображение (grayscale или RGB).
        :param x: Координата X.
        :param y: Координата Y.
        :param value: Новое значение (int или tuple).
        """
        if not (0 <= y < image.shape[0]) or not (0 <= x < image.shape[1]):
            raise ValueError("Координаты вне границ изображения.")

        if image.ndim == 2:
            image[y, x] = value
        elif image.ndim == 3:
            image[y, x] = tuple(value)
        else:
            raise ValueError("Неподдерживаемый формат изображения.")
        return image


    @staticmethod
    def generate_piecewise_grid(image: np.ndarray, block_size: int = 32) -> np.ndarray:
        """
        Генерирует изображение с кусочно-постоянными амплитудами, разбивая исходное на сетку блоков.

        :param image: Исходное изображение (используется размер).
        :param block_size: Размер одного блока в пикселях.
        :return: Изображение с мозаикой из блоков с постоянной амплитудой.
        """
        h, w = image.shape[:2]
        is_rgb = image.ndim == 3

        output = np.zeros_like(image)

        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                y_end = min(y + block_size, h)
                x_end = min(x + block_size, w)

                if is_rgb:
                    block = image[y:y_end, x:x_end, :]
                    mean_color = tuple(int(np.mean(block[:, :, c])) for c in range(3))
                    output[y:y_end, x:x_end] = mean_color
                else:
                    block = image[y:y_end, x:x_end]
                    mean_val = int(np.mean(block))
                    output[y:y_end, x:x_end] = mean_val

        return output


    @staticmethod
    def estimate_correlation_function(image: np.ndarray, roi_mask: np.ndarray = None) -> np.ndarray:
        """
        Оценивает двумерную корреляционную функцию изображения или его ROI.

        :param image: Входное изображение (градационное или RGB, np.uint8).
        :param roi_mask: Бинарная маска зоны интереса (0 и 255), такого же размера как изображение.
        :return: Нормализованная корреляционная функция.
        """
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if roi_mask is not None:
            if roi_mask.shape != image.shape:
                raise ValueError("Размер маски должен совпадать с изображением.")
            image = np.where(roi_mask == 255, image, 0)

        image = image.astype(np.float32)
        image -= np.mean(image[image != 0])  # Центрирование по среднему

        correlation = correlate2d(image, image, mode='full', boundary='fill', fillvalue=0)
        correlation /= np.max(np.abs(correlation))  # Нормализация

        return correlation


    @staticmethod
    def generate_random_scene(h: int, w: int, mode: str = 'uniform', params: dict = None,
                              channels: int = 1) -> np.ndarray:
        """
        Генерирует изображение сцены размером h×w с независимыми случайными амплитудами.

        :param h: Высота изображения (кол-во строк).
        :param w: Ширина изображения (кол-во столбцов).
        :param mode: Тип распределения: 'uniform' или 'normal'.
        :param params: Параметры распределения:
                       - для 'uniform': {'a': ..., 'b': ...}
                       - для 'normal': {'mean': ..., 'std': ...}
        :param channels: Кол-во каналов (1 — grayscale, 3 — RGB).
        :return: Сгенерированное изображение (np.uint8).
        """
        if params is None:
            params = {}

        if mode == 'uniform':
            a = params.get('a', 0)
            b = params.get('b', 255)
            data = np.random.uniform(a, b, size=(h, w, channels) if channels == 3 else (h, w))
        elif mode == 'normal':
            mean = params.get('mean', 127)
            std = params.get('std', 20)
            data = np.random.normal(mean, std, size=(h, w, channels) if channels == 3 else (h, w))
        else:
            raise ValueError("Неподдерживаемый режим: выберите 'uniform' или 'normal'.")

        data = np.clip(data, 0, 255).astype(np.uint8)
        return data


    @staticmethod
    def generate_smoothed_scene(h: int, w: int, r: int, mean: float = 127, std: float = 20) -> tuple:
        """
        Генерирует изображение сцены с нормальным распределением и сглаживанием по квадратной окрестности.

        :param h: Высота изображения (строки).
        :param w: Ширина изображения (столбцы).
        :param r: Радиус квадратной окрестности для сглаживания.
        :param mean: Среднее значение нормального распределения.
        :param std: Стандартное отклонение нормального распределения.
        :return: Кортеж (сглаженное изображение [np.uint8], оценка среднего, оценка std).
        """
        # 1. Выделение памяти + генерация сцены с шумом
        raw = np.random.normal(mean, std, size=(h, w)).astype(np.float32)

        # 2. Сглаживание методом скользящего суммирования (блюр по окрестности)
        ksize = 2 * r + 1
        smoothed = cv2.blur(raw, ksize=(ksize, ksize))

        # 3. Квантование результата в диапазон [0, 255]
        smoothed_clipped = np.clip(smoothed, 0, 255).astype(np.uint8)

        # 4. Оценка среднего и стандартного отклонения по сглаженному изображению
        m_est = float(np.mean(smoothed_clipped))
        std_est = float(np.std(smoothed_clipped))

        return smoothed_clipped, m_est, std_est
