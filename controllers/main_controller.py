import cv2
import numpy as np
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QObject

from views.views_enums import ScaleMode
from views.main_view import MainWindow
from models.image_model import ImageModel
from controllers.utils import Utils
from controllers.show_utils import UtilsWithDisplay
from views.views_enums import CalcMode


class MainController(QObject):

    signal_send_image = pyqtSignal(np.ndarray)
    signal_send_rgb = pyqtSignal(int, int, int)

    def __init__(self, image_model: ImageModel, view: MainWindow):
        super().__init__()
        self.image_model = image_model
        self.view = view
        self.connect_signals()
        self.connect_slot()


    def connect_signals(self):
        self.view.signal_open_image.connect(self.image_sender)
        self.view.signal_save_image.connect(self.save_image)
        self.view.signal_undo_image.connect(self.image_model.get_undo_image)
        self.view.signal_redo_image.connect(self.image_model.get_redo_image)
        self.view.signal_coordinates.connect(self.get_image_rgb)
        self.view.signal_scale_image.connect(self.scale_image)
        self.view.signal_grayscale_image.connect(self.convert_grayscale)
        self.view.signal_contrast_image.connect(self.contrast_stretching_metod)
        self.view.signal_quantized_image.connect(self.quantization_metod)
        self.view.signal_calc_statistics.connect(self.calc_statistics_metod)
        self.view.signal_send_selected_zone.connect(self.put_select_zone)
        self.view.smoothing_dialog.signal_smoothing_image.connect(self.smoothing_image)
        self.view.denoise_dialog.signal_denoise_image.connect(self.denoise_image)
        self.view.denoise_dialog.signal_estimate_noise.connect(self.measure_noise)
        self.view.rotate_dialog.signal_rotate_image.connect(self.rotate_image)
        self.view.ui.btn_crop.clicked.connect(self.crop_image)
        self.view.pixel_edit_dialog.signal_get_amplitude.connect(self.get_amplitude)
        self.view.pixel_edit_dialog.signal_set_amplitude.connect(self.set_amplitude)
        self.view.pixel_edit_dialog.signal_build_piecewise.connect(self.generate_piecewise_map)
        self.view.random_scene_dialog.signal_generate_scene.connect(self.create_random_scene)
        self.view.smoothed_scene_dialog.signal_generate_smoothed.connect(self.create_smoothed_scene)
        self.view.ui.btn_correlation.clicked.connect(self.calculate_correlation_fun)
        self.view.ui.btn_segmentation.clicked.connect(self.segmentation)
        self.view.ui.btn_hypothesis.clicked.connect(self.hypothesis_testing)
        self.view.ui.btn_complex.clicked.connect(self.segmentation_complex)
        self.view.signal_clear_selected_zone.connect(self.clear_select_zone)
        self.view.object_projection_dialog.signal_run_projection.connect(self.add_projection)

        self.image_model.signal_image_change.connect(self.view.put_image)

    def connect_slot(self):
        self.signal_send_image.connect(self.image_model.image_set)
        self.signal_send_rgb.connect(self.view.put_rgb_in_point)


#Methods
#-----------------------------------------------------------------


    def image_sender(self, file_name: str):
        image = Utils.load_image(file_name)
        if image is not None:
            self.signal_send_image.emit(image)


    def save_image(self, file_name: str):
        image = self.image_model.get_current_image()
        if image is not None:
            Utils.save_image(file_name, image)


    def get_image_rgb(self, x, y):
        image = self.image_model.get_current_image()
        if image is not None:
            rgb_image = Utils.ensure_rgb(image)
            r, g, b = Utils.get_rgb(rgb_image, x, y)
            self.signal_send_rgb.emit(r, g, b)


    def scale_image(self, metod, ratio):
        image = self.image_model.get_current_image()
        if image is not None:
            if metod == ScaleMode.BY_SELECTION.value:
                scaled_image = Utils.scale_image_subsampling(image, ratio)
            else:
                scaled_image = Utils.scale_image_interpolation(image, ratio)

            self.signal_send_image.emit(scaled_image)


    def convert_grayscale(self):
        image = self.image_model.get_current_image()
        if image is not None:
            grayscale_image = Utils.convert_to_24bit_grayscale(image)
            self.signal_send_image.emit(grayscale_image)


    def contrast_stretching_metod(self):
        image = self.image_model.get_current_image()
        if image is not None:
            select_zone = self.image_model.get_select_zone()
            contrast_stretching_image = Utils.contrast_stretching(image, select_zone)
            self.signal_send_image.emit(contrast_stretching_image)


    def quantization_metod(self, levels=2):
        image = self.image_model.get_current_image()
        if image is not None:
            quantization_image = Utils.quantization(image, levels)
            self.signal_send_image.emit(quantization_image)


    def calc_statistics_metod(self, calc_mode: int, is_selected_zone: bool):
        image = self.image_model.get_current_image()
        select_zone = self.image_model.get_select_zone()

        if image is not None:
            if select_zone is not None:
                # if is_selected_zone:
                if CalcMode.MIN_MAX_AMP.value == calc_mode:
                    UtilsWithDisplay.show_min_max(image, select_zone)
                elif CalcMode.MEAN_STD.value == calc_mode:
                    UtilsWithDisplay.show_mean_std(image, select_zone)
                elif CalcMode.HISTOGRAM.value == calc_mode:
                    UtilsWithDisplay.show_histogram(image, select_zone)
            else:
                if CalcMode.MIN_MAX_AMP.value == calc_mode:
                    UtilsWithDisplay.show_min_max(image)
                elif CalcMode.MEAN_STD.value == calc_mode:
                    UtilsWithDisplay.show_mean_std(image)
                elif CalcMode.HISTOGRAM.value == calc_mode:
                    UtilsWithDisplay.show_histogram(image)


    def put_select_zone(self, selected_zone):
        image = self.image_model.get_current_image()
        if image is not None:
            selected_zone_image = Utils.extract_roi_from_region(image, selected_zone)
            self.image_model.put_select_zone(selected_zone_image)


    def smoothing_image(self, radius):
        image = self.image_model.get_current_image()
        if image is not None:
            select_zone = self.image_model.get_select_zone()
            smooth_image = Utils.smooth_roi(image, select_zone, radius)
            self.signal_send_image.emit(smooth_image)


    def denoise_image(self, level):
        image = self.image_model.get_current_image()
        if image is not None:
            select_zone = self.image_model.get_select_zone()
            denoise_image = Utils.reduce_white_noise(image, select_zone, level)
            self.signal_send_image.emit(denoise_image)


    def measure_noise(self):
        image = self.image_model.get_current_image()
        if image is not None:
            select_zone = self.image_model.get_select_zone()
            UtilsWithDisplay.show_noise_estimation(image, select_zone)


    def rotate_image(self, angle):
        image = self.image_model.get_current_image()
        if image is not None:
            rotate_image = Utils.rotate_image(image, angle)
            self.signal_send_image.emit(rotate_image)


    def crop_image(self):
        select_zone = self.image_model.get_select_zone()
        if select_zone is not None:
            crop = Utils.crop_to_roi(select_zone)
            self.signal_send_image.emit(crop)


    def get_amplitude(self, x, y):
        image = self.image_model.get_current_image()
        if image is not None:
            pixel = Utils.get_pixel_value(image, x, y)
            print(pixel)
            self.view.pixel_edit_dialog.set_amplitude_value(pixel)


    def set_amplitude(self, x, y, rgb):
        image = self.image_model.get_current_image()
        if image is not None:
            image = Utils.set_pixel_value(image, x, y, rgb)
            self.signal_send_image.emit(image)


    def generate_piecewise_map(self, block_size):
        image = self.image_model.get_current_image()
        if image is not None:
            image = Utils.generate_piecewise_grid(image, block_size)
            self.signal_send_image.emit(image)


    def create_random_scene(self, h, w, mode, params, channels):
        scene = Utils.generate_random_scene(h, w, mode=mode, params=params, channels=channels)
        if scene is not None:
            self.signal_send_image.emit(scene)


    def create_smoothed_scene(self, h, w, r, mean, std):
        smoothed, m_est, std_est = Utils.generate_smoothed_scene(h=h, w=w, r=r, mean=mean, std=std)
        if smoothed is not None:
            self.signal_send_image.emit(smoothed)


    def calculate_correlation_fun(self):
        image = self.image_model.get_current_image()
        select_zone = self.image_model.get_select_zone()
        if image is not None:
            UtilsWithDisplay.show_correlation_function(None, image, select_zone)


    def segmentation(self):
        image = self.image_model.get_current_image()
        select_zone = self.image_model.get_select_zone()
        if image is not None:
            classified_mask, prob_correct = Utils.segment_simple_roi(image, select_zone)
            self.signal_send_image.emit(classified_mask)
            UtilsWithDisplay.show_segmentation_accuracy(prob_correct)


    def hypothesis_testing(self):
        image = self.image_model.get_current_image()
        select_zone = self.image_model.get_select_zone()
        if image is not None:
            UtilsWithDisplay.show_uniformity_test_result(image, select_zone)


    def segmentation_complex(self):
        image = self.image_model.get_current_image()
        select_zone = self.image_model.get_select_zone()
        if image is not None:
            classified_mask = Utils.segment_complex_roi(image, select_zone)
            self.signal_send_image.emit(classified_mask)


    def clear_select_zone(self):
        self.image_model.clear_select_zone()


    def add_projection(self, mask_path, mean, std, radius, distribution):
        image = self.image_model.get_current_image()
        if image is not None:
            result = Utils.add_object_projection_from_file(
                base_image=image,
                mask_path=mask_path,
                mean=mean,
                std=std,
                radius=radius,
                distribution=distribution
            )

            self.signal_send_image.emit(result)