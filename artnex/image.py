# MIT License

# Copyright (c) 2023 Yuan-Man

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


class ImageLoaderResizer:
    def __init__(self, target_size=(224, 224)):
        """
        Initializes the class for loading and resizing images.

        Parameters:
        - target_size (tuple): Target image size (height, width).
        """
        self.target_size = target_size

    def load_image(self, image_path):
        """
        Loads an image and returns the resized image data.

        Parameters:
        - image_path (str): Path to the image file.

        Returns:
        - numpy.ndarray: Resized image data.
        """
        try:
            # Use the PIL library to load the image
            image = Image.open(image_path)
            
            # Resize the image
            resized_image = image.resize(self.target_size)
            
            # Convert image data to a NumPy array
            image_array = np.array(resized_image)
            
            return image_array

        except Exception as e:
            print(f"Error loading or resizing image: {e}")
            return None

class ImageCropper:
    def __init__(self, target_size=(224, 224), crop_size=(200, 200)):
        """
        Initializes the class for loading, resizing, and cropping images.

        Parameters:
        - target_size (tuple): Target image size (height, width) after resizing.
        - crop_size (tuple): Size of the cropped region (height, width).
        """
        self.target_size = target_size
        self.crop_size = crop_size

    def load_resize_crop_image(self, image_path):
        """
        Loads an image, resizes it, and returns the cropped image data.

        Parameters:
        - image_path (str): Path to the image file.

        Returns:
        - numpy.ndarray: Cropped and resized image data.
        """
        try:
            # Use the PIL library to load the image
            original_image = Image.open(image_path)
            
            # Resize the image
            resized_image = original_image.resize(self.target_size)
            
            # Crop the center region of the resized image
            left = (self.target_size[1] - self.crop_size[1]) // 2
            top = (self.target_size[0] - self.crop_size[0]) // 2
            right = left + self.crop_size[1]
            bottom = top + self.crop_size[0]
            cropped_image = resized_image.crop((left, top, right, bottom))
            
            # Convert image data to a NumPy array
            image_array = np.array(cropped_image)
            
            return image_array

        except Exception as e:
            print(f"Error loading, resizing, or cropping image: {e}")
            return None

class ImageDenoiserSmoother:
    def __init__(self, target_size=(224, 224), sigma=1.5):
        """
        Initializes the class for loading, resizing, denoising, and smoothing images.

        Parameters:
        - target_size (tuple): Target image size (height, width) after resizing.
        - sigma (float): Standard deviation for Gaussian smoothing filter.
        """
        self.target_size = target_size
        self.sigma = sigma

    def load_resize_denoise_smooth_image(self, image_path):
        """
        Loads an image, resizes it, applies denoising, and returns the smoothed image data.

        Parameters:
        - image_path (str): Path to the image file.

        Returns:
        - numpy.ndarray: Smoothed and resized image data.
        """
        try:
            # Use the PIL library to load the image
            original_image = Image.open(image_path)
            
            # Resize the image
            resized_image = original_image.resize(self.target_size)
            
            # Convert image to NumPy array for denoising and smoothing
            image_array = np.array(resized_image)
            
            # Apply denoising (Gaussian filter)
            denoised_array = Image.fromarray(image_array).filter(ImageFilter.GaussianBlur(self.sigma))
            
            # Convert denoised image data back to NumPy array
            denoised_array = np.array(denoised_array)
            
            return denoised_array

        except Exception as e:
            print(f"Error loading, resizing, denoising, or smoothing image: {e}")
            return None

class ImageNormalizer:
    def __init__(self, target_size=(224, 224), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        """
        Initializes the class for loading, resizing, and normalizing images.

        Parameters:
        - target_size (tuple): Target image size (height, width) after resizing.
        - mean (tuple): Mean values for image normalization (per channel).
        - std (tuple): Standard deviation values for image normalization (per channel).
        """
        self.target_size = target_size
        self.mean = mean
        self.std = std

    def load_resize_normalize_image(self, image_path):
        """
        Loads an image, resizes it, applies normalization, and returns the normalized image data.

        Parameters:
        - image_path (str): Path to the image file.

        Returns:
        - numpy.ndarray: Normalized and resized image data.
        """
        try:
            # Use the PIL library to load the image
            original_image = Image.open(image_path)
            
            # Resize the image
            resized_image = original_image.resize(self.target_size)
            
            # Convert image to NumPy array for normalization
            image_array = np.array(resized_image) / 255.0  # Normalize pixel values to the range [0, 1]
            
            # Normalize image by subtracting mean and dividing by standard deviation
            normalized_array = (image_array - np.array(self.mean)) / np.array(self.std)
            
            return normalized_array

        except Exception as e:
            print(f"Error loading, resizing, or normalizing image: {e}")
            return None

class ImageRotatorTransformer:
    def __init__(self, target_size=(224, 224), rotation_range=(-30, 30), shear_range=(-10, 10)):
        """
        Initializes the class for loading, resizing, rotating, and applying affine transformations to images.

        Parameters:
        - target_size (tuple): Target image size (height, width) after resizing.
        - rotation_range (tuple): Range of rotation angles in degrees.
        - shear_range (tuple): Range of shear angles in degrees.
        """
        self.target_size = target_size
        self.rotation_range = rotation_range
        self.shear_range = shear_range

    def load_resize_rotate_transform_image(self, image_path):
        """
        Loads an image, resizes it, applies rotation and affine transformation, and returns the transformed image data.

        Parameters:
        - image_path (str): Path to the image file.

        Returns:
        - numpy.ndarray: Transformed and resized image data.
        """
        try:
            # Use the PIL library to load the image
            original_image = Image.open(image_path)
            
            # Resize the image
            resized_image = original_image.resize(self.target_size)
            
            # Randomly rotate the image within the specified range
            rotation_angle = random.uniform(self.rotation_range[0], self.rotation_range[1])
            rotated_image = resized_image.rotate(rotation_angle)
            
            # Randomly apply shear transformation within the specified range
            shear_angle = random.uniform(self.shear_range[0], self.shear_range[1])
            affine_transformed_image = rotated_image.transform(
                self.target_size, Image.AFFINE, (1, shear_angle, 0, shear_angle, 1, 0)
            )
            
            # Convert transformed image data to a NumPy array
            image_array = np.array(affine_transformed_image)
            
            return image_array

        except Exception as e:
            print(f"Error loading, resizing, rotating, or transforming image: {e}")
            return None

class ImagePadderCropper:
    def __init__(self, target_size=(224, 224), padding_value=(0, 0, 0)):
        """
        Initializes the class for loading, resizing, padding, and cropping images.

        Parameters:
        - target_size (tuple): Target image size (height, width) after resizing.
        - padding_value (tuple): Tuple representing the RGB values for padding.
        """
        self.target_size = target_size
        self.padding_value = padding_value

    def load_resize_pad_crop_image(self, image_path):
        """
        Loads an image, resizes it, pads it, and returns the cropped image data.

        Parameters:
        - image_path (str): Path to the image file.

        Returns:
        - numpy.ndarray: Padded, resized, and cropped image data.
        """
        try:
            # Use the PIL library to load the image
            original_image = Image.open(image_path)
            
            # Resize the image
            resized_image = original_image.resize(self.target_size)
            
            # Calculate padding values
            pad_width = (
                (self.target_size[0] - resized_image.size[1]) // 2,
                (self.target_size[1] - resized_image.size[0]) // 2
            )
            
            # Pad the image
            padded_image = Image.new('RGB', self.target_size, self.padding_value)
            padded_image.paste(resized_image, pad_width)
            
            # Convert padded image data to a NumPy array
            image_array = np.array(padded_image)
            
            return image_array

        except Exception as e:
            print(f"Error loading, resizing, padding, or cropping image: {e}")
            return None

class ImageBlurrer:
    def __init__(self, target_size=(224, 224), blur_radius=2):
        """
        Initializes the class for loading, resizing, and applying a blur effect to images.

        Parameters:
        - target_size (tuple): Target image size (height, width) after resizing.
        - blur_radius (int): Radius of the blur effect.
        """
        self.target_size = target_size
        self.blur_radius = blur_radius

    def load_resize_apply_blur_image(self, image_path):
        """
        Loads an image, resizes it, applies a blur effect, and returns the blurred image data.

        Parameters:
        - image_path (str): Path to the image file.

        Returns:
        - numpy.ndarray: Blurred and resized image data.
        """
        try:
            # Use the PIL library to load the image
            original_image = Image.open(image_path)
            
            # Resize the image
            resized_image = original_image.resize(self.target_size)
            
            # Apply a blur effect
            blurred_image = resized_image.filter(ImageFilter.GaussianBlur(self.blur_radius))
            
            # Convert blurred image data to a NumPy array
            image_array = np.array(blurred_image)
            
            return image_array

        except Exception as e:
            print(f"Error loading, resizing, or applying blur to image: {e}")
            return None

class ImageSharpener:
    def __init__(self, target_size=(224, 224), sharpen_factor=1.5):
        """
        Initializes the class for loading, resizing, and applying a sharpening effect to images.

        Parameters:
        - target_size (tuple): Target image size (height, width) after resizing.
        - sharpen_factor (float): Factor controlling the strength of the sharpening effect.
        """
        self.target_size = target_size
        self.sharpen_factor = sharpen_factor

    def load_resize_apply_sharpen_image(self, image_path):
        """
        Loads an image, resizes it, applies a sharpening effect, and returns the sharpened image data.

        Parameters:
        - image_path (str): Path to the image file.

        Returns:
        - numpy.ndarray: Sharpened and resized image data.
        """
        try:
            # Use the PIL library to load the image
            original_image = Image.open(image_path)
            
            # Resize the image
            resized_image = original_image.resize(self.target_size)
            
            # Apply a sharpening effect
            sharpened_image = resized_image.filter(ImageFilter.UnsharpMask(radius=2, percent=self.sharpen_factor))
            
            # Convert sharpened image data to a NumPy array
            image_array = np.array(sharpened_image)
            
            return image_array

        except Exception as e:
            print(f"Error loading, resizing, or applying sharpening to image: {e}")
            return None

class ImageDistorterWarper:
    def __init__(self, target_size=(224, 224), distortion_factor=0.2):
        """
        Initializes the class for loading, resizing, and applying distortion and warping to images.

        Parameters:
        - target_size (tuple): Target image size (height, width) after resizing.
        - distortion_factor (float): Factor controlling the strength of the distortion effect.
        """
        self.target_size = target_size
        self.distortion_factor = distortion_factor

    def distort_image(self, image):
        """
        Distorts an image by applying random displacement to control points.

        Parameters:
        - image (PIL.Image): The input image.

        Returns:
        - PIL.Image: Distorted image.
        """
        width, height = image.size
        control_points = [
            (0, 0),
            (width - 1, 0),
            (0, height - 1),
            (width - 1, height - 1)
        ]

        # Randomly displace control points
        displaced_points = [(x + random.uniform(-1, 1) * self.distortion_factor * width,
                             y + random.uniform(-1, 1) * self.distortion_factor * height) for x, y in control_points]

        # Apply perspective transformation
        warped_image = image.transform(
            self.target_size, Image.PERSPECTIVE, displaced_points,
            Image.BICUBIC, fill=0
        )

        return warped_image

    def load_resize_distort_warp_image(self, image_path):
        """
        Loads an image, resizes it, applies distortion and warping, and returns the distorted and warped image data.

        Parameters:
        - image_path (str): Path to the image file.

        Returns:
        - numpy.ndarray: Distorted, warped, and resized image data.
        """
        try:
            # Use the PIL library to load the image
            original_image = Image.open(image_path)
            
            # Resize the image
            resized_image = original_image.resize(self.target_size)
            
            # Distort and warp the image
            distorted_warped_image = self.distort_image(resized_image)
            
            # Convert distorted and warped image data to a NumPy array
            image_array = np.array(distorted_warped_image)
            
            return image_array

        except Exception as e:
            print(f"Error loading, resizing, or applying distortion and warping to image: {e}")
            return None

class ImageRandomEraser:
    def __init__(self, target_size=(224, 224), erasing_prob=0.5, erasing_ratio=(0.02, 0.4), aspect_ratio_range=(0.3, 3.0)):
        """
        Initializes the class for loading, resizing, and applying random erasing to images.

        Parameters:
        - target_size (tuple): Target image size (height, width) after resizing.
        - erasing_prob (float): Probability of applying random erasing to an image.
        - erasing_ratio (tuple): Range of ratios for the area of erased region relative to the input image.
        - aspect_ratio_range (tuple): Range of aspect ratios for the erased region.
        """
        self.target_size = target_size
        self.erasing_prob = erasing_prob
        self.erasing_ratio = erasing_ratio
        self.aspect_ratio_range = aspect_ratio_range

    def random_erase_image(self, image):
        """
        Applies random erasing to an image.

        Parameters:
        - image (PIL.Image): The input image.

        Returns:
        - PIL.Image: Image with random erasing applied.
        """
        if random.uniform(0, 1) > self.erasing_prob:
            return image

        width, height = image.size
        area = width * height
        target_area = random.uniform(*self.erasing_ratio) * area
        aspect_ratio = random.uniform(*self.aspect_ratio_range)

        # Calculate the dimensions of the erased region
        erase_width = int(round(np.sqrt(target_area * aspect_ratio)))
        erase_height = int(round(np.sqrt(target_area / aspect_ratio)))

        # Randomly position the erased region
        erase_x = random.randint(0, width - erase_width)
        erase_y = random.randint(0, height - erase_height)

        # Create a mask for the erased region
        erase_mask = Image.new('L', (erase_width, erase_height), 255)

        # Paste the mask onto the image to perform the erasing
        image.paste(erase_mask, (erase_x, erase_y))

        return image

    def load_resize_random_erase_image(self, image_path):
        """
        Loads an image, resizes it, and applies random erasing, then returns the processed image data.

        Parameters:
        - image_path (str): Path to the image file.

        Returns:
        - numpy.ndarray: Processed and resized image data.
        """
        try:
            # Use the PIL library to load the image
            original_image = Image.open(image_path)
            
            # Resize the image
            resized_image = original_image.resize(self.target_size)
            
            # Apply random erasing
            erased_image = self.random_erase_image(resized_image)
            
            # Convert processed image data to a NumPy array
            image_array = np.array(erased_image)
            
            return image_array

        except Exception as e:
            print(f"Error loading, resizing, or applying random erasing to image: {e}")
            return None

class ImageEdgeDetector:
    def __init__(self, target_size=(224, 224), edge_detection_method='sobel'):
        """
        Initializes the class for loading, resizing, and applying edge detection to images.

        Parameters:
        - target_size (tuple): Target image size (height, width) after resizing.
        - edge_detection_method (str): Edge detection method ('sobel' or 'prewitt').
        """
        self.target_size = target_size
        self.edge_detection_method = edge_detection_method.lower()

    def apply_edge_detection(self, image):
        """
        Applies edge detection to an image.

        Parameters:
        - image (PIL.Image): The input image.

        Returns:
        - PIL.Image: Image with edge detection applied.
        """
        if self.edge_detection_method == 'sobel':
            edge_detected_image = image.filter(ImageFilter.FIND_EDGES)
        elif self.edge_detection_method == 'prewitt':
            edge_detected_image = image.filter(ImageFilter.FIND_EDGES)
        else:
            raise ValueError("Invalid edge detection method. Use 'sobel' or 'prewitt'.")

        return edge_detected_image

    def load_resize_apply_edge_detection(self, image_path):
        """
        Loads an image, resizes it, and applies edge detection, then returns the processed image data.

        Parameters:
        - image_path (str): Path to the image file.

        Returns:
        - numpy.ndarray: Processed and resized image data.
        """
        try:
            # Use the PIL library to load the image
            original_image = Image.open(image_path)
            
            # Resize the image
            resized_image = original_image.resize(self.target_size)
            
            # Apply edge detection
            edge_detected_image = self.apply_edge_detection(resized_image)
            
            # Convert processed image data to a NumPy array
            image_array = np.array(edge_detected_image)
            
            return image_array

        except Exception as e:
            print(f"Error loading, resizing, or applying edge detection to image: {e}")
            return None

class ImageLocalContrastEnhancer:
    def __init__(self, target_size=(224, 224), contrast_factor=1.5):
        """
        Initializes the class for loading, resizing, and applying local contrast enhancement to images.

        Parameters:
        - target_size (tuple): Target image size (height, width) after resizing.
        - contrast_factor (float): Factor controlling the strength of the contrast enhancement.
        """
        self.target_size = target_size
        self.contrast_factor = contrast_factor

    def enhance_local_contrast(self, image):
        """
        Enhances local contrast in an image.

        Parameters:
        - image (PIL.Image): The input image.

        Returns:
        - PIL.Image: Image with enhanced local contrast.
        """
        # Apply a high-pass filter to extract details
        details_image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

        # Enhance the details and blend with the original image
        enhanced_image = ImageEnhance.Contrast(image).enhance(self.contrast_factor)
        enhanced_image = Image.blend(enhanced_image, details_image, 0.5)

        return enhanced_image

    def load_resize_enhance_local_contrast(self, image_path):
        """
        Loads an image, resizes it, and enhances local contrast, then returns the processed image data.

        Parameters:
        - image_path (str): Path to the image file.

        Returns:
        - numpy.ndarray: Processed and resized image data.
        """
        try:
            # Use the PIL library to load the image
            original_image = Image.open(image_path)
            
            # Resize the image
            resized_image = original_image.resize(self.target_size)
            
            # Enhance local contrast
            contrast_enhanced_image = self.enhance_local_contrast(resized_image)
            
            # Convert processed image data to a NumPy array
            image_array = np.array(contrast_enhanced_image)
            
            return image_array

        except Exception as e:
            print(f"Error loading, resizing, or enhancing local contrast in image: {e}")
            return None




