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


import os
import time
import urllib.request
import requests
import logging
import numpy as np
import onnx
import onnxruntime as ort


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url, file_name=None, overwrite=False, show_progress=True):
    """
    Downloads a file from the specified URL.

    Parameters:
    - url (str): The URL of the file to be downloaded.
    - file_name (str, optional): The name to be used for the downloaded file. If not provided,
      it uses the part of the URL after the last slash as the file name.
    - overwrite (bool, optional): If True, overwrite the file if it already exists. Default is False.
    - show_progress (bool, optional): If True, display download progress. Default is True.

    Returns:
    - file_path (str): The local path where the downloaded file is saved.
    """
    # If file name is not provided, extract it from the URL
    if file_name is None:
        file_name = url[url.rfind('/') + 1:]

    # Determine the local cache directory and file path
    cache_dir = os.path.join(os.path.expanduser('~'), '.artnex')
    file_path = os.path.join(cache_dir, file_name)

    # If the file already exists and overwrite is False, return its path immediately
    if not overwrite and os.path.exists(file_path):
        logger.info("File already exists. Skipping download.")
        return file_path

    # Print download information
    logger.info("Downloading: " + file_name)

    try:
        # Use requests library for more robust error handling 
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Raise HTTPError for bad responses

            # Save the file
            with open(file_path, 'wb') as file:
                total_size = int(response.headers.get('content-length', 0))
                if show_progress:
                    show_progress_bar(response.iter_content(chunk_size=1024), total_size, file_path)

    except (requests.RequestException, Exception) as e:
        logger.error(f"Error downloading file: {e}")
        raise

    # Display success message after successful download
    logger.info("Download complete!")
    return file_path

def show_progress_bar(response_iter_content, total_size, file_path):
    """
    Displays a download progress bar.

    Parameters:
    - response_iter_content (iterable): The iterable content of the download response.
    - total_size (int): The total size of the file to be downloaded.
    - file_path (str): The local path where the downloaded file is being saved.
    """
    downloaded = 0
    with open(file_path, 'wb') as file:
        for chunk in response_iter_content:
            if chunk:
                file.write(chunk)
                downloaded += len(chunk)
                finish_rate = min(downloaded / total_size, 1.0)
                display_progress_bar(finish_rate)

def display_progress_bar(finish_rate):
    """
    Displays a progress bar.

    Parameters:
    - finish_rate (float): The completion rate of the download (0.0 to 1.0).
    """
    progress = '{:.2%}'.format(finish_rate)
    bar_length = 30
    bar = '#' * int(bar_length * finish_rate)
    spaces = '.' * (bar_length - len(bar))
    progress_bar = '[{}{}] {}'.format(bar, spaces, progress)
    print('\r' + progress_bar, end='', flush=True)

def download_file_v2(url, file_name=None):
    """
    Downloads a file from the specified URL.

    Parameters:
    - url (str): The URL of the file to be downloaded.
    - file_name (str, optional): The name to be used for the downloaded file. If not provided,
      it uses the part of the URL after the last slash as the file name.

    Returns:
    - file_path (str): The local path where the downloaded file is saved.
    """
    # If file name is not provided, extract it from the URL
    if file_name is None:
        file_name = url[url.rfind('/') + 1:]

    # Determine the local cache directory and file path
    cache_dir = os.path.join(os.path.expanduser('~'), '.artnex')
    file_path = os.path.join(cache_dir, file_name)

    # If the cache directory doesn't exist, create it
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    # If the file already exists, return its path immediately
    if os.path.exists(file_path):
        return file_path

    # Print download information
    print("Downloading: " + file_name)

    try:
        # Use urllib.request.urlretrieve to download the file and display the download progress
        urllib.request.urlretrieve(url, file_path, display_progress)
    except (Exception, KeyboardInterrupt):
        # If an exception or keyboard interrupt occurs during the download, remove the partially downloaded file
        if os.path.exists(file_path):
            os.remove(file_path)
        raise

    # Display success message after successful download
    print(" Download complete!")
    return file_path

def display_progress(block_num, block_size, total_size, start_time):
    """
    Display a progress bar indicating the download progress along with additional information.
    
    Parameters:
    - block_num (int): The current number of data blocks downloaded.
    - block_size (int): The size of each data block in bytes.
    - total_size (int): The total size of the file being downloaded in bytes.
    - start_time (float): The timestamp indicating when the download started.
    """
    # Calculate the amount of downloaded data
    downloaded = block_num * block_size
    
    # Ensure finish rate doesn't exceed 100%
    finish_rate = min(downloaded / total_size, 1.0)
    
    # Format the progress percentage and create the progress bar
    progress = '{:.2%}'.format(finish_rate)
    bar_length = 30
    bar = '#' * int(bar_length * finish_rate)
    spaces = '.' * (bar_length - len(bar))
    progress_bar = '[{}{}] {}'.format(bar, spaces, progress)

    # Calculate elapsed time since the start of the download
    elapsed_time = time.time() - start_time

    # print('Downloaded:', downloaded)
    # print('Finish Rate:', finish_rate)
    # print('Elapsed Time:', elapsed_time)
    
    # Display download speed
    if elapsed_time > 0:
        download_speed = downloaded / (1024 * elapsed_time)  # Download speed in KB/s
        print('\r{}  - Download speed: {:.2f} KB/s'.format(progress_bar, download_speed), end='', flush=True)

        # Display estimated remaining time
        if finish_rate > 0:
            remaining_time = (1 / finish_rate - 1) * elapsed_time
            remaining_minutes = int(remaining_time // 60)
            remaining_seconds = int(remaining_time % 60)
            print('  - Remaining time: {} min {} sec'.format(remaining_minutes, remaining_seconds), end='', flush=True)

    # Display completion message when the download is finished
    if finish_rate == 1:
        print('\nDownload completed!', end='', flush=True)

class ONNXExporter:
    def __init__(self, model, onnx_path):
        """
        Construct an ONNX Exporter.

        Args:
            model: The model to be exported.
            onnx_path (str): Path to save the ONNX model.
        """
        self.model = model
        self.onnx_path = onnx_path

    def export(self, input_shape):
        """
        Export the model to ONNX format.

        Args:
            input_shape (tuple): Input shape of the model.

        Returns:
            None
        """
        # Prepare dummy input with appropriate shape
        dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)

        # Export the model to ONNX format
        onnx.export(self.model, dummy_input, self.onnx_path, verbose=True, input_names=['input'],
                    output_names=['output'])

class ONNXModelLoader:
    def __init__(self, onnx_path):
        """
        Construct an ONNX Model Loader.

        Args:
            onnx_path (str): Path to the ONNX model file.
        """
        self.onnx_path = onnx_path
        self.session = ort.InferenceSession(onnx_path)

    def predict(self, input_data):
        """
        Make predictions using the ONNX model.

        Args:
            input_data: Input data for prediction.

        Returns:
            Output predictions.
        """
        return self.session.run(None, {'input': input_data})
