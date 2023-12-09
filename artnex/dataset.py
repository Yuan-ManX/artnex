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
import numpy as np 
import gzip
import pickle
import requests
from urllib.request import urlretrieve
from collections import Counter
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
from nn import *


class Dataset:
    def __init__(self, transformer=None, target_transformer=None):
        """
        Initializes the Dataset class.

        Args:
            transformer (callable, optional): A function to transform input data.
            target_transformer (callable, optional): A function to transform target data.
        """
        self.transformer = transformer  # Used for transforming input data
        self.target_transformer = target_transformer  # Used for transforming target data

    def __len__(self):
        """
        Returns the length of the dataset. Must be implemented by specific dataset classes.
        """
        return self._len()

    def __getitem__(self, index):
        """
        Retrieves data based on the given index and applies the respective transformers.

        Args:
            index (int): Index to retrieve data.

        Returns:
            tuple: Transformed input data and target data.
        """
        x, t = self._getitem(index)  # Calls the method implemented in subclasses to get raw data and target

        # Applies input data transformer if it exists
        if x is not None and self.transformer is not None:
            x = self.transformer(x)

        # Applies target data transformer if it exists
        if t is not None and self.target_transformer is not None:
            t = self.target_transformer(t)

        # Returns the transformed input data and target data
        return x, t
    
    def select_subset(self, indices):
        """
        Selects a subset of the dataset based on the given indices.

        Args:
            indices (list): List of indices to select the subset.

        Returns:
            Dataset: A new dataset instance containing the selected subset.
        """
        subset = self.__class__(transformer=self.transformer, target_transformer=self.target_transformer)
        subset._select_subset(indices)
        return subset

    # The following two methods must be implemented by specific dataset classes

    def _getitem(self, index):
        """
        Retrieves raw data and target data based on the given index. Must be implemented by subclasses.

        Args:
            index (int): Index to retrieve data.

        Returns:
            tuple: Raw input data and raw target data.
        """
        raise NotImplementedError()

    def _len(self):
        """
        Returns the length of the dataset. Must be implemented by subclasses.

        Returns:
            int: Length of the dataset.
        """
        raise NotImplementedError()
    
    def _select_subset(self, indices):
        """
        Selects a subset of the dataset based on the given indices. Must be implemented by subclasses.

        Args:
            indices (list): List of indices to select the subset.
        """
        raise NotImplementedError()

class RandomDataset:
    def __init__(self, num_samples=1000, input_dim=2, output_dim=1, transformer=None, target_transformer=None):
        """
        Initializes the RandomDataset class.

        Args:
            num_samples (int): Number of samples in the dataset.
            input_dim (int): Dimensionality of input data.
            output_dim (int): Dimensionality of target data.
            transformer (callable, optional): Function to transform input data.
            target_transformer (callable, optional): Function to transform target data.
        """
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.transformer = transformer
        self.target_transformer = target_transformer

        # Generate random input data and target data
        self.x = np.random.rand(num_samples, input_dim)
        self.t = np.random.rand(num_samples, output_dim)

    def _len(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return self.num_samples

    def _getitem(self, index):
        """
        Retrieves data based on the given index.

        Args:
            index (int): Index used to retrieve data.

        Returns:
            tuple: Tuple containing input data and target data.
        """
        # Get input data and target data at the specified index
        x_sample = self.x[index]
        t_sample = self.t[index]

        # Apply transformers if provided
        if self.transformer:
            x_sample = self.transformer(x_sample)
        if self.target_transformer:
            t_sample = self.target_transformer(t_sample)

        return x_sample, t_sample

class MNISTDataset:
    def __init__(self, train=True, transform=None, target_transform=None):
        """
        Initializes the MNIST dataset.

        Parameters:
        - train (bool): If True, loads the training set; otherwise, loads the test set.
        - transform (callable, optional): A function/transform that takes input data and returns a transformed version.
        - target_transform (callable, optional): A function/transform that takes input labels and returns a transformed version.
        """
        self.transform = transform
        self.target_transform = target_transform

        # Define URLs and file names for training and test sets
        base_url = 'http://yann.lecun.com/exdb/mnist/'
        train_files = {'data': 'train-images-idx3-ubyte.gz', 'label': 'train-labels-idx1-ubyte.gz'}
        test_files = {'data': 't10k-images-idx3-ubyte.gz', 'label': 't10k-labels-idx1-ubyte.gz'}

        # Choose files based on the 'train' parameter
        files = train_files if train else test_files

        # Get file paths for data and labels
        self.data_path = self._get_file(base_url + files['data'])
        self.label_path = self._get_file(base_url + files['label'])

        # Load data and labels into the dataset
        self.data = self._load_data(self.data_path)
        self.labels = self._load_labels(self.label_path)

    def _get_file(self, url, file_name=None):
        """
        Downloads a file from the given URL and returns the local file path.

        Parameters:
        - url (str): URL of the file to download.
        - file_name (str, optional): Name of the file to save. If not provided, extracts from the URL.

        Returns:
        - str: Local file path.
        """
        if file_name is None:
            file_name = url.split('/')[-1]

        cache_dir = os.path.join(os.path.expanduser('~'), '.new-mnist')
        file_path = os.path.join(cache_dir, file_name)

        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

        if not os.path.exists(file_path):
            print("Downloading:", file_name)
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            with open(file_path, 'wb') as file, tqdm(total=total_size, unit='B', unit_scale=True) as bar:
                for data in response.iter_content(block_size):
                    bar.update(len(data))
                    file.write(data)
            print("Download complete!")

        return file_path

    def _load_data(self, file_path):
        """
        Load image data from a gzip-compressed file.

        Parameters:
        - file_path (str): Path to the gzip-compressed file containing image data.

        Returns:
        - numpy.ndarray: 4D array containing image data.
        """
        with gzip.open(file_path, 'rb') as file:
            image_data = np.frombuffer(file.read(), np.uint8, offset=16)
        image_data = image_data.reshape(-1, 1, 28, 28)
        return image_data

    def _load_labels(self, file_path):
        """
        Load label data from a gzip-compressed file.

        Parameters:
        - file_path (str): Path to the gzip-compressed file containing label data.

        Returns:
        - numpy.ndarray: Array containing label data.
        """
        with gzip.open(file_path, 'rb') as file:
            labels = np.frombuffer(file.read(), np.uint8, offset=8)
        return labels

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
        - int: Number of images in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns a tuple containing the image data and corresponding label for the given index.

        Parameters:
        - index (int): Index of the desired sample.

        Returns:
        - tuple: A tuple containing the image data and label.
        """
        return self.data[index], self.labels[index]

class CIFAR10Dataset:
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        """
        Initializes the CIFAR-10 dataset.

        Parameters:
        - train (bool): If True, loads the training set; otherwise, loads the test set.
        - transform (callable, optional): A function/transform that takes input data and returns a transformed version.
        - target_transform (callable, optional): A function/transform that takes input labels and returns a transformed version.
        - download (bool): If True, downloads the dataset.

        Additional Features:
        - Normalization: Data normalization can be applied during initialization.
        - Data Visualization: You can visualize a random sample from the dataset.
        """
        self.transform = transform
        self.target_transform = target_transform

        base_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        file_name = 'cifar-10-python.tar.gz'
        extract_folder = 'cifar-10-batches-py'

        # Download and extract CIFAR-10 dataset
        if download:
            self.data_path = self._get_file(base_url, file_name, extract_folder)
        
        # Load data and labels into the dataset
        self.data, self.labels = self._load_data()

        # Additional Features
        self.normalize_data()  # Normalize data during initialization
        self.visualize_sample()  # Visualize a random sample

    def _get_file(self, url, file_name, extract_folder):
        """
        Downloads a file from the given URL and returns the local file path.

        Parameters:
        - url (str): URL of the file to download.
        - file_name (str, optional): Name of the file to save. If not provided, extracts from the URL.
        - extract_folder (str): Folder to extract the contents of the downloaded file.

        Returns:
        - str: Local file path.
        """
        if file_name is None:
            file_name = url.split('/')[-1]

        cache_dir = os.path.join(os.path.expanduser('~'), '.new-cifar10')
        file_path = os.path.join(cache_dir, file_name)

        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

        if not os.path.exists(file_path):
            print("Downloading:", file_name)
            response = urlretrieve(url, file_path)
            print("Download complete!")

        return file_path

    def _load_data(self):
        """
        Load data and labels from CIFAR-10 dataset.

        Returns:
        - tuple: Tuple containing data and labels.
        """
        with open(self.data_path, 'rb') as file:
            dataset = pickle.load(file, encoding='bytes')
        
        data = dataset[b'data'].reshape(-1, 3, 32, 32) / 255.0  # Normalize pixel values to [0, 1]
        labels = np.array(dataset[b'labels'])

        return data, labels

    def normalize_data(self):
        """
        Normalizes the image data to have zero mean and unit variance.
        """
        self.data = (self.data - np.mean(self.data)) / np.std(self.data)

    def visualize_sample(self):
        """
        Visualizes a random sample from the dataset.
        """
        index = np.random.randint(len(self))
        sample_data, sample_label = self.__getitem__(index)

        # Display the image
        plt.imshow(np.transpose(sample_data, (1, 2, 0)))
        plt.title(f"Sample Label: {sample_label}")
        plt.show()

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
        - int: Number of images in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns a tuple containing the image data and corresponding label for the given index.

        Parameters:
        - index (int): Index of the desired sample.

        Returns:
        - tuple: A tuple containing the image data and label.
        """
        return self.data[index], self.labels[index]

class FashionMNISTDataset:
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        """
        Initializes the Fashion-MNIST dataset.

        Parameters:
        - train (bool): If True, loads the training set; otherwise, loads the test set.
        - transform (callable, optional): A function/transform that takes input data and returns a transformed version.
        - target_transform (callable, optional): A function/transform that takes input labels and returns a transformed version.
        - download (bool): If True, downloads the dataset.

        Additional Features:
        - Normalization: Data normalization can be applied during initialization.
        - Data Visualization: You can visualize a random sample from the dataset.
        """
        self.transform = transform
        self.target_transform = target_transform

        base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
        train_files = {'data': 'train-images-idx3-ubyte.gz', 'label': 'train-labels-idx1-ubyte.gz'}
        test_files = {'data': 't10k-images-idx3-ubyte.gz', 'label': 't10k-labels-idx1-ubyte.gz'}

        # Choose files based on the 'train' parameter
        files = train_files if train else test_files

        # Get file paths for data and labels
        self.data_path = self._get_file(base_url + files['data'])
        self.label_path = self._get_file(base_url + files['label'])

        # Load data and labels into the dataset
        self.data = self._load_data(self.data_path)
        self.labels = self._load_labels(self.label_path)

        # Additional Features
        self.normalize_data()  # Normalize data during initialization
        self.visualize_sample()  # Visualize a random sample

    def _get_file(self, url, file_name=None):
        """
        Downloads a file from the given URL and returns the local file path.

        Parameters:
        - url (str): URL of the file to download.
        - file_name (str, optional): Name of the file to save. If not provided, extracts from the URL.

        Returns:
        - str: Local file path.
        """
        if file_name is None:
            file_name = url.split('/')[-1]

        cache_dir = os.path.join(os.path.expanduser('~'), '.new-fashion-mnist')
        file_path = os.path.join(cache_dir, file_name)

        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

        if not os.path.exists(file_path):
            print("Downloading:", file_name)
            urlretrieve(url, file_path)
            print("Download complete!")

        return file_path

    def _load_data(self, file_path):
        """
        Load image data from a gzip-compressed file.

        Parameters:
        - file_path (str): Path to the gzip-compressed file containing image data.

        Returns:
        - numpy.ndarray: 3D array containing image data.
        """
        with gzip.open(file_path, 'rb') as file:
            image_data = np.frombuffer(file.read(), np.uint8, offset=16)
        image_data = image_data.reshape(-1, 28, 28)
        return image_data

    def _load_labels(self, file_path):
        """
        Load label data from a gzip-compressed file.

        Parameters:
        - file_path (str): Path to the gzip-compressed file containing label data.

        Returns:
        - numpy.ndarray: Array containing label data.
        """
        with gzip.open(file_path, 'rb') as file:
            labels = np.frombuffer(file.read(), np.uint8, offset=8)
        return labels

    def normalize_data(self):
        """
        Normalizes the image data to have zero mean and unit variance.
        """
        self.data = (self.data - np.mean(self.data)) / np.std(self.data)

    def visualize_sample(self):
        """
        Visualizes a random sample from the dataset.
        """
        index = np.random.randint(len(self))
        sample_data, sample_label = self.__getitem__(index)

        # Display the image
        plt.imshow(sample_data, cmap='gray')
        plt.title(f"Sample Label: {sample_label}")
        plt.show()

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
        - int: Number of images in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns a tuple containing the input data and corresponding label for the given index.

        Parameters:
        - index (int): Index of the desired sample.

        Returns:
        - tuple: A tuple containing the input data and label.
        """
        return self.data[index], self.labels[index]

class AudioMNISTDataset:
    def __init__(self, data_dir, transform=None, target_transform=None):
        """
        Initializes the AudioMNIST dataset.

        Parameters:
        - data_dir (str): Directory containing AudioMNIST dataset files.
        - transform (callable, optional): A function/transform that takes input data and returns a transformed version.
        - target_transform (callable, optional): A function/transform that takes input labels and returns a transformed version.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

        # Load data and labels into the dataset
        self.data, self.labels = self._load_data()

    def _load_data(self):
        """
        Load audio data and labels from AudioMNIST dataset.

        Returns:
        - tuple: Tuple containing data and labels.
        """
        data = []
        labels = []

        for filename in os.listdir(self.data_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(self.data_dir, filename)
                label = int(filename.split('_')[0])  # Extract label from filename
                audio_data, sr = librosa.load(file_path, sr=None)

                # Additional feature: Visualize a random sample
                if np.random.rand() < 0.1:
                    self.visualize_sample(audio_data, label)

                data.append(audio_data)
                labels.append(label)

        return data, labels

    def visualize_sample(self, audio_data, label):
        """
        Visualizes a random sample from the dataset.

        Parameters:
        - audio_data (numpy.ndarray): Audio waveform data.
        - label (int): Label associated with the audio sample.
        """
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(audio_data, sr=22050)
        plt.title(f"Label: {label}")
        plt.show()

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
        - int: Number of audio samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns a tuple containing the input data and corresponding label for the given index.

        Parameters:
        - index (int): Index of the desired sample.

        Returns:
        - tuple: A tuple containing the audio data and label.
        """
        return self.data[index], self.labels[index]

class CIFAR100Dataset:
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        """
        Initializes the CIFAR-100 dataset.

        Parameters:
        - train (bool): If True, loads the training set; otherwise, loads the test set.
        - transform (callable, optional): A function/transform that takes input data and returns a transformed version.
        - target_transform (callable, optional): A function/transform that takes input labels and returns a transformed version.
        - download (bool): If True, downloads the dataset.

        Additional Features:
        - Normalization: Data normalization can be applied during initialization.
        - Data Visualization: You can visualize a random sample from the dataset.
        - Class Balancing: Balances the class distribution in the training set.
        """
        self.transform = transform
        self.target_transform = target_transform

        base_url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        file_name = 'cifar-100-python.tar.gz'
        extract_folder = 'cifar-100-python'

        # Download and extract CIFAR-100 dataset
        if download:
            self.data_path = self._get_file(base_url, file_name, extract_folder)
        
        # Load data and labels into the dataset
        self.data, self.fine_labels, self.coarse_labels = self._load_data()

        # Additional Features
        if train:
            self.balance_classes()  # Balance class distribution in the training set
        self.normalize_data()  # Normalize data during initialization
        self.visualize_sample()  # Visualize a random sample

    def _get_file(self, url, file_name, extract_folder):
        """
        Downloads a file from the given URL and returns the local file path.

        Parameters:
        - url (str): URL of the file to download.
        - file_name (str): Name of the file to save.
        - extract_folder (str): Folder to extract the contents of the downloaded file.

        Returns:
        - str: Local file path.
        """
        if not os.path.exists(extract_folder):
            os.mkdir(extract_folder)

        file_path = os.path.join(extract_folder, file_name)

        if not os.path.exists(file_path):
            print("Downloading:", file_name)
            urlretrieve(url, file_path)
            print("Download complete!")

        return file_path

    def _load_data(self):
        """
        Load data and labels from CIFAR-100 dataset.

        Returns:
        - tuple: Tuple containing data, fine labels, and coarse labels.
        """
        with open(os.path.join('cifar-100-python', 'train' if self.train else 'test'), 'rb') as file:
            dataset = pickle.load(file, encoding='bytes')

        data = np.array(dataset[b'data'], dtype=np.float32) / 255.0
        fine_labels = np.array(dataset[b'fine_labels'])
        coarse_labels = np.array(dataset[b'coarse_labels'])

        return data, fine_labels, coarse_labels

    def balance_classes(self):
        """
        Balances the class distribution in the training set by oversampling minority classes.
        """
        class_counter = Counter(self.coarse_labels)
        max_class_count = max(class_counter.values())

        balanced_data = []
        balanced_fine_labels = []
        balanced_coarse_labels = []

        for class_label, count in class_counter.items():
            oversample_factor = max_class_count // count
            indices = np.where(np.array(self.coarse_labels) == class_label)[0]
            oversampled_indices = np.repeat(indices, oversample_factor)
            
            balanced_data.extend([self.data[i] for i in oversampled_indices])
            balanced_fine_labels.extend([self.fine_labels[i] for i in oversampled_indices])
            balanced_coarse_labels.extend([self.coarse_labels[i] for i in oversampled_indices])

        self.data = balanced_data
        self.fine_labels = balanced_fine_labels
        self.coarse_labels = balanced_coarse_labels

    def normalize_data(self):
        """
        Normalizes the image data to have zero mean and unit variance.
        """
        self.data = (self.data - np.mean(self.data)) / np.std(self.data)

    def visualize_sample(self):
        """
        Visualizes a random sample from the dataset.
        """
        index = np.random.randint(len(self))
        sample_data, fine_label, coarse_label = self.__getitem__(index)

        # Display the image
        plt.imshow(np.transpose(sample_data, (1, 2, 0)))
        plt.title(f"Fine Label: {fine_label}, Coarse Label: {coarse_label}")
        plt.show()

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
        - int: Number of images in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns a tuple containing the image data, fine label, and corresponding coarse label for the given index.

        Parameters:
        - index (int): Index of the desired sample.

        Returns:
        - tuple: A tuple containing the image data, fine label, and coarse label.
        """
        return self.data[index], self.fine_labels[index], self.coarse_labels[index]

