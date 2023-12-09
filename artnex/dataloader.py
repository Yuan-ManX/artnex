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


import math
import numpy as np
import threading
import matplotlib.pyplot as plt
from core import Variable


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, num_threads=1):
        """
        Initializes the data loader.

        Args:
        - dataset: The dataset, usually a list containing input samples and their corresponding targets.
        - batch_size: The number of samples in each batch.
        - shuffle: Whether to shuffle the dataset before each epoch, default is True.
        - num_threads: The number of threads used for loading data, default is 1.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_threads = num_threads
        # Calculate the number of batches, rounding up
        self.batch_number = math.ceil(len(dataset) / batch_size)  
        # Call the internal method for initialization and shuffling the dataset
        self._reset()  

    def _reset(self):
        """
        Internal method to reset the state and potentially shuffle the dataset before each epoch.
        """
        # Current batch iterator
        self.batch_iter = 0  
        if self.shuffle:
            # Shuffle the dataset indices
            self.batch_index = np.random.permutation(len(self.dataset))  
        else:
            # Do not shuffle the dataset indices
            self.batch_index = np.arange(len(self.dataset))  

    def __iter__(self):
        """
        Makes the DataLoader instance an iterable object.
        """
        return self
    
    def _get_batch_data(self):
        """
        Internal method to get data for the current batch.
        """
        start_idx = self.batch_iter * self.batch_size
        end_idx = (self.batch_iter + 1) * self.batch_size

        # Iterate over the indices of the current batch and get the corresponding input and target
        batch_data = [(self.dataset[self.batch_index[i]][0], self.dataset[self.batch_index[i]][1])
                      for i in range(start_idx, end_idx)]

        # Use zip to unpack the batch data into separate lists for input and target
        x, t = zip(*batch_data)

        return x, t

    def __next__(self, dynamic_batch_size=None):
        """
        The method of the iterator, returns a batch of data each time it is called.

        Args:
        - dynamic_batch_size: Dynamically specified batch size. If None, use the batch size set during initialization.
        """
        if dynamic_batch_size is not None:
            self.batch_size = dynamic_batch_size
            
        if self.batch_iter == self.batch_number:
            # When all batches are iterated, reset the state and raise StopIteration to indicate the end of iteration
            self._reset()
            raise StopIteration()

        # Use the new internal method to get data for the current batch
        x, t = self._get_batch_data()  
        # Update the batch iterator
        self.batch_iter += 1  

        x = Variable(x) 
        t = Variable(t) 

        return x, t 
    
    def __len__(self):
        """
        Get the length of the dataset.
        """
        return len(self.dataset)
    
    def get_index(self, index):
        """
        Get a sample from the dataset at a specific position.

        Args:
        - index: Index of the sample to retrieve.
        """
        return self.dataset[index]
    
    def add_sample(self, sample):
        """
        Add a sample to the dataset.

        Args:
        - sample: The sample to add, should be a tuple containing the input sample and its corresponding target.
        """
        self.dataset.append(sample)
        self._reset()  # Since the dataset has changed, it needs to be reinitialized and shuffled

    def remove_sample(self, index):
        """
        Remove a sample from the dataset at the specified index.

        Args:
        - index: Index of the sample to remove.
        """
        if 0 <= index < len(self.dataset):
            del self.dataset[index]
            self._reset()  # Since the dataset has changed, it needs to be reinitialized and shuffled
        else:
            raise ValueError("Index out of range.")
        
    def clear_dataset(self):
        """
        Clear the entire dataset.
        """
        self.dataset = []
        self._reset()  # Since the dataset has changed, it needs to be reinitialized and shuffled
    
    def compute_statistics(self, batch_size=None):
        """
        Compute the mean and standard deviation of the dataset.

        Args:
        - batch_size: Specify the batch size for computing statistics, if None, use the DataLoader's batch size.

        Returns:
        - mean: The mean of the dataset.
        - std: The standard deviation of the dataset.
        """
        if batch_size is None:
            batch_size = self.batch_size

        total_samples = len(self.dataset)
        total_batches = math.ceil(total_samples / batch_size)
        all_data = []

        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_samples)
            batch_data = [self.dataset[j] for j in range(start_idx, end_idx)]
            all_data.extend(batch_data)

        all_data = np.concatenate([sample[0] for sample in all_data], axis=0)
        mean = np.mean(all_data, axis=0)
        std = np.std(all_data, axis=0)

        return mean, std
    
    def visualize_samples(self, num_samples=5, cmap='gray', input_title='Input', target_title='Target'):
        """
        Visualizes samples from the dataset.

        Args:
        - num_samples: Number of samples to visualize, default is 5.
        - cmap: Colormap used in Matplotlib for displaying images, default is 'gray'.
        - input_title: Title for the input image, default is 'Input'.
        - target_title: Title for the target image, default is 'Target'.
        """
        if num_samples > len(self.dataset):
            num_samples = len(self.dataset)

        # Randomly select num_samples samples
        sample_indices = np.random.choice(len(self.dataset), num_samples, replace=False)

        # Visualize each sample
        for i, idx in enumerate(sample_indices):
            sample = self.dataset[idx]
            input_data, target_data = sample[0], sample[1]

            # Visualize input data
            plt.subplot(2, num_samples, i + 1)
            plt.imshow(input_data, cmap=cmap)
            plt.axis('off')
            plt.title(f'Sample {i + 1}\n{input_title}')

            # Visualize target data (if available)
            if target_data is not None:
                plt.subplot(2, num_samples, num_samples + i + 1)
                plt.imshow(target_data, cmap=cmap)
                plt.axis('off')
                plt.title(target_title)

        plt.show()
    
    # def _load_data_thread(self, start_idx, end_idx, result):
    #     """
    #     Internal method to load data in a thread.
    #     """
    #     batch_data = [(self.dataset[self.batch_index[i]][0], self.dataset[self.batch_index[i]][1])
    #                   for i in range(start_idx, end_idx)]
    #     result.extend(batch_data)

    # def _load_data_multi_thread(self):
    #     """
    #     Internal method to load data using multiple threads.
    #     """
    #     threads = []
    #     results = [[] for _ in range(self.num_threads)]
    #     step = len(self.dataset) // self.num_threads

    #     for i in range(self.num_threads):
    #         start_idx = i * step
    #         end_idx = (i + 1) * step if i < self.num_threads - 1 else len(self.dataset)
    #         thread = threading.Thread(target=self._load_data_thread, args=(start_idx, end_idx, results[i]))
    #         threads.append(thread)
    #         thread.start()

    #     for thread in threads:
    #         thread.join()

    #     return [item for sublist in results for item in sublist]

    # def _get_batch_data_multi_thread(self):
    #     """
    #     Internal method to get data for the current batch using multiple threads.
    #     """
    #     start_idx = self.batch_iter * self.batch_size
    #     end_idx = (self.batch_iter + 1) * self.batch_size

    #     batch_data = self._load_data_multi_thread()

    #     # Use zip to unpack the batch data into separate lists for input and target
    #     x, t = zip(*batch_data[start_idx:end_idx])

    #     return x, t

    # def __del__(self):
    #     """
    #     When the instance is destroyed, wait for all threads to finish.
    #     """
    #     threading.Thread.join(self)
