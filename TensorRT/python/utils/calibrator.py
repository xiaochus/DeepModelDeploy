import os
import cv2
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, training_data, cache_file, image_size, mean=[0, 0, 0], std = [1, 1, 1], batch_size=2):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.data = self._load_data(training_data, image_size, mean, std)
        self.batch_size = batch_size
        self.current_index = 0

        print("Calibration data shape: ", self.data.shape)

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    def _load_data(self, training_data, image_size, mean, std):
        data = []
        mean = np.array(mean)
        std = np.array(std)

        with open(training_data, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            if os.path.exists(line):
                img = cv2.imread(line)

                img = cv2.resize(img, image_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255.0
                img = img - mean
                img = img / std
                img = np.transpose(img, (2, 0, 1))
                img = np.expand_dims(img, axis=0)

                data.append(img)
            else:
                print("file {} not exist, skip it!".format(line))
                continue

        return np.concatenate(data, axis=0)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            print("read cache file: {}".format(self.cache_file))

            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
