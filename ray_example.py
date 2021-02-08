import argparse
import logging
import os
from utils.ray_test import compute_reciprocals
from execution_time import ExecutionTime
import ray
import ray.util
import numpy as np
np.random.seed(0)
import time


e = ExecutionTime()


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', default=False, action="store_true", help="Run computation on CPU")
    parser.add_argument('--ray', default=False, action="store_true", help="Run computation on CPU")
    args = parser.parse_args()
    return args


@ray.remote(num_cpus=4)
class Counter(object):
    def __init__(self):
        self.n = 0

    def increment(self):
        self.n += 2

    def read(self):
        return self.n


@ray.remote
def compute_reciprocals_ray(values):
    return compute_reciprocals(values)


@e.timeit
def main():
    # set TF_XLA_FLAGS env variable to increase GPU utilization
    # os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    # if args.cpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("This runs on the VM")


if __name__ == '__main__':
    args = parse_args()
    repeater = 10

    if args.ray:
        if ray.is_initialized() == False:
            # ray.init(local_mode=True)  # for debug purposes
            ray.init(address='192.168.1.11:6379', _redis_password='5241590000000000', logging_level=logging.DEBUG)
            # ray.util.connect("192.168.1.11:10001")  # replace with the appropriate host and port
        # call main function as ray.remote
        # future = main.remote()
        # print("This runs locally")
        # ray.get(future)
        # counters = [Counter.remote() for i in range(4)]
        # [c.increment.remote() for c in counters]
        # futures = [c.read.remote() for c in counters]
        big_array = np.random.randint(1, 100, size=2000000)
        start = time.time()
        results = [compute_reciprocals_ray.remote(big_array) for i in range(repeater)]
        print(len(ray.get(results)))
        print("duration =", time.time() - start)
    else:
        big_array = np.random.randint(1, 100, size=2000000)
        start = time.time()
        results = [compute_reciprocals(big_array) for _ in range(repeater)]
        print(len(results))
        # print(e.logtime_data)
        print("duration =", time.time() - start)
