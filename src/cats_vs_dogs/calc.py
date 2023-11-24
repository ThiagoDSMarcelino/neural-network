
from util.image import count_images

def calc_steps_per_epoch(path: str, batch_size: int, test_size: float) -> int:
    n_images = int(count_images(path) * test_size)

    steps_per_epoch = n_images // batch_size

    return steps_per_epoch

def calc_validation_steps(steps_per_epoch: int) -> int:
    return int(steps_per_epoch / 10)