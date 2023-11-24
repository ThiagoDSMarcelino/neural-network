from images_util import find_images

def calc_steps_per_epoch(batch_size: int, test_size: float) -> int:
    path = 'data'
    n_images = int(len(find_images(path)) * test_size)

    steps_per_epoch = int(n_images / batch_size)

    return steps_per_epoch

def calc_validation_steps(steps_per_epoch: int) -> int:
    return int(steps_per_epoch / 10)