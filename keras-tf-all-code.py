import datetime
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data_using_keras(folders):
    """
    Load the images in batches using Keras.
    Shuffle images (for training set only) using keras.

    Returns:
    Data Generator to be used while training the model.

    Note: Keras might need 'pillow' library to be installed. Use-
    # pip install pillow
    """
    image_generator = {}
    data_generator = {}
    for x in folders:
        image_generator[x] = ImageDataGenerator(rescale=1./255)

        shuffle_images = True if x == 'train' else False

        data_generator[x] = image_generator[x].flow_from_directory(
            batch_size=batch_size,
            directory=os.path.join(dir_path, x),
            shuffle=shuffle_images,
            target_size=(img_dims[0], img_dims[1]),
            class_mode='categorical')

    return data_generator


def load_data_using_tfdata(folders):
    """
    Load the images in batches using Tensorflow (tfdata).
    Cache can be used to speed up the process.
    Faster method in comparison to image loading using Keras.

    Returns:
    Data Generator to be used while training the model.
    """
    def parse_image(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        class_names = np.array(os.listdir(dir_path + '/train'))
        # The second to last is the class-directory
        label = parts[-2] == class_names
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        img = tf.image.resize(img, [img_dims[0], img_dims[1]])
        return img, label

    def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
        # If a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets
        # that don't fit in memory.
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        # Repeat forever
        ds = ds.repeat()
        ds = ds.batch(batch_size)
        # `prefetch` lets the dataset fetch batches in the background
        # while the model is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    data_generator = {}
    for x in folders:
        dir_extend = dir_path + '/' + x
        list_ds = tf.data.Dataset.list_files(str(dir_extend+'/*'))  # dir_extend+'/*/*'
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        # Set `num_parallel_calls` so that multiple images are
        # processed in parallel
        labeled_ds = list_ds.map(
            parse_image, num_parallel_calls=AUTOTUNE)
        # cache = True, False, './file_name'
        # If the dataset doesn't fit in memory use a cache file,
        # eg. cache='./data.tfcache'
        data_generator[x] = prepare_for_training(
            labeled_ds,
            cache='cocodata.tfcache'
            # cache=True
        )

    return data_generator


def timeit(ds, steps=1000):
        """
        Check performance/speed for loading images using Keras or tfdata.
        """
        start = time.time()
        it = iter(ds)
        for i in range(steps):
            next(it)
            print('   >> ', i, '/1000', end='\r')
        duration = time.time()-start
        print(f'''{steps} batches: '''
                f'''{datetime.timedelta(seconds=int(duration))}''')
        print(f'{round(batch_size*steps/duration)} Images/s')


def train_model(data_generator):
    """
    Create and train model to perform Transfer learning using pretrained models.
    Base layers of pretrained models are freezed.
    Stack the classification layers on top of the pretrained model.
    """
    img_shape = (img_dims[0], img_dims[1], 3)

    base_model = tf.keras.applications.MobileNetV2(
            input_shape=img_shape, include_top=False, weights='imagenet')

    # Freeze the base layers of pretrained model
    base_model.trainable = False

    model = tf.keras.Sequential([base_model,
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(
                                        256, activation='relu'),
                                    tf.keras.layers.Dense(num_classes)])

    # Define parameters for model compilation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    since = time.time()

    history = model.fit(
        data_generator['train'],
        steps_per_epoch=num_images_train // batch_size,
        epochs=epochs,
        )

    time_elapsed = time.time()-since
    print(f'''\nTraining time: '''
                f'''{datetime.timedelta(seconds=int(time_elapsed))}''')


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # set TF_XLA_FLAGS env variable to increase GPU utilization
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    # Need to change this w.r.t data
    # dir_path = './data/dog_vs_cat/dataset'
    dir_path = './data/train2017'  # http://images.cocodataset.org/zips/train2017.zip
    num_classes = 80
    # num_classes = 2  # for dog_cat dataset
    folders = ['train']
    num_images_train = 118287  # up to max of total images in directory
    load_data_using = 'tfdata'

    batch_size = 256
    img_dims = [256, 256]
    epochs = 10
    learning_rate = 0.0001

    if load_data_using == 'keras':
        data_generator = load_data_using_keras(folders)
    elif load_data_using == 'tfdata':
        data_generator = load_data_using_tfdata(folders)

    timeit(data_generator['train'])

    train_model(data_generator)
