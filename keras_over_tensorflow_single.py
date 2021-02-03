import matplotlib.pyplot as plt
import os, time
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        # use this value as reference to calculate cumulative time taken
        self.timetaken = time.process_time()
    def on_epoch_end(self,epoch,logs = {}):
        self.times.append((epoch, time.process_time() - self.timetaken))
    def on_train_end(self,logs = {}):
        plt.xlabel('Epoch')
        plt.ylabel('Total time taken until an epoch in seconds')
        plt.plot(*zip(*self.times))
        plt.show()
        from operator import itemgetter
        previous_time = 0
        for item in self.times:
            print("Epoch ", item[0], " run time is: ", item[1]-previous_time)
            previous_time = item[1]
        print("Total trained time is: ", previous_time)

def get_compiled_model_mnist():
    # Make a simple 2-layer densely-connected neural network.
    inputs = keras.Input(shape=(784,))
    x = keras.layers.Dense(256, activation="relu")(inputs)
    x = keras.layers.Dense(256, activation="relu")(x)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model_svhn()
    # return get_compiled_model_mnist()


def get_compiled_model_svhn():
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
    from tensorflow.keras.layers import Dropout, Flatten, Input, Dense

    def add_conv_block(model, num_filters):
        model.add(Conv2D(num_filters, 3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(num_filters, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.5))
        return model

    model = tf.keras.models.Sequential()
    model.add(Input(shape=(32, 32, 1)))

    model = add_conv_block(model, 32)
    model = add_conv_block(model, 64)
    model = add_conv_block(model, 128)

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(10, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


def get_dataset_from_npz():
    # Return the MNIST dataset in the form of a `tf.data.Dataset`.
    full_mnist_path = os.path.join(os.getcwd(), 'mnist.npz')
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path=full_mnist_path)

    # Preprocess the data (these are Numpy arrays)
    x_train = x_train.reshape(-1, 784).astype("float32") / 255 # x_train.size
    x_test = x_test.reshape(-1, 784).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    return [(x_train, y_train), (x_test, y_test)]

def get_dataset_fashion_mnist():
    import httplib2
    # detect presense of proxy and use env varibles if they exist
    pi = httplib2.proxy_info_from_environment()
    if pi:
        import socks
        socks.setdefaultproxy(pi.proxy_type, pi.proxy_host, pi.proxy_port)
        socks.wrapmodule(httplib2)

        # now all calls through httplib2 should use the proxy settings
    httplib2.Http()
    return tf.keras.datasets.fashion_mnist.load_data()


def get_dataset_from_mat(path_train, path_test):
    import scipy.io as sio
    # Train
    train_set = sio.loadmat(path_train)
    x_train = train_set['X']
    y_train = train_set['y']

    # Test
    test_set = sio.loadmat(path_test)
    x_test = test_set['X']
    y_test = test_set['y']

    # Preprocess the data (these are Numpy arrays)
    x_train = x_train.reshape(-1, 784).astype("float32") / 255 # x_train.size
    x_test = x_test.reshape(-1, 784).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    return [(x_train, y_train), (x_test, y_test)]

def get_dataset_from_h5(hdf_file):
    import h5py
    h5f = h5py.File(hdf_file, 'r')
    # Load the training, test and validation set
    X_train = h5f['X_train'][:]
    y_train = h5f['y_train'][:]
    X_test = h5f['X_test'][:]
    y_test = h5f['y_test'][:]

    from tensorflow.keras.utils import to_categorical
    X_train = X_train.reshape(-1, 32, 32, 1)
    X_test = X_test.reshape(-1, 32, 32, 1)
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return [(X_train, y_train), (X_test, y_test)]

def load_dataset(global_batch_size):
    # get custom dataset
    # (x_train, y_train), (x_test, y_test) = get_dataset_from_npz()
    # (x_train, y_train), (x_test, y_test) = get_dataset_fashion_mnist()
    train_mat, test_mat = 'train_32x32.mat', 'test_32x32.mat'
    # (x_train, y_train), (x_test, y_test) = get_dataset_from_mat(train_mat, test_mat)
    (X_train, y_train), (x_test, y_test) = get_dataset_from_h5('SVHN_single_grey1.h5')

    # num_val_samples = round(17.7 * x_train.size / 100.0)  # https://stackoverflow.com/a/13612921/336558
    #
    # # Reserve num_val_samples samples for validation
    # x_val = x_train[-num_val_samples:]
    # y_val = y_train[-num_val_samples:]
    # x_train = x_train[:-num_val_samples]
    # y_train = y_train[:-num_val_samples]

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2
    steps_per_epoch = X_train.shape[0]//global_batch_size

    return (
        tf.data.Dataset.from_tensor_slices((X_train, y_train)).cache().batch(global_batch_size).prefetch(buffer_size=1),
        tf.data.Dataset.from_tensor_slices((X_val, y_val)).cache().batch(global_batch_size).prefetch(buffer_size=1),
        tf.data.Dataset.from_tensor_slices((x_test, y_test)).cache().batch(global_batch_size).prefetch(buffer_size=1)
    )


def run_training(epochs, global_batch_size):
    # Create a MirroredStrategy.
    # strategy = tf.distribute.MirroredStrategy()  # single-host, multi-device synchronous training with a Keras model
    strategy = tf.distribute.OneDeviceStrategy("/gpu:0")  # place all variables and computation on a single specified device
    # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy() # NCCL vs RING  # synchronous distributed training across multiple workers, each with potentially multiple GPUs
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    # Open a strategy scope.
    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        # model = get_compiled_model_mnist()
        model = make_or_restore_model()

    timetaken = timecallback()
    callbacks = [
        # This callback saves a SavedModel every epoch
        # We include the current epoch in the folder name.
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(os.getcwd(), "ckpt-{epoch}"), save_freq="epoch"
        ),
        timetaken
    ]
    # Train the model on all available devices.
    train_dataset, val_dataset, test_dataset = load_dataset(global_batch_size)

    tf.profiler.experimental.start('logdir')
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)
    model.fit(train_dataset, epochs=epochs, callbacks=callbacks, validation_data=val_dataset, verbose=2)
    tf.profiler.experimental.stop()
    # Test the model on all available devices.
    return model.evaluate(test_dataset)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('TensorFlow version:', tf.__version__)
    print('Is using GPU?', tf.test.is_gpu_available())

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # Prepare a directory to store all the checkpoints.
    checkpoint_dir = "./ckpt"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    global_batch_size = 1024
    epochs = 5

    results = run_training(epochs=epochs, global_batch_size=global_batch_size)
    print("Successfully finished CNN learning with ", epochs, "epochs, results are:\n",  results)
