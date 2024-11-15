import gin
from tensorflow.data.experimental import AUTOTUNE
import tensorflow as tf
import logging
import keras
from preprocessing import preprocess, augment
import tensorflow_datasets as tfds

batch_size = 32

@gin.configurable
def load(name, data_dir, batch_size=batch_size, caching=True):
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")

        # Load dataset from directory structure, where each subdirectory represents a class,return an object, which is an iterable tuples (image, label)
        full_ds = keras.preprocessing.image_dataset_from_directory(
            data_dir,
            batch_size=batch_size,
            label_mode='int'  # use 'int' for integer label , for classification
        )

        # Calculate the number of examples for shuffle buffer size
        num_examples = sum(1 for _ in full_ds.unbatch())
        
        for images, _ in full_ds.take(1):
            image_shape = images.shape[1:]
            break

        class_names = full_ds.class_names
        num_classes = len(class_names)

        # Define df_info
        ds_info = {
            "num_examples" : num_examples,
            "features" : {
                "image" : {"shape" : image_shape , "dtype": tf.float32},
                "label" : {"num_classes": num_classes, "dtype": tf.int64}
            }
        }

        # Split into training and validation sets
        total_samples = sum(1 for _ in full_ds.unbatch())
        val_size = int(0.2 * total_samples)
        train_size = int(total_samples) - val_size
        ds_train = full_ds.take(train_size)
        ds_val = full_ds.skip(train_size)

        # Prepare and return the training and validation datasets
        return prepare(ds_train, ds_val, ds_info=ds_info, batch_size=batch_size, caching=caching)

    elif name == "eyepacs":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'diabetic_retinopathy_detection/btgraham-300',
            split=['train', 'validation', 'test'],
            shuffle_files=True,
            with_info=True,
            data_dir=data_dir
        )



        def _preprocess(img_label_dict):
            return img_label_dict['image'], img_label_dict['label']

        ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "mnist":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train[:90%]', 'train[90%:]', 'test'],
            shuffle_files=True,
            with_info=True,
            data_dir=data_dir
        )

        return prepare(ds_train, ds_val, ds_test, ds_info)

    else:
        raise ValueError

@gin.configurable
def prepare(ds_train, ds_val, ds_test=None, ds_info=None, batch_size = batch_size, caching=True):
    """Prepare datasets with preprocessing, augmentation, batching, caching, and prefetching"""
    # Prepare training dataset
    ds_train = ds_train.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()
    if ds_info:
        shuffle_buffer_size = ds_info.get("num_examples", 1000) // 10  # Default to 1000 if ds_info not provided
        ds_train = ds_train.shuffle(shuffle_buffer_size)
    else:
        ds_train = ds_train.shuffle(1000)  # Fallback shuffle size

    ds_train = ds_train.repeat().prefetch(tf.data.AUTOTUNE)

    # Prepare validation dataset (no augmentation)
    ds_val = ds_val.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    # Prepare test dataset if available (no augmentation)
    if ds_test is not None:
        ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if caching:
            ds_test = ds_test.cache()
        ds_test = ds_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info



