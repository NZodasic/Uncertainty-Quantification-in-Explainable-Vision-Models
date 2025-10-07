import math
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Optional, Tuple
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE
SEED = 42

class DataLoader:
    """Unified dataset loader for FACCP project"""

    def __init__(self, config):
        self.config = config

    def load_cifar10(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load and preprocess CIFAR-10 dataset"""

        def _augment(image, label):
            """Data augmentation for training"""
            # Pad + crop to inject spatial jitter similar to standard CIFAR policy
            image = tf.pad(image, [[4, 4], [4, 4], [0, 0]], mode='REFLECT')
            image = tf.image.random_crop(image, size=self.config.input_shape_cifar10)

            # Horizontal flip and mild color jitter
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.15)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            image = tf.image.random_saturation(image, 0.8, 1.2)

            # Random erasing style cutout (single patch)
            erase_prob = 0.5
            def _cutout():
                mask_size = tf.random.uniform([], 8, 16, dtype=tf.int32)
                offset_x = tf.random.uniform([], 0, 32 - mask_size, dtype=tf.int32)
                offset_y = tf.random.uniform([], 0, 32 - mask_size, dtype=tf.int32)
                mask = tf.ones([mask_size, mask_size, 3], dtype=image.dtype)
                paddings = [[offset_y, 32 - mask_size - offset_y],
                            [offset_x, 32 - mask_size - offset_x],
                            [0, 0]]
                mask = tf.pad(mask, paddings, constant_values=0.0)
                return image * (1.0 - mask)

            image = tf.cond(tf.random.uniform([], 0, 1) < erase_prob, _cutout, lambda: image)
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image, label
        
        def _prepare_dataset(dataset: tf.data.Dataset, is_training: bool) -> tf.data.Dataset:
            """Prepare dataset with batching and prefetching"""
            if is_training:
                dataset = dataset.shuffle(buffer_size=10000)
            
            dataset = dataset.batch(self.config.batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

            return dataset
        
        def _create_dataset(x, y, is_training: bool) -> tf.data.Dataset:
                """Create tf.data dataset from numpy arrays"""
                dataset = tf.data.Dataset.from_tensor_slices((x, y))
                
                if is_training and self.config.data_augmentation:
                    dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
                
                return _prepare_dataset(dataset, is_training)

        # Load dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
        # Normalize to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Convert to categorical
        y_train = tf.keras.utils.to_categorical(y_train, self.config.num_classes_cifar10)
        y_test = tf.keras.utils.to_categorical(y_test, self.config.num_classes_cifar10)
        
        # Stratified shuffle before splitting so validation mirrors training distribution
        rng = np.random.default_rng(SEED)
        indices = np.arange(len(x_train))
        rng.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]

        val_size = int(len(x_train) * self.config.validation_split)
        x_val, y_val = x_train[:val_size], y_train[:val_size]
        x_train, y_train = x_train[val_size:], y_train[val_size:]
        
        # Create tf.data datasets
        train_ds = _create_dataset(x_train, y_train, is_training=True)
        val_ds = _create_dataset(x_val, y_val, is_training=False)
        test_ds = _create_dataset(x_test, y_test, is_training=False)
        
        return train_ds, val_ds, test_ds
    
    def load_cifar100(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load and preprocess CIFAR-100 dataset"""

        def _augment(image, label):
            """Data augmentation for training"""
            image = tf.pad(image, [[4, 4], [4, 4], [0, 0]], mode='REFLECT')
            image = tf.image.random_crop(image, size=self.config.input_shape_cifar100)
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.15)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            image = tf.image.random_saturation(image, 0.8, 1.2)

            erase_prob = 0.5

            def _cutout():
                mask_size = tf.random.uniform([], 8, 16, dtype=tf.int32)
                offset_x = tf.random.uniform([], 0, 32 - mask_size, dtype=tf.int32)
                offset_y = tf.random.uniform([], 0, 32 - mask_size, dtype=tf.int32)
                mask = tf.ones([mask_size, mask_size, 3], dtype=image.dtype)
                paddings = [[offset_y, 32 - mask_size - offset_y],
                            [offset_x, 32 - mask_size - offset_x],
                            [0, 0]]
                mask = tf.pad(mask, paddings, constant_values=0.0)
                return image * (1.0 - mask)

            image = tf.cond(tf.random.uniform([], 0, 1) < erase_prob, _cutout, lambda: image)
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image, label
        
        def _prepare_dataset(dataset: tf.data.Dataset, is_training: bool) -> tf.data.Dataset:
            """Prepare dataset with batching and prefetching"""
            if is_training:
                dataset = dataset.shuffle(buffer_size=10000)
            
            dataset = dataset.batch(self.config.batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

            return dataset
        
        def _create_dataset(x, y, is_training: bool) -> tf.data.Dataset:
                """Create tf.data dataset from numpy arrays"""
                dataset = tf.data.Dataset.from_tensor_slices((x, y))
                
                if is_training and self.config.data_augmentation:
                    dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
                
                return _prepare_dataset(dataset, is_training)

        # Load dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        
        # Normalize to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Convert to categorical
        y_train = tf.keras.utils.to_categorical(y_train, self.config.num_classes_cifar100)
        y_test = tf.keras.utils.to_categorical(y_test, self.config.num_classes_cifar100)
        
        rng = np.random.default_rng(SEED)
        indices = np.arange(len(x_train))
        rng.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]

        val_size = int(len(x_train) * self.config.validation_split)
        x_val, y_val = x_train[:val_size], y_train[:val_size]
        x_train, y_train = x_train[val_size:], y_train[val_size:]
        
        # Create tf.data datasets
        train_ds = _create_dataset(x_train, y_train, is_training=True)
        val_ds = _create_dataset(x_val, y_val, is_training=False)
        test_ds = _create_dataset(x_test, y_test, is_training=False)
        
        return train_ds, val_ds, test_ds
    
    def load_stanford_dogs(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load and preprocess Stanford Dogs dataset"""
        

        # Preprocessing and Augmentation functions
        vgg16_preprocess = tf.keras.applications.vgg16.preprocess_input  # required for VGG16
        # Data augmentation (on-the-fly, GPU-friendly)
        data_augment = tf.keras.Sequential(
                [
                    layers.RandomFlip("horizontal"),
                    layers.RandomRotation(0.05),
                    layers.RandomZoom(0.1),
                    layers.RandomContrast(0.1),
                ],
                name="augment"
            )

        def _resize(image):
            # antialias=True reduces distortions on down/up sampling
            return tf.image.resize(image, (224, 224), method="bicubic", antialias=True)

        def _preprocess_train(image, label):
            image = _resize(image)
            image = tf.cast(image, tf.float32)
            image = data_augment(image)

            # VGG16 preprocess expects [0,255] range; if in [0,1], scale up.
            image = vgg16_preprocess(image)  # handles RGB->BGR & channel-wise mean-centering
            label = tf.one_hot(label, self.config.num_classes_stanford_dogs)

            return image, label

        def _preprocess_eval(image, label):
            image = _resize(image)
            image = tf.cast(image, tf.float32)
            image = vgg16_preprocess(image)
            label = tf.one_hot(label, self.config.num_classes_stanford_dogs)

            return image, label
        
        print("Loading Stanford Dogs dataset...")
        raw_train, raw_val, raw_test = tfds.load(
            "stanford_dogs",
            split=["train[:90%]", "train[90%:]", "test"],
            as_supervised=True,
            shuffle_files=True,
            with_info=False
        )

        print("Preprocessing data...")

        def _prepare(dataset: tf.data.Dataset,
                     preprocess_fn,
                     num_examples: Optional[int],
                     shuffle: bool = False) -> tf.data.Dataset:
            if shuffle:
                dataset = dataset.shuffle(10_000, seed=SEED, reshuffle_each_iteration=True)

            dataset = dataset.map(preprocess_fn, num_parallel_calls=AUTOTUNE)
            dataset = dataset.batch(self.config.batch_size)

            # Preserve a known cardinality so downstream schedulers behave as expected.
            if num_examples is not None:
                total_batches = math.ceil(num_examples / self.config.batch_size)
                dataset = dataset.apply(tf.data.experimental.assert_cardinality(total_batches))

            return dataset.prefetch(AUTOTUNE)

        def _count_examples(dataset: tf.data.Dataset) -> Optional[int]:
            count = tf.data.experimental.cardinality(dataset).numpy()
            return int(count) if count >= 0 else None

        train_examples = _count_examples(raw_train)
        val_examples = _count_examples(raw_val)
        test_examples = _count_examples(raw_test)

        train_ds = _prepare(raw_train, _preprocess_train, train_examples, shuffle=True)
        val