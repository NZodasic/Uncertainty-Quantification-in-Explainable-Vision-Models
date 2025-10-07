import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from typing import Tuple
# from data.augmentation import DataAugmentation
from config import Config

class VGG16:
    """VGG16 model"""

    def __init__(self, config):
        self.config = config

    def build_cifar10_model(self, num_classes: int, input_shape: Tuple[int, int, int]) -> Model:
        """Build VGG16 model with BatchNormalization"""
        inputs = layers.Input(shape=input_shape)

        # L2 regularizer
        l2_reg = regularizers.l2(self.config.l2_regularization)

        # Block 1
        x = layers.Conv2D(64, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(inputs)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Spatial Dropout
        if self.config.use_spatial_dropout:
            x = layers.SpatialDropout2D(self.config.spatial_dropout_rate)(x)
        
        # Block 2
        x = layers.Conv2D(128, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Conv2D(128, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Spatial Dropout
        if self.config.use_spatial_dropout:
            x = layers.SpatialDropout2D(self.config.spatial_dropout_rate)(x)
        
        # Block 3
        x = layers.Conv2D(256, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Conv2D(256, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Conv2D(256, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.MaxPooling2D(2)(x)
        # Spatial Dropout
        if self.config.use_spatial_dropout:
            x = layers.SpatialDropout2D(self.config.spatial_dropout_rate)(x)
        
        # Block 4
        x = layers.Conv2D(512, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Conv2D(512, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Conv2D(512, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Spatial Dropout
        if self.config.use_spatial_dropout:
            x = layers.SpatialDropout2D(self.config.spatial_dropout_rate)(x)
        
        # Block 5
        x = layers.Conv2D(512, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Conv2D(512, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Conv2D(512, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Flatten and Dense layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu', kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        
        # Output layer with label smoothing
        outputs = layers.Dense(num_classes, activation='softmax',
                               kernel_regularizer=l2_reg)(x)
        
        model = Model(inputs, outputs, name='vgg_for_cifar10')
        return model
    
    # New: Predict with UQ (MC Dropout)
    def predict_with_uq(self, model: Model, inputs: tf.Tensor, num_samples: int = 20) -> Tuple[tf.Tensor, tf.Tensor]:
        """Inference with uncertainty using MC Dropout"""
        predictions = [model(inputs, training=True) for _ in range(num_samples)]
        predictions = tf.stack(predictions)
        mean = tf.reduce_mean(predictions, axis=0)
        variance = tf.math.reduce_variance(predictions, axis=0)
        return mean, variance.mean(axis=-1)  # Average variance per sample
    
    def build_cifar100_model(self, num_classes: int, input_shape: Tuple[int, int, int]) -> Model:
        """Build VGG16 model with BatchNormalization"""
        inputs = layers.Input(shape=input_shape)

        # L2 regularizer
        l2_reg = regularizers.l2(self.config.l2_regularization)

        # Block 1
        x = layers.Conv2D(64, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(inputs)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Spatial Dropout
        if self.config.use_spatial_dropout:
            x = layers.SpatialDropout2D(self.config.spatial_dropout_rate)(x)
        
        # Block 2
        x = layers.Conv2D(128, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Conv2D(128, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Spatial Dropout
        if self.config.use_spatial_dropout:
            x = layers.SpatialDropout2D(self.config.spatial_dropout_rate)(x)
        
        # Block 3
        x = layers.Conv2D(256, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Conv2D(256, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Conv2D(256, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.MaxPooling2D(2)(x)
        # Spatial Dropout
        if self.config.use_spatial_dropout:
            x = layers.SpatialDropout2D(self.config.spatial_dropout_rate)(x)
        
        # Block 4
        x = layers.Conv2D(512, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Conv2D(512, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Conv2D(512, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Spatial Dropout
        if self.config.use_spatial_dropout:
            x = layers.SpatialDropout2D(self.config.spatial_dropout_rate)(x)
        
        # Block 5
        x = layers.Conv2D(512, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Conv2D(512, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Conv2D(512, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Flatten and Dense layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu', kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        
        # Output layer with label smoothing
        outputs = layers.Dense(num_classes, activation='softmax',
                               kernel_regularizer=l2_reg)(x)
        
        model = Model(inputs, outputs, name='vgg_for_cifar100')
        return model
    
    def build_stanford_dogs_model(self, num_classes: int, input_shape: Tuple[int, int, int]) -> Model:
        """Load VGG16 with ImageNet weights and modify for Stanford Dogs"""
        
        backbone = tf.keras.applications.VGG16(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape
        )

        # Freeze all layers for warm-up
        for l in backbone.layers:
            l.trainable = False

        x = layers.GlobalAveragePooling2D(name="gap")(backbone.output)
        x = layers.BatchNormalization(name="bn")(x)
        x = layers.Dense(
            512, activation="relu", kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(self.config.weight_decay), name="fc_512")(x)
        x = layers.Dropout(self.config.dropout_rate, name="dropout")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

        model = Model(inputs=backbone.input, outputs=outputs, name="vgg16_stanford_dogs")
        return model