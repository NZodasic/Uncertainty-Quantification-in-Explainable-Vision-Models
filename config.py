import os
from dataclasses import dataclass, field
from typing import Tuple, Dict
from datetime import datetime


@dataclass
class Config:
    """Configuration for FACCP pruning method"""

    # ========== DATA CONFIGURATION ==========
    task: str = 'cifar10'  # 'cifar10' or 'dogs'
    validation_split: float = 0.2
    data_augmentation: bool = True
    batch_size: int = 128

    # Dataset specific
    num_classes_cifar10: int = 10
    num_classes_cifar100: int = 100
    input_shape_cifar10: Tuple[int, int, int] = (32, 32, 3)
    input_shape_cifar100: Tuple[int, int, int] = (32, 32, 3)
    num_classes_stanford_dogs: int = 120
    input_shape_stanford_dogs: Tuple[int, int, int] = (224, 224, 3)

    # ========== TRAINING CONFIGURATION ==========
    default_epochs: int = 450  # | 200 for CIFAR-10 | 450 for CIFAR-100 |
    warmup_epochs: int = 10
    finetune_epochs: int = 100  # For Stanford Dogs
    growth_epochs: int = 100
    iterative_pruning_stages: int = 3  # | 0 for CIFAR-10 | 3 for CIFAR-100 |
    iterative_regrowth_epochs: int = 120 # | N/A for CIFAR-10 | 60 for CIFAR-100 |
    auto_finetune_after_prune: bool = True
    initial_lr: float = 7e-4 # | 1e-3 for CIAFR-10 | 7e-4 for CIFAR-100 |
    initial_lr_ft: float = 3e-4  # for fine-tuning
    pruned_growth_lr: float = 1e-3
    min_lr: float = 1e-7
    momentum: float = 0.9  # for SGD
    optimizer: str = 'adamw'  # 'adam', 'adamw', 'sgd', 'rmsprop'

    # Learning rate schedule
    lr_schedule: str = 'cosine'  # 'cosine', 'exponential', 'step' | 'cosine' for CIFAR-10 |
    lr_warmup_epochs: int = 10 # | 10 for CIFAR-10 |
    lr_decay_rate: float = 0.95 # | None for CIFAR-10 | 0.97 for CIFAR-100 |
    lr_decay_steps: int = 15 # | 10 for CIFAR-10 | 15 for CIFAR-100 |

    # Early Stopping (updated)
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    early_stopping_restore_best: bool = True
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5

    # Regularization
    l2_regularization: float = 1e-4
    batch_norm_momentum: float = 0.9

    dropout_rate: float = 0.5 # | 0.35 for CIFAR-10 | 0.5 for CIFAR-100
    use_spatial_dropout: bool = True
    spatial_dropout_rate: float = 0.25
    use_dropout_schedule: bool = True # | False for CIFAR-10 | True for CIFAR-100

    weight_decay: float = 1e-4

    # Label Smoothing
    label_smoothing: float = 0.08 # | 0.05 for CIFAR-10 | 0.08 for CIFAR-100 |

    # ========== PRUNING CONFIGURATION ==========
    base_pruning_ratio: float = 0.5  # Overall target pruning ratio
    # Frequency analysis
    frequency_bands: Dict[str, tuple] = None
    dct_block_size: int = 8

    # Comlexity weights
    complexity_weights: Dict[str, float] = None
    complexity_sensitivity: float = 0.5  # Sensitivity factor for complexity adjustment

    # ========== UNCERTAINTY QUANTIFICATION (UQ) CONFIGURATION ==========
    uq_num_samples: int = 20  # Number of MC Dropout samples for uncertainty estimation
    uq_threshold: float = 0.1  # Threshold for high uncertainty (for flagging or visualization)

    # ========== EXPLAINABLE AI (XAI) CONFIGURATION ==========
    xai_layer_name: str = 'conv5_block3_conv3'  # Default layer for Grad-CAM (last conv layer)
    xai_alpha: float = 0.5  # Alpha for heatmap overlay in visualization

    # ========== PATHs CONFIGURATION ==========
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%d%m%Y_%H%M%S"))
    checkpoint_dir: str = field(init=False)
    tensorboard_dir: str = field(init=False)
    results_dir: str = field(init=False)
    logs_dir: str = field(init=False)
    models_dir: str = field(init=False)
    plots_dir: str = field(init=False)

    def __post_init__(self):
        if self.frequency_bands is None:
            self.frequency_bands = {
                'low': (0.0, 0.25),
                'mid': (0.25, 0.5),
                'high': (0.5, 1.0)
            }

        if self.complexity_weights is None:
            self.complexity_weights = {
                'inter_class_confusion': 0.2,
                'feature_entropy': 0.2,
                'spatial_resolution': 0.3,
                'class_diversity': 0.3
            }
        # Build directories using the provided task and a timestamp run_id
        exp_root = f"./EXPERIMENT/{self.run_id}_{self.task}"
        self.checkpoint_dir = os.path.join(exp_root, "checkpoints")
        self.tensorboard_dir = os.path.join(exp_root, "tensorboard_logs")
        self.results_dir = os.path.join(exp_root, "results")
        self.logs_dir = os.path.join(exp_root, "logs")
        self.models_dir = os.path.join(exp_root, "models")
        self.plots_dir = os.path.join(exp_root, "plots")

        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)