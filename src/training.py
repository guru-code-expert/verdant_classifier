"""Main training script."""

import tensorflow as tf
from config.settings import *
from src.data_loader import load_dataset, get_dataset_partitions_tf, apply_performance_optimizations
from src.preprocessing import get_data_augmentation
from src.model import build_cnn_model
from src.utils import plot_sample_images, plot_training_history
from src.evaluation import print_evaluation_report

def main():
    # Load dataset (sparse labels for SparseCategoricalCrossentropy)
    dataset = load_dataset(label_mode="int")
    class_names = dataset.class_names
    print("Classes:", class_names)
    
    # Auto-detect number of classes
    global NUM_CLASSES
    NUM_CLASSES = len(class_names)
    
    # Split
    train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)
    
    # Optimize pipelines
    train_ds = apply_performance_optimizations(train_ds)
    val_ds = apply_performance_optimizations(val_ds)
    test_ds = apply_performance_optimizations(test_ds)
    
    # Data augmentation only on training set
    aug_layer = get_data_augmentation()
    train_ds = train_ds.map(
        lambda x, y: (aug_layer(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    
    # Optional: visualize samples
    # plot_sample_images(train_ds, class_names)
    
    # Build & compile model
    model = build_cnn_model(num_classes=NUM_CLASSES)
    model.summary()
    
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    # Train
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        verbose=1
    )
    
    # Plot curves
    plot_training_history(history)
    
    # Evaluate
    print_evaluation_report(model, test_ds, "Test Set")
    
    # Save
    save_path = PROJECT_ROOT / "models" / "verdant_classifier_final"
    model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()