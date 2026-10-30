from dataset import load_dataset, CLASSES
import matplotlib.pyplot as plt
import numpy as np

def show_samples(images, labels, classes, count=8):
    count = min(count, len(images))

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    for i, ax in enumerate(axes.flat):
        if i < count:
            ax.imshow(images[i], cmap="gray" if images[i].ndim == 2 else None)
            ax.set_title(classes[labels[i]])

        ax.axis("off")

    plt.tight_layout()
    plt.show()

def show_class_samples(images, labels, classes):
    fig, axes = plt.subplots(1, len(classes), figsize=(12, 3))

    for label, class_name in enumerate(classes):
        index = np.where(labels == label)[0][0]

        axes[label].imshow(images[index])
        axes[label].set_title(class_name)
        axes[label].axis("off")

    plt.tight_layout()
    plt.show()

train_images, train_labels = load_dataset("Data/Train")
test_images, test_labels = load_dataset("Data/Test")

print("\nTraining dataset:")
print("Images shape: ", train_images.shape)
print("Labels shape: ", train_labels.shape)

print("\nTesting dataset:")
print("Images shape: ", test_images.shape)
print("Labels shape: ", test_labels.shape)

show_class_samples(train_images, train_labels, CLASSES)