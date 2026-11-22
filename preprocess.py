import cv2
from pathlib import Path

def preprocess_image(image, output_size=(20, 20)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(
        gray,
        output_size,
        interpolation=cv2.INTER_NEAREST
    )

    return gray, resized

def process_dataset(input_directory, output_directory, size):
    input_directory = Path(input_directory)
    output_directory = Path(output_directory)

    extensions = {".png", ".jpg", ".jpeg"}

    processed_count = 0

    for file_path in sorted(input_directory.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in extensions:
            continue

        image = cv2.imread(str(file_path))
        if image is None:
            print(f"Could not read: {file_path}")
            continue

        gray, resized = preprocess_image(image, (size, size))

        relative_path = file_path.relative_to(input_directory)

        output_path = output_directory / relative_path.with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(output_path), resized):
            raise OSError(f"Could not save: {output_path}")

        processed_count += 1

    print(f"Processed {processed_count} images into {output_directory}")

if __name__ == "__main__":
    process_dataset("Data/Train", "Processed/Size20/Train", 20)
    process_dataset("Data/Train", "Processed/Size30/Train", 30)
    process_dataset("Data/Train", "Processed/Size40/Train", 40)
    process_dataset("Data/Train", "Processed/Size50/Train", 50)
    process_dataset("Data/Test", "Processed/Size20/Test", 20)
    process_dataset("Data/Test", "Processed/Size30/Test", 30)
    process_dataset("Data/Test", "Processed/Size40/Test", 40)
    process_dataset("Data/Test", "Processed/Size50/Test", 50)