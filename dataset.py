from pathlib import Path
import cv2
import numpy as np

CLASSES=["Banana","Bell Pepper","Orange","Strawberry"]

def load_dataset(directory,image_size=(100,100),grayscale=False):
    images=[]
    labels=[]

    directory=Path(directory)

    for label,class_name in enumerate(CLASSES):
        class_directory=directory/class_name

        if not class_directory.exists():
            raise FileNotFoundError(f"Class folder not found: {class_directory}")

        count=0

        for file_path in sorted(class_directory.rglob("*")):
            if file_path.suffix.lower() not in {".jpg",".jpeg",".png"}:
                continue

            mode=cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
            image=cv2.imread(str(file_path),mode)

            if image is None:
                print(f"Could not load image: {file_path}")
                continue

            if not grayscale:
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

            image=cv2.resize(image,image_size)

            images.append(image)
            labels.append(label)
            count+=1

        print(f"{class_name}: {count} images")

    if not images:
        raise ValueError(f"No images found in: {directory}")

    return np.array(images,dtype=np.uint8),np.array(labels,dtype=np.int64)