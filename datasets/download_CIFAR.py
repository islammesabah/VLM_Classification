
import os
import json
from tqdm import tqdm
import requests
import tarfile
import numpy as np
import argparse
from PIL import Image
import shutil

# dataset description 
'''Loaded in this way, each of the batch files contains a dictionary with the following elements:
data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
The dataset contains another file, called batches.meta. It too contains a Python dictionary object. It has the following entries:
label_names -- a 10-element list which gives meaningful names to the numeric labels in the labels array described above. For example, label_names[0] == "airplane", label_names[1] == "automobile", etc.
'''

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def read_images(path: str, batch_id: int) -> tuple[np.ndarray, list, dict[str, str]]:
    """
    Reads images from a specified batch file in the CIFAR-10 dataset.

    Args:
        path (str): The path to the CIFAR-10 batch file.
        batch_id (int): The ID of the batch to read.

    Returns:
        tuple: A tuple containing:
            - images (numpy.ndarray): An array of shape (10000, 32, 32, 3) containing the images.
            - labels (list): A list of labels corresponding to the images.
            - class_names (list): A list of class names.
    """
    batch_file = f"{path}/data_batch_{batch_id}"
    data_dict = unpickle(batch_file)
    
    # Reshape the data correctly: the data is in format [R, G, B] channels flattened
    raw_data = np.array(data_dict[b'data'], dtype=np.uint8)
    
    # Reshape to (10000, 3, 32, 32) then transpose to (10000, 32, 32, 3)
    images = raw_data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = data_dict[b'labels']
    
    # Read meta data
    meta_file = f"{path}/batches.meta"
    meta_dict = unpickle(meta_file)
    # class names is dict "0": "airplane", "1": "automobile", ..., "9": "truck"
    class_names = meta_dict[b'label_names']
    class_names = {str(i): name.decode('utf-8') for i, name in enumerate(class_names)}
    return images, labels, class_names

def save_images(images, labels, class_names, output_dir, batch_id, save_format='PNG'):
    """
    Save images to individual files.
    
    Args:
        images (numpy.ndarray): Array of images with shape (N, 32, 32, 3)
        labels (list): List of labels for each image
        class_names (list): List of class names
        output_dir (str): Directory to save images
        batch_id (int): Batch ID for naming
        save_format (str): Image format ('PNG', 'JPEG', etc.)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save class names
    with open(f"{output_dir}/class_names.json", "w") as f:
        json.dump(class_names, f, indent=4)

    output_dir = output_dir + "images"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories for each class
    for class_name in class_names:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    # Save each image
    for i, (image, label) in enumerate(zip(images, labels)):
        filename = f"batch_{batch_id}_image_{i:04d}.{save_format.lower()}"
        filepath = os.path.join(output_dir, str(label), filename)
        
        # Convert numpy array to PIL Image and save
        pil_image = Image.fromarray(image)
        pil_image.save(filepath, save_format)
        
        if i % 1000 == 0:  # Progress indicator
            print(f"Saved {i+1}/{len(images)} images...")
    
    print(f"All {len(images)} images from batch {batch_id} saved to {output_dir}")

def main():

    # add argument to save images or not
    parser = argparse.ArgumentParser(description="Download and process CIFAR-10 dataset.")
    parser.add_argument('--save_images', action='store_true', help="Save images to disk")
    args = parser.parse_args()

    # download the dataset
    path = "datasets/CIFAR-10/"
    response = requests.get("http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")

    with open(path+'cifar-10-python.tar.gz', "wb") as f:
        f.write(response.content)
            
    with tarfile.open(path+'cifar-10-python.tar.gz', 'r:gz') as tar:
        tar.extractall(path='./'+path)
            
    if os.path.exists(path+'cifar-10-python.tar.gz'):
        os.rename(path+'cifar-10-batches-py', path+'images')
        os.remove(path+'cifar-10-python.tar.gz')

        

    if args.save_images:
        print("Saving images...")
        # Set your paths
        downloaded_cifar10_path = path+"images"
        output_directory = path
            
        for batch_id in range(1, 6):
                
            # Read images from batch id
            images, labels, class_names = read_images(downloaded_cifar10_path, batch_id)
                
            print(f"Loaded {len(images)} images from batch {batch_id}")
            print(f"Image shape: {images[0].shape}")
            print(f"labels example: {labels[:10]}")
            print(f"Classes: {class_names}")
                
            # Save images organized by class in subdirectories
            save_images(images, labels, class_names, output_directory, batch_id, 'PNG')
                
        print(f"Images and metadata of CIFAR-10 dataset saved successfully!")

if __name__ == "__main__":
    main() 