# See https://huggingface.co/docs/datasets/en/create_dataset

from datasets import load_dataset, DatasetDict
import os

def load_data_and_save_dictionary(dataset, image_type):
    try:
        if not os.path.exists(f"{dataset}_dictionary_{image_type}_images_224_x_224"):
            print(f"Creating dictionary for {image_type} images of {dataset}.")

            # Load the training, validation and testing sets
            train_dataset = load_dataset("imagefolder", data_dir=f"train_{dataset}_{image_type}_images_224_x_224")
            validation_dataset = load_dataset("imagefolder", data_dir=f"val_{dataset}_{image_type}_images_224_x_224")
            test_dataset = load_dataset("imagefolder", data_dir=f"test_{dataset}_{image_type}_images_224_x_224")

            # Create a DatasetDict directly with the train, validation and testing datasets.
            dataset_dict = DatasetDict({
                'train': train_dataset['train'],
                'validation': validation_dataset['train'],
                'test': test_dataset['train']
            })

            dataset_dict.save_to_disk(f"{dataset}_dictionary_{image_type}_images_224_x_224")
            print("Dictionary saved")
        else:
            print(f"Dictionary for {image_type} images of {dataset} already exists.")
    except Exception as e:
        print(f"Error: {e}")



datasets = ["dataset_1", "dataset_2", "dataset_3"]

for dataset in datasets:
    for image_type in ["gri", "urz", "side_by_side"]:
        load_data_and_save_dictionary(dataset, image_type)

