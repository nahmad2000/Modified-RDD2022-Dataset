# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 20:34:53 2023

@author: ahmad
"""

# Importation

import matplotlib.pyplot as plt
import os 
import xml.etree.ElementTree as ET
import random
import shutil
import yaml
import numpy as np


RDD_dataset_path = r'' # Path to RDD2022 dataset as described in github
pothole_dataset_path = r'' # Path to Pothole dataset as described in github

all_classes = ['D00', 'D01', 'D10', 'D11', 'D20', 'D40', 'D43', 'D44']
remove_labels = ['D40', 'D43', 'D44']
convert_labels = {('D00', 'D01', 'D10', 'D11'):'Linear-Crack', ('D20',):'Alligator-Crack'}
final_classes = ['Linear-Crack', 'Alligator-Crack', 'pothole']

#%%

def check_RDD_dataset(dataset_path):
    '''
    This function will check whether each label has a corresponding image or not.
    If a label does not have a corresponding image, then a print statement will show the path of that label.
    
    It takes:
        - dataset_path (str): path to the dataset folder.
    '''
    
    print("\n\n*** Checking if each label file in the RDD dataset has an existing image ***")
    
    missing_images = 0
    
    # List of country folders in the dataset
    country_folders = os.listdir(dataset_path)
    
    for country in country_folders:
        
        print(f"Checking {country} dataset...")

        images_path = os.path.join(dataset_path, country, "train", "images")
        labels_path = os.path.join(dataset_path, country, "train", "annotations", "xmls")

        # Get a list of label files
        labels_files = os.listdir(labels_path)

        for label_file in labels_files:
            label_file_path = os.path.join(labels_path, label_file)
            image_file_path = os.path.join(images_path, label_file.replace(".xml", ".jpg"))
    
            if not os.path.exists(image_file_path):
                missing_images += 1
                print(f"Label {label_file_path} does not have a corresponding image.")
    
    if missing_images == 0:
        print("No missing images in the RDD dataset! You are ready to go")
    else:
        print(f"You have {missing_images} missing images in the RDD dataset... You need to do something about this...")



def check_pothole_dataset(dataset_path):
    '''
    This function will check whether each label has a corresponding image or not.
    If a label does not have a corresponding image, then a print statement will show the path of that label.
    
    It takes:
        - dataset_path (str): path to the dataset folder.
    '''
    
    print("\n\n*** Checking if each label file in the pothole dataset has an existing image ***")
    
    missing_images = 0
    
    images_path = os.path.join(dataset_path, "IMG")
    labels_path = os.path.join(dataset_path, "XML")

    # Get a list of label files
    labels_files = os.listdir(labels_path)

    for label_file in labels_files:
        label_file_path = os.path.join(labels_path, label_file)
        image_file_path = os.path.join(images_path, label_file.replace(".xml", ".jpg"))

        if not os.path.exists(image_file_path):
            missing_images += 1
            print(f"Label {label_file_path} does not have a corresponding image.")
                
    if missing_images == 0:
        print("No missing images in the pothole dataset! You are ready to go")
    else:
        print(f"You have {missing_images} missing images in the pothole dataset... You need to do something about this...")
        
        

def combine_coutries_datasets(RDD_dataset_path):
    '''
    This function will combine all countries dataset into one dataset.
    The combination process will contain:
        1) combining images together.
        2) combining txt files together.
    
    It takes:
        RDD_dataset_path (str): the path to the folder containing all coutries datasets.
        
    It returns:
        combined_RDD_dataset_path (str): the path to the combined coutries datasets.
    '''
    
    print("\n\n*** Combining RDD countries datasets into one single dataset ***")
    
    combined_RDD_dataset_path = os.path.join(RDD_dataset_path, "combined_RDD_dataset")
    all_images_path = os.path.join(combined_RDD_dataset_path, "images") 
    all_labels_path = os.path.join(combined_RDD_dataset_path, "labels")
    
    for folder in [combined_RDD_dataset_path, all_images_path, all_labels_path]:
        if not os.path.exists(folder):
            os.makedirs(folder)   
        
    for country in os.listdir(RDD_dataset_path):
        if country != "combined_RDD_dataset":
            print(f"Copying {country} dataset into combined_RDD_dataset...")
            
            country_path = os.path.join(RDD_dataset_path, country, "train")
            country_images_path = os.path.join(country_path, "images")
            country_labels_path = os.path.join(country_path, "annotations", "xmls")
            
            for label_file in os.listdir(country_labels_path):
                
                label_file_path = os.path.join(country_labels_path, label_file)
                destination_label_path = os.path.join(all_labels_path, label_file)
                shutil.copy(label_file_path, destination_label_path)
                
                image_file_path = os.path.join(country_images_path, label_file.replace(".xml", ".jpg"))
                destination_image_path = os.path.join(all_images_path, label_file.replace(".xml", ".jpg"))
                shutil.copy(image_file_path, destination_image_path)
            
    print("All countried dataset have been copied sucessfully to combined_RDD_dataset!")
    print(f"combined_RDD_dataset_path = {combined_RDD_dataset_path}")
    
    return combined_RDD_dataset_path



def remove_empty_images_and_labels(dataset, dataset_path):
    '''
    This function will go through all dataset labels and remove every empty label indicating no object present in the image.
    Additionally, the corresponding image will be deleted.
    It takes:
        - dataset (str): it can be either "RDD" or "pothole"
        - dataset_path (str): path of the dataset_path, either combined RDD dataset or pothole dataset.
    '''
    
    print("\n\n*** Checking for any empty labels and removing them with their corresponding images ***")
    empty_files = 0
    
    if dataset.lower() == "rdd":
        labels_path = os.path.join(dataset_path, 'labels')
        images_path = os.path.join(dataset_path, 'images')
    elif dataset.lower() == "pothole":
        labels_path = os.path.join(dataset_path, 'XML')
        images_path = os.path.join(dataset_path, 'IMG')
    else:
        print("Invalid 'dataet' name... please provide one of the following 'dataset' names: [RDD, pothole]")
        return
    
    for label_file in os.listdir(labels_path):
        label_file_path = os.path.join(labels_path, label_file)
        image_file_path = os.path.join(images_path, label_file.replace(".xml", ".jpg"))
        
        tree = ET.parse(label_file_path)
        root = tree.getroot()
        
        
        if len(root.findall(".//object")) == 0:
            empty_files += 1
            # If the label file has no objects, delete both the label and image files
            os.remove(label_file_path)
            os.remove(image_file_path)
    
    print(f"There are {empty_files} empty labels and corresponding images that have been removed from the {dataset} dataset.")    
 
        
    
    

def remove_classes(dataset_path, remove_labels):
    '''
    This function will iterate throw all txt files and check for each one of them if it contains object from these given "remove labels" list.
    If it doesn't containt any object from this "remove labels" list, then nothing will be done...
    However, if a file contains an object from this "remove labels" list, then it will check:
        if this txt file contains only objects from "remove labels" list, then the txt file will be deleted and its corresponding image will also be deleted.
        and if this txt file contians objects from outside the "remove labels" list,
        then only the related information to the objects from the "remove labels" list will be delted from the txt file.
        
    It takes:
        dataset_path (str): the path of the combined RDD dataset.
        remove_labels (list): list of strings containing the labels to be removed.
    
    '''
    
    print(f"\n\n*** Reomving the following labels {remove_labels} from the combined RDD dataset ***")
    
    images_path = os.path.join(dataset_path, 'images')
    labels_path = os.path.join(dataset_path, 'labels')
    
    removed_files = 0
    
    for label_file in os.listdir(labels_path):
        label_file_path = os.path.join(labels_path, label_file)
        image_file_path = os.path.join(images_path, label_file.replace(".xml", ".jpg"))
        
        tree = ET.parse(label_file_path)
        root = tree.getroot()
        
        # Create a flag to check if the label file should be deleted
        delete_label_file = False
        for obj in root.findall(".//object"):
            name = obj.find("name").text
            if name in remove_labels:
                # If the object's name is in the "remove_labels" list, mark the label file for deletion
                delete_label_file = True
                root.remove(obj)  # Remove the object from the XML
            
        if delete_label_file:
            # If the label file contains only objects from "remove_labels" list, delete the label and image files
            if len(root.findall(".//object")) == 0:
                os.remove(label_file_path)
                os.remove(image_file_path)
                removed_files +=1
            else:
                # If the label file contains other objects as well, rewrite the label file
                tree.write(label_file_path)

                
    print(f"{removed_files} labels containing objects from {remove_labels} have been removed successfully with their corresponding images!")

    
    

    
def convert_classes(dataset_path, convert_labels):
    '''
    This function will iterate throw all txt files and convert the labels according the given "convert labels" dictionary.
    
    It takes:
        dataset_path (str): the path of the combined RDD dataset.
        convert_labels (dict): a dictionary containing the previous labels (list of strings) and new labels (str).
    '''
    
    print("\n\n*** Converting the following labels (from the combined RDD dataset) ***")
    for prev_labels, new_label in convert_labels.items():
        for item in prev_labels:
            print(f"{item} ==> {new_label}")
    print()
    labels_path = os.path.join(dataset_path, 'labels')
    
    for label_file in os.listdir(labels_path):
        label_file_path = os.path.join(labels_path, label_file)
        
        tree = ET.parse(label_file_path)
        root = tree.getroot()
        
        # Create a flag to check if the label file needs to be rewritten
        update_label_file = False
        
        for obj in root.findall(".//object"):
            name = obj.find("name").text
            
            for prev_labels, new_label in convert_labels.items():
                if name in prev_labels:
                    # Replace the name with the new label
                    obj.find("name").text = new_label
                    update_label_file = True
        
        if update_label_file:
            # If the label file has been updated, rewrite it
            tree.write(label_file_path)
    
    print("Labels have been converted successfully in the dataset!")
            
    



def merge_datasets(combined_RDD_dataset_path, pothole_dataset_path):
    '''
    This function will take the path to the two datasets and create a new merged dataset.
    The merge process will contain:
        1) merging images together.
        2) merging txt files together.
    
    It takes:
        combined_RDD_dataset_path (str): the path of the combined RDD dataset.
        pothole_dataset_path (str): the path of the pothole dataset.
    '''
    
    print("\n\n*** Moving 'pothole' dataset into 'combined RDD' dataset ***")
    
    rdd_images_path = os.path.join(combined_RDD_dataset_path, 'images')
    rdd_labels_path = os.path.join(combined_RDD_dataset_path, 'labels')
    
    pothole_images_path = os.path.join(pothole_dataset_path, 'IMG')
    pothole_labels_path = os.path.join(pothole_dataset_path, 'XML')
    
        
    print("Copying labels from 'pothole' dataset into 'combine RDD' dataset...")
    for label_file in os.listdir(pothole_labels_path):
        label_file_path = os.path.join(pothole_labels_path, label_file)
        final_label_path = os.path.join(rdd_labels_path, label_file)
        shutil.copy(label_file_path, final_label_path)
        
        
    print("Copying images from 'pothole' dataset into 'combine RDD' dataset...")
    for image_file in os.listdir(pothole_images_path):
        image_file_path = os.path.join(pothole_images_path, image_file)
        final_image_path = os.path.join(rdd_images_path, image_file)
        shutil.copy(image_file_path, final_image_path)

    
    print("\nPothole dataset has been successfully Moved into the 'combined RDD' dataset.")
    


def convert_annotation(dataset_path, final_classes):
    '''
    This function will convert the annotation format from the original format (similar to pascal voc) to YOLOv8 format.
    This will create a new "labels" folder for each "annotations" folder and a YAML file for class mapping.
        
    It takes:
        - dataset_path (str): path to the combined RDD dataset folder.
        - final_classes (list): list of strings containing all classes.
    '''
    
    print("\n\n*** Converting annotations from xml format to YOLO format *** ")
    
    old_xmls_path = os.path.join(dataset_path, "labels")
    xmls_path = os.path.join(os.path.dirname(old_xmls_path), "xmls")
    shutil.move(old_xmls_path, xmls_path)



    # Create a "labels" folder for YOLOv8 format
    labels_path = os.path.join(dataset_path, "labels")
    os.makedirs(labels_path, exist_ok=True)

    # Create a YAML file for class mapping
    yaml_file = os.path.join(dataset_path, "class_mapping.yaml")
    with open(yaml_file, "w") as f:
        for i, class_name in enumerate(final_classes):
            f.write(f'"{i}": "{class_name}"\n')  # Use double quotes for class names

    # Iterate through XML files and convert to YOLO format
    for xml_file in os.listdir(xmls_path):
        if xml_file.endswith(".xml"):
            tree = ET.parse(os.path.join(xmls_path, xml_file))
            root = tree.getroot()

            image_width = int(root.find(".//width").text)
            image_height = int(root.find(".//height").text)

            yolo_lines = []
            for obj in root.findall(".//object"):
                class_name = obj.find("name").text
                if class_name in final_classes:
                    class_index = final_classes.index(class_name)

                    
                    # Ensure coordinates are integers and within valid range
                    xmin = min(int(round(float(obj.find(".//xmin").text))), image_width - 1)
                    ymin = min(int(round(float(obj.find(".//ymin").text))), image_height - 1)
                    xmax = min(int(round(float(obj.find(".//xmax").text))), image_width - 1)
                    ymax = min(int(round(float(obj.find(".//ymax").text))), image_height - 1)


                    # Normalize coordinates
                    x_center = (xmin + xmax) / (2.0 * image_width)
                    y_center = (ymin + ymax) / (2.0 * image_height)
                    width = (xmax - xmin) / image_width
                    height = (ymax - ymin) / image_height

                    yolo_lines.append(f"{class_index} {x_center} {y_center} {width} {height}")

            # Write YOLO format annotation to a .txt file
            output_txt_file = os.path.splitext(xml_file)[0] + ".txt"
            output_txt_path = os.path.join(labels_path, output_txt_file)
            with open(output_txt_path, "w") as f:
                f.write("\n".join(yolo_lines))

    print("Annotations have been converted successfully!")



def check_again_empty_labels(dataset_path):
    
    print("\n\n ***Checking again for any empty labels***\n")
    labels_path = os.path.join(dataset_path, 'labels')
    images_path = os.path.join(dataset_path, 'images')
    
    empty_files = 0
    for label_file in os.listdir(labels_path):
        label_file_path = os.path.join(labels_path, label_file)
        image_file_path = os.path.join(images_path, label_file.replace('.txt', '.jpg'))
        with open(label_file_path, 'r') as file:
            lines = file.readlines()
        if len(lines) == 0:
            empty_files += 1
            os.remove(label_file_path)
            os.remove(image_file_path)
    
    print(f"There are {empty_files} empty files that have been removed with thier corresponding images")
    
    
    
def get_dataset_statistics(dataset_path, final_classes):
    '''
    This function will loop through all label files in the dataset to generate a dictionary containing
    the number of objects per class for the entire dataset.
    
    It takes:
        - dataset_path (str): path to the final combined dataset folder.
        - final_classes (list): list of strings of the classes/labels avaliable in the dataset
    It returns:
        - dataset_statistics (dict): Dictionary containing the number of objects per class for the entire dataset.
    '''
    
    print("\n\n*** Getting Dataset Statistics ***")
    dataset_statistics = {class_idx: 0 for class_idx in range(len(final_classes))}
    

    labels_path = os.path.join(dataset_path, 'labels')

    for label_file in os.listdir(labels_path):
        label_file_path = os.path.join(labels_path, label_file)
        
        with open(label_file_path, 'r') as file:
            lines = file.readlines()
            
            for line in lines:
                object_idx = int(line.split()[0])
                if object_idx in dataset_statistics:
                    dataset_statistics[object_idx] += 1

    print(f"dataset_statistics dictionary has been generated successfully!\n{dataset_statistics}")
    return dataset_statistics
   


def undersampling_majority_classes(dataset_path, dataset_statistics):
    '''
    This function will perform undersampling to majority classes by removing a big amount of label files with their corresponding images.
    It takes:
        - dataset_path (str): path to the final combined dataset folder.
        - dataset_statistics (dict): dictionary containing the number of objects per class.
    '''
    
    print("\n\n*** Performing undersampling ***")
    print(f"dataset_statistics before undersampling: \n{dataset_statistics}\n")
    
    threshold = np.power(10, np.sum([np.log10(n_objects) for n_objects in dataset_statistics.values()])/len(dataset_statistics))
    margin = np.power(np.log(threshold),2)
    hardness = 0.6 
    
    
    labels_path = os.path.join(dataset_path, 'labels')
    images_path = os.path.join(dataset_path, 'images')
    
    empty_files = 0
    
    for clss in dataset_statistics.keys():
        if dataset_statistics[clss]  >= (threshold + margin):
            for label in os.listdir(labels_path):
                if dataset_statistics[clss]  >= (threshold + margin):
                    label_file_path =  os.path.join(labels_path, label)
                    image_file_path =  os.path.join(images_path, label.replace('.txt', '.jpg'))
                    
                    with open(label_file_path, 'r') as file:
                        lines = file.readlines()
                        objects = [int(line.split()[0]) for line in lines]
                    
                    if len(objects) == 0:
                        empty_files += 1
                        os.remove(label_file_path)
                        os.remove(image_file_path)
                    else:
                        num_clss_obj = objects.count(clss)
                        prob_obj = num_clss_obj/len(objects)
                        if prob_obj > hardness:
                            for obj in objects:
                                for dictkey in dataset_statistics.keys():
                                    if obj == dictkey:
                                        dataset_statistics[dictkey] -= 1
                                        break
                            os.remove(label_file_path)
                            os.remove(image_file_path)   

     
    print("\nFinished undersampling!\n")
    print(f"dataset_statistics after undersampling: \n{dataset_statistics}\n")
    print(f"Additionally, {empty_files} empty label files were found and deleted with their corresponding images.\n")
        
    return dataset_statistics




def split_dataset(dataset_path, dataset_statistics):
    
    '''
    This function will split the dataset into train/valid/test with a split ratio 0.7/0.1/0.2
    
    It takes:
        - dataset_path (str): path to the dataset folder. 
        - dataset_statistics (dict): dictionary containing the number of objects per class.

        
    The combined_dataset_dir contains only "train" folder, so after runing this function:
        We will have 2 more folders: "valid" and "test"
    
    This function will take into account not only the number of images in train/valid/test. It will also
    take into account the number of objects per class in train/valid/test.
    
    Approach to achieve this:
        1) rearrange the dataset_statistics dict to have classes ranked from min to max ==> call it 'sorted_classes'.
        2) iterate thorugh keys of dataset_statistics dictionary class by class:
        3) for class_idx in sorted_classes:
            3.1) create empty files_list list
            3.2) itertate thoruh all labels
            3.3) if a label contains an object_idx = class_idx, then add this label file to files_list
            3.4) if len(files_list) = dataset_statistics[class_idx]: stop itertating through labels
            3.5) else: keep iterating unitl you finish all labels
            3.6) randomly shuffle the files_list
            3.7) move 10% of files_list into "valid" folder
            3.8) move 20% of files_list into "test" folder
            3.9) 70% of files_list should remiain in same path "train" folder
            3.10) clear files_list
    '''
    
    print("\n\n*** Splitting the dataset into trina/valid/test ***")
    
    # Define the split ratios
    train_ratio = 0.7
    valid_ratio = 0.1
    #test_ratio = 0.2
    
    hardness = 0.5
  
    # Create the train/valid/test folders if they don't exist
    for folder in ["train", "valid", "test"]:
        for sub_folder in ["images", "labels"]:
            folder_path = os.path.join(dataset_path, folder, sub_folder)
            os.makedirs(folder_path, exist_ok=True)
    
    # Rearrange the dataset_statistics dict to have classes ranked from min to max
    sorted_classes = sorted(dataset_statistics.keys(), key=lambda k: dataset_statistics[k])
    sorted_classes.append(-1) # This will be used to transfer remaining files
    
    labels_dir = os.path.join(dataset_path, "labels")
    images_dir = os.path.join(dataset_path, "images")
        
    for class_idx in sorted_classes:
        
        # Create an empty list to collect label files for this class
        files_list = []

        if class_idx == -1: # Transfer remaining files
            files_list = os.listdir(labels_dir)
        else:
            # Get the number of label files for this class
            max_objects = dataset_statistics[class_idx]
            num_objects = 0
    
            for label_file in os.listdir(labels_dir):
                if num_objects <= max_objects:
                    label_path = os.path.join(labels_dir, label_file)
                    
                    with open(label_path, 'r') as file:
                        lines = file.readlines()
                        objects = [int(line.split()[0]) for line in lines]
                        
                    identical_objects = objects.count(class_idx)
                    prob_obj = identical_objects/len(objects)
                    if prob_obj > hardness:
                        files_list.append(label_file)
                        num_objects += identical_objects
                    
                    else:
                        break

        
        # Randomly shuffle the list
        random.shuffle(files_list)
        
        # Determine the split sizes
        num_train = int(train_ratio * len(files_list))
        num_valid = int(valid_ratio * len(files_list))
        
        # Split the files_list into train, valid, and test
        train_files = files_list[:num_train]
        valid_files = files_list[num_train:num_train + num_valid]
        test_files = files_list[num_train + num_valid:]
        
        # Move files to the corresponding folders
        splitting_dict = {'train': train_files, 'valid': valid_files, 'test': test_files}
        
        for splitting_folder, splitting_files in splitting_dict.items():
            if class_idx == -1:
                print(f"Moving remaining files into {splitting_folder} folder...")
            else:
                print(f"Moving class {class_idx} files into {splitting_folder} folder...")
            for file in splitting_files:
                source_label_path = os.path.join(labels_dir, file)
                source_image_path = os.path.join(images_dir, file.replace(".txt", ".jpg"))
                dest_label_path = os.path.join(dataset_path, splitting_folder, "labels", file)
                dest_image_path = os.path.join(dataset_path, splitting_folder, "images", file.replace(".txt", ".jpg"))
                
                shutil.move(source_label_path, dest_label_path)
                shutil.move(source_image_path, dest_image_path)
    
    # Remove unwanted folders
    shutil.rmtree(labels_dir)
    shutil.rmtree(images_dir)
    xml_dir = labels_dir.replace('labels', 'xmls')
    shutil.rmtree(xml_dir)
    old_yaml = os.path.join(dataset_path, 'class_mapping.yaml')
    os.remove(old_yaml)
    
    
    print("\nDataset has been split successfully into train/valid/test!")
    

    
def plot_class_distribution(dataset_path, final_classes):
    '''
    This function will print and plot the class distribution after we split dataset into train, valid, test
    '''
    
    print("\n\n*** Plotting the class distrubution ***")
    dataset_split = ["train", "valid", "test"]
    class_counts = {split: {class_name: 0 for class_name in final_classes} for split in dataset_split}
    
    for split in dataset_split:
        for class_name in final_classes:
            labels_dir = os.path.join(dataset_path, split, "labels")
            for label_file in os.listdir(labels_dir):
                label_path = os.path.join(labels_dir, label_file)
                
                with open(label_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        if int(line.split()[0]) == final_classes.index(class_name):
                            class_counts[split][class_name] += 1
    
    # Print the results
    for split in dataset_split:
        print(f"Class distribution in {split} split:")
        for class_name in final_classes:
            print(f"{class_name}: {class_counts[split][class_name]} objects")
        print()  # Add an empty line for separation
    
    # Create a bar plot
    x = range(len(final_classes))
    width = 0.2
    for split in dataset_split:
        counts = [class_counts[split][class_name] for class_name in final_classes]
        plt.bar([i + width * dataset_split.index(split) for i in x], counts, width, label=split)
    
    plt.xlabel('Classes')
    plt.ylabel('Number of Objects')
    plt.title('Class Distribution in Train, Valid, and Test Splits')
    plt.xticks([i + width for i in x], final_classes)
    plt.legend()
    
    plt.show()




def create_yaml(dataset_path, final_classes):
    '''
    This function will create a yaml file for a given dataset.
    The created yaml file will be in the same given dataset path.
    
    It takes:
        - dataset_path (str): path to the dataset folder. 
        - final_classes (list): list of strings containing all final_classes.
        
    '''
    
    print("\n\n*** Creating the YAML file ***")
    
    data = {
        'train': f'{os.path.join(dataset_path, "train", "images")}',
        'val': f'{os.path.join(dataset_path, "valid", "images")}',
        'nc': len(final_classes),
        'names': final_classes
    }

    yaml_content = yaml.dump(data, default_flow_style=False)

    yaml_file_path = os.path.join(dataset_path, 'data.yaml')
    
    with open(yaml_file_path, 'w') as yaml_file:
        yaml_file.write(yaml_content)
        
    print("Finally... We are done!!!\nDataset is ready for use")


#%%

check_RDD_dataset(RDD_dataset_path)
check_pothole_dataset(pothole_dataset_path)
combined_RDD_dataset_path = combine_coutries_datasets(RDD_dataset_path)
remove_empty_images_and_labels('pothole', pothole_dataset_path)
remove_empty_images_and_labels('RDD', combined_RDD_dataset_path)
remove_classes(combined_RDD_dataset_path, remove_labels)
convert_classes(combined_RDD_dataset_path, convert_labels)
merge_datasets(combined_RDD_dataset_path, pothole_dataset_path)
convert_annotation(combined_RDD_dataset_path, final_classes)
check_again_empty_labels(combined_RDD_dataset_path)
dataset_statistics = get_dataset_statistics(combined_RDD_dataset_path, final_classes)
dataset_statistics = undersampling_majority_classes(combined_RDD_dataset_path, dataset_statistics)
split_dataset(combined_RDD_dataset_path, dataset_statistics)
plot_class_distribution(combined_RDD_dataset_path, final_classes)
create_yaml(combined_RDD_dataset_path, final_classes)

