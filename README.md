# Modified-RDD2022-Dataset
This code is designed to apply several modifications to the RDD2022 dataset, and combine it with one of two avaliable pothole datasets.

* The RDD2022 dataset: available at this [link](https://doi.org/10.48550/arXiv.2209.08538) and can be downloaded from this [link](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547?file=38030910)
* The first pothole dataset (small pothole dataset): avaliable at this [link](https://www.sciencedirect.com/science/article/pii/S2352340923003256?via%3Dihub) and can be downloaded from this [link](https://data.mendeley.com/datasets/tp95cdvgm8/1).
* The second pothole dataset (big pothole dataset): avaliable at this [link](https://learnopencv.com/train-yolov8-on-custom-dataset/) and can be downloaded from this [link](https://www.dropbox.com/s/qvglw8pqo16769f/pothole_dataset_v8.zip?dl=1).


# Original RDD2022 Dataset
The original RDD2022 dataset comprises eight classes, including:
* D00: Wheel mark part (Longitudinal) [Linear Crack]
* D01: Construction joint part (Longitudinal) [Linear Crack]
* D10: Equal interval (Lateral) [Linear Crack]
* D11: Construction joint part (Lateral) [Linear Crack]
* D20: Partial pavement, overall pavement (Alligator Crack)
* D40: Rutting, bump, pothole, separation (Other Corruption)
* D43: Crosswalk blur (Other Corruption)
* D44: White line blur (Other Corruption)


# Modifications to RDD2022 Dataset
The modifications applied to the RDD2022 dataset are as follows:
- Changing the following classes: [D00, D01, D10, D11] into a single class "Linear-Crack"
- Keeping the D20 class as "Alligator-Crack"
- Removing the following classes: [D40, D43, D44]
- Combining the "pothole" dataset with the modified RDD2022 dataset


# Additional Functions
Apart from the modifications, this code also performs the following tasks:
- Checks if each label file in each dataset has a corresponding image
- Converts annotations from ".xml" to ".txt"
- Performs undersampling of the majority classes to create a balanced dataset based on class objects
- Splits the dataset into train/valid/test sets with a ratio of 0.7/0.1/0.2 while maintaining a balanced split based on class objects
- Plots the class distribution after generating the desired dataset


# Running the Code
In general, there are two main sccipts for combining RDD2022 dataset with a pothole dataset [**preprocessing_1.py** and **preprocessing_2.py**]. 


1) **preprocessing_1.py** is used to combine **RDD2022 dataset** with the **samll pothole dataset**.
2) **preprocessing_2.py** is used to combine **RDD2022 dataset** with the **big pothole dataset**.

To run anyone of these scripts, follow these steps:

1) Download and extract the RDD2022 dataset
2) Download and extract the desired pothole dataset 
3) Download **preprocessing_1.py** or **preprocessing_2.py** file (depending on which pothole dataset you want to combine).
4) In the **preprocessing_1.py** or **preprocessing_2.py** file (depending on which pothole dataset you want to combine):
	* Specify the path to the RDD2022 dataset using the variable **"RDD_dataset_path"**.
	* Specify the path to the pothole dataset using the variable **"pothole_dataset_path"**.
5) Run the code 
6) if you choose:
	* the small pothole dataset: then your final combined dataset will be called "combined_RDD_dataset" and it will be located inside the "RDD2022" dataset folder.
	* the big pothole dataset: then your final combined dataset will be called "new_combined_RDD_dataset" and it will be located inside the "RDD2022" dataset folder.

# Apply Augmnetation
To apply augmentation to generated dataset follow these steps:
1) Run the**preprocessing_1.py** or **preprocessing_2.py** file (depending on which pothole dataset you want to combine) as explained earlier **only if you didn't run it beofre**.
2) Download the **augmentation.py** file.
3) In the **augmentation.py** file: Specify the path to the **'train'** folder - of the dataset you generated earlier - using the variable **"train_folder"**.
4) Run the code.
5) Now you have a new folder called **"augmented_train"** and the **"data.yaml"** file will point toward this folder as the folder used for training.

**Note**: The execution time of the code may be up to 30 minutes.




