# COVID DATASET:  Feature extraction and classification of images with ORB algorithm and SVM Classifier 

## ORB (Oriented FAST and Rotated BRIEF)
It combines the FAST (Features from Accelerated Segment Test) corner detector to locate keypoints and the BRIEF (Binary Robust Independent Elementary Features) descriptor to describe the region around these points, introducing orientation to handle rotations. ORB is notable for being computationally efficient, robust to noise, and invariant to rotations, making it suitable for real-time applications and resource-constrained devices, in addition to being free and patent-unencumbered.

### Description
This project it's part of my study about image processing  and computer vision using Python, it focus in a classification analisys using ORB for a dataset with 280 x-ray lungs images divided by people infected with COVID-19 and people who wasn't infected. It was developed under the guide of professor Claiton de Oliveira of UTFPR in the Image Processing discipline.  

### Enviroment:
- Python (3.12.4)
- Visual Studio Code
- Pyenv-win


## Results
- Accuracy: 96.96%
- Confusion Matrix
    <img src="" alt="Description of Image">

## How to use

1. ### Clone project
    ```bash
        git clone https://github.com/Henrique-BL/COVID-DATASET-ORB.git
    ```
2. ### Install dependencies
    ```bash
        pip install -r requirements.txt
    ```
    #### In case of failure:
    ```bash
        pip install numpy progress scikit-learn progress split-folders opencv-python matplotlib
    ```

3. ### Download images dataset
    Access: [Kaggle -  Covid dataset](https://www.kaggle.com/datasets/tarandeep97/covid19-normal-posteroanteriorpa-xrays)
    Extract the files "covid" and "normal" to /images_dataset

4. ### Execute Makefile commands
    ```bash
        make split-images
        make extract
        make classify
    ```    
    After this steps you'll have the features extracted and can dump all images files, they'll no longer be needed.

    ```bash
        make classify
    ```
    Finally, this last command will generate the result graphic with the confusion matrix, datetime, and accuracy of the test.
    It'll all be saved under /resuts.


### AUTOR
Henrique Bezerra Lucas