import os
import cv2
import numpy as np
from sklearn import preprocessing
from progress.bar import Bar
from sklearn.cluster import MiniBatchKMeans
import time

def main():
    mainStartTime = time.time()
    trainImagePath = 'classificacao/images_split/train/'
    testImagePath = 'classificacao/images_split/val/'
    trainFeaturePath = 'classificacao/features_labels/orb/train/'
    testFeaturePath = 'classificacao/features_labels/orb/test/'
    
    print(f'[INFO] ========= TRAINING IMAGES ========= ')
    trainImages, trainLabels = getData(trainImagePath)
    trainEncodedLabels, encoderClasses = encodeLabels(trainLabels)
    trainOrbDescriptors = extractOrbDescriptors(trainImages)
    kmeans, k = trainKMeans(trainOrbDescriptors)
    trainFeatures = buildHistogram(trainOrbDescriptors, kmeans, k)
    saveData(trainFeaturePath, trainEncodedLabels, trainFeatures, encoderClasses)
    
    print(f'[INFO] =========== TEST IMAGES =========== ')
    testImages, testLabels = getData(testImagePath)
    testEncodedLabels, encoderClasses = encodeLabels(testLabels)
    testOrbDescriptors = extractOrbDescriptors(testImages)
    testFeatures = buildHistogram(testOrbDescriptors, kmeans, k)
    saveData(testFeaturePath, testEncodedLabels, testFeatures, encoderClasses)
    
    elapsedTime = round(time.time() - mainStartTime, 2)
    print(f'[INFO] Code execution time: {elapsedTime}s')

def getData(path):
    images = []
    labels = []
    
    if os.path.exists(path):
        for dirpath, dirnames, filenames in os.walk(path):
            if len(filenames) > 0:  # it's inside a folder with files
                folder_name = os.path.basename(dirpath)
                bar = Bar(f'[INFO] Getting images and labels from {folder_name}', max=len(filenames), suffix='%(index)d/%(max)d Duration:%(elapsed)ds')
                
                for index, file in enumerate(filenames):
                    label = folder_name
                    labels.append(label)
                    full_path = os.path.join(dirpath, file)
                    image = cv2.imread(full_path)
                    images.append(image)
                    bar.next()
                
                bar.finish()
    
    return images, np.array(labels, dtype=object)

def extractOrbDescriptors(images):
    orbDescriptorsList = []
    bar = Bar('[INFO] Extracting ORB descriptors...', max=len(images), suffix='%(index)d/%(max)d  Duration:%(elapsed)ds')
    orb = cv2.ORB.create()
    
    for image in images:
        if len(image.shape) > 2:  # color image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(image, None)
        orbDescriptorsList.append(descriptors)
        bar.next()
    
    bar.finish()
    return np.array(orbDescriptorsList, dtype=object)

def trainKMeans(orbDescriptors):
    print('[INFO] Clustering the ORB descriptors of all train images...')
    k = 100  # number of clusters for KMeans
    kmeans = MiniBatchKMeans(n_clusters=k, n_init='auto', random_state=42)
    
    startTime = time.time()
    kmeans.fit(np.vstack(orbDescriptors))
    elapsedTime = round(time.time() - startTime, 2)
    
    print(f'[INFO] Clustering done in {elapsedTime}s')
    return kmeans, k

def buildHistogram(orbDescriptors, kmeans_model, n_clusters):
    print('[INFO] Building histograms...')
    startTime = time.time()
    histogramList = []
    
    for i in range(len(orbDescriptors)):
        histogram = np.zeros(n_clusters)
        idx_arr = kmeans_model.predict(orbDescriptors[i])
        
        for d in range(len(idx_arr)):
            histogram[idx_arr[d]] += 1 
        
        histogramList.append(histogram)
    
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Histogram built in {elapsedTime}s')
    return np.array(histogramList, dtype=object)

def encodeLabels(labels):
    startTime = time.time()
    print(f'[INFO] Encoding labels to numerical labels')
    encoder = preprocessing.LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Encoding done in {elapsedTime}s')
    return np.array(encoded_labels, dtype=object), encoder.classes_

def saveData(path, labels, features, encoderClasses):
    startTime = time.time()
    print(f'[INFO] Saving data')
    
    label_filename = f'{labels=}'.split('=')[0] + '.csv'
    feature_filename = f'{features=}'.split('=')[0] + '.csv'
    encoder_filename = f'{encoderClasses=}'.split('=')[0] + '.csv'
    
    np.savetxt(path + label_filename, labels, delimiter=',', fmt='%i')
    np.savetxt(path + feature_filename, features, delimiter=',') 
    np.savetxt(path + encoder_filename, encoderClasses, delimiter=',', fmt='%s')
    
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Saving done in {elapsedTime}s')

if __name__ == "__main__":
    main()
