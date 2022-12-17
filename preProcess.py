import librosa
import json
import os
import math
import numpy as np
import pandas as pd

DATASET_PATH = 'dataset'
JSON_PATH = "deepDasta.json"
CSV_PATH = "deepData.csv"
SAMPLE_RATE = 22050
TRACK_DURATION = 15 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.
        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "className": [],
        "labelNumber": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    
    print("\nProcessing:")
   
    for root, dirs, files in os.walk((dataset_path), topdown=False):
        for name in files:
            print(os.path.join(root, name))
        for name in dirs:
            print(os.path.join(root, name))
    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        print("\nProcessing:")
        
        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:

		# load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # process all segments of audio file
                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T
                    #print(mfcc)
                    #mfccs_scaled_features = np.mean(mfcc,axis=0)
                    #print(mfcc)
                    # using mean is not right way hence reducing the duration !!!
                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["className"].append(semantic_label)
                        data["mfcc"].append(mfcc.tolist())
                        data["labelNumber"].append(i-1)
                        print("{}, segment:{}".format(file_path, d+1))
                    

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

def jsonToCsv(jsonPath,csvPath):
    with open(jsonPath, encoding='utf-8') as inputfile:
        df = pd.read_json(inputfile)
        print("no of mfcc:"+(str(len(df["mfcc"][0]))))
        count = 0
        for itr in df["mfcc"]:
            count = count+1
        print(count)

    df.to_csv(csvPath, encoding='utf-8', index=False)
    print("csv file write")


        
if __name__ == "__main__":
    print(DATASET_PATH)
    print(JSON_PATH)
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
    jsonToCsv(JSON_PATH,CSV_PATH)
    
    
    
    
    

    
    
    
    
    
    