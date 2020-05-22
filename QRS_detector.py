import numpy as np
import wfdb
from scipy.signal import resample
from scipy import io
from decision import *
import os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
try:
    from keras.models import model_from_json
except:
    from tensorflow.keras.models import model_from_json

'''
Written by: Wenjie Cai
            School of Medical Instrument and Food Engineering
            University of Shanghai for Science and Technology, China
            wjcai@usst.edu.cn
'''

def pp(data):   #preprocessing
    x = np.max(data)
    if x>20:
        b = np.argwhere(data>20)
        for k in b[:,0]:
            if k>0 and data[k]-data[k-1]>20:
                data[k] = data[k-1]
    return data

def load_data(data_path):
    db = []
    filenames = []
    ecg_files = os.listdir(data_path)
    for file in ecg_files:
        ecg_data = np.loadtxt(data_path+file)
        db.append(pp(ecg_data))
        filenames.append(file)
    print('%d records are found.'%len(db))
    return db,filenames
        
def resamp(db,fs):
    data = []
    for i in range(len(db)):
        if fs!=500:
            samp = db[0]
            ln = len(samp)//fs
            remain = len(samp)%fs
            new = resample(db[i][:ln*fs],ln*500)
            if remain>1:
                rem = resample(db[i][ln*fs:],int(remain/fs*500))
                new = np.concatenate((new,rem))
        else:
            new = db[i]
        mean = np.mean(new)
        data.append(new-mean)
    print('Date were resampled at 500 Hz.\n')
    return data

def load_model(modelname):
    print('loading the deep learning model')
    if modelname == 'crnn':
        model = model_from_json(open('models/CRNN.json').read())
        model.load_weights('models/CRNN.h5')
        print('\nCRNN model is loaded.\n')
    else:
        model = model_from_json(open('models/CNN.json').read())
        model.load_weights('models/CNN.h5')
        print('\nCNN model is loaded.\n')
        modelname = 'cnn'
    return model

def main(argv):
    if len(argv)!=3:
        print('Wrong input')
        exit()
    modelname = argv[0].lower()
    database = argv[1].lower()
    if int(argv[2])>100 and int(argv[2])<10001:
        fs = int(argv[2])
    else:
        print('Please check the sampling frequency of your records.')
        print('Normal it is more than 100 Hz and less than 10001 Hz.')
        exit()
    write_to_file = True
    
    db,filenames = load_data('./data/'+database+'/')
    data = resamp(db,fs)
    model = load_model(modelname)
   
    print('\nPredicting QRS complexes with',modelname,'model')
    print('\nThe results will be saved to ./output/')
    pf = performance(data,None,model,write_to_file,filenames,fs)
    print('\nThe results have been saved to ./output/')

if __name__ == "__main__":
   main(sys.argv[1:])