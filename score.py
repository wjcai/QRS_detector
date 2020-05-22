import numpy as np
import wfdb
from scipy.signal import resample
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

def load_data(database):
    real_r = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']    
    with open('./data/'+database+'/RECORDS','r') as f:
        lines = f.readlines()
    db = []
    db_r = []
    name = []
    print('Reading data from '+database)
    for line in lines:
        fname = line.strip()
        name.append(fname)
        sample = wfdb.rdsamp('./data/'+database+'/'+fname)
        ann = wfdb.rdann('./data/'+database+'/'+fname,'atr')
        beats = ann.sample[np.isin(ann.symbol,real_r)]
        db.append(sample)
        db_r.append(beats)
    fs = sample[1]['fs']
    print('%d records are found.'%len(db))
    return db,db_r,name,fs
        
def resamp(db,db_r,fs):
    data = []
    ref = []
    for i in range(len(db)):
        samp = db[0][0][:,0]
        ln = len(samp)//fs
        remain = len(samp)%fs
        new = resample(db[i][0][:ln*fs,0],ln*500)
        if remain>1:
            rem = resample(db[i][0][ln*fs:,0],int(remain/fs*500))
            new = np.concatenate((new,rem))
        mean = np.mean(new)
        data.append(new-mean)
        new_r = (db_r[i]/fs*500).astype(int)
        ref.append(new_r)
    print('Date were resampled at 500 Hz from %d Hz.\n'%fs)
    return data,ref

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
    
def score(data,ref,model,write_to_file,name,fs):
    pf = performance(data,ref,model,write_to_file,name,fs)
    ppr = pf[0]/(pf[0]+pf[1])
    se = pf[0]/(pf[0]+pf[2])
    er = (pf[1]+pf[2])/(pf[0]+pf[1]+pf[2])
    f1 = 2*ppr*se/(se+ppr)
    print('\n%9s %10s %10s %10s %10s'%(' ','Se','Ppr','Er','F1'))
    print('%10s %10.4f %10.4f %10.4f %10.4f'%('Total',se,ppr,er,f1))
    if write_to_file:
        with open('./output/results.txt','a') as f:
            f.write('\n%10s %11s %11s %11s %11s'%(' ','Se','Ppr','Er','F1'))
            f.write('\n%10s %9.4f %9.4f %9.4f %9.4f'%('Total',se,ppr,er,f1))
        print('\nThe results have been saved to ./output/results.txt')

def main(argv):
    if len(argv)!=3:
        print('Wrong input')
        exit()
    modelname = argv[0].lower()
    database = argv[1].lower()
    if database not in ['mitdb','qtdb','nstdb']:
        print('%s is not for cross-database testing. Test with mitdb instead.'%database)
        database = 'mitdb'
    if argv[2]:
        write_to_file = True
    else:
        write_to_file = False
    
    model = load_model(modelname)
    db,db_r,name,fs = load_data(database)
    print('\nResampling '+database+' data')
    data,ref = resamp(db,db_r,fs)

    print('\nPredicting QRS complexes with',modelname,'model')
    print('\n%10s %10s %10s %10s'%('Record','TP','FP','FN'))
    if write_to_file:
        with open('./output/results.txt','a') as f:
            f.write('\n\nThe performance of %s model on %s'%(modelname,database))
            f.write('\n%10s %10s %10s %10s\n'%('Record','TP','FP','FN'))
    score(data,ref,model,write_to_file,name,fs)

if __name__ == "__main__":
   main(sys.argv[1:])