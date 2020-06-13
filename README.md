# QRS_detector
CPSC2019 winner's algorithm  

## Requirements
numpy  
wfdb  
scipy  
tensorflow 2.X or tensorflow 1.X + keras  

## DATA
All 12 records of NSTDB are included in ./data/nstdb.  
10 records of CPSC2019, 1 record of MITDB and 1 record of QTDB are included.  
1 record in txt file is in ./data/mydb.  

## Models
The details of the CNN and CRNN models are described in the paper listed below.  

## Usage
   * To run CPSC2019 data, run   
		`python cpsc2019_score.py`  
		
   * To evaluate the model perfomance on NSTDB/MITDB/QTDB or others, run  
		python score.py model[cnn,crnn] database[nstdb,mitdb,qtdb] write_to_file[0,1]  
		`python score.py cnn nstdb 1`  
		
   * To get predictins of QRS complex locations, run  
		python QRS_detector.py model[cnn,crnn] database[mydb(txt only)] sampling_frequency[100-10000]  
		`python QRS_detector.py cnn mydb 250`
		
_If you think this algorithm is helpful, please cite this paper as a reference:_  
Wenjie Cai, Danqin Hu, "QRS Complex Detection Using Novel Deep Learning Neural Networks", IEEE Access, vol. 8, pp. 97082-97089, 2020.
