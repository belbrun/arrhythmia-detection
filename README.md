# Arrythmia Detection

Detection approach consists of beat-wise classification. Classes represent irregularities in a single beat.
MIT-BIH Arrythmia Database is used as a standard in the field for this approach (available at: https://physionet.org/content/mitdb/1.0.0/). 

The database consists of 48 ECG recordings, each 30 minutes long. Peaks of R waves are labeled, and those labels are used to split the recording into beats.
A simple but effective algorithm for beat splitting is proposed and its implementation can be found in the data module.

Additional libraries used for this subroject include:
- wfdb
- pywt

for loading and denoising the ECG signal.

Runner module contains the options for running this subproject in main function (train, evaluate and demo).

To choose a model to use for evaluation and demo, model_name property in path section should be set to a choosen model name.
Models and their training logs are available in state dict folder. The architecture of the RNN model needs to be changed in models module to match the one described in the training log of a model. 
Architecture is set to fit the best model, and nothing needs to be changed for the demo mode to work. 
