# Arrhythmia Diagnois
Diagnosis approach uses 10-seconed ECG recording excerpts and additional patient and recording informations for classification.
Classes are arrythmia types and this approach has more potential for real life applications in medicine.
The database used was created under the auspices of Chapman University and Shaoxing People’s Hospital (available at: https://figshare.com/collections/ChapmanECG/4560497/2).

Additional libraries used in this subproject:
- pandas

Runner module contains the options for running this subproject in main function (train, evaluate and demo).

To choose a model to use for evaluation and demo, model_name property in path section should be set to a choosen model name. Models and their training logs are available in state dict folder. The architecture of the RNN model needs to be changed in models module to match the one described in the training log of a model. Architecture is set to fit the best model, and nothing needs to be changed for the demo mode to work.
