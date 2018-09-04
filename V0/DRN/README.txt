#Installation:
#Install python 3.6
#If you do not have python3.6 installed, then in Ubuntu you can run this command
sudo apt-get install python3.6-dev

#Install the necessary python libraries from the requirements.txt
pip install -r requirements.txt

#For correct operation of the polyglot library, additional packages must be downloaded. This can be done by running the script download_polyglot.py:
python download_polyglot.py

#Preprocessing
#Settings.
#Configure all algorithms in the config.py. There are basic parameter groups:
#config_processing - Сontains settings for word processing such as a list of stop words, delete numeric values and so on
#config_models - contains parameters for all semantic transformation algorithms. For each type of model you can set the number of selected topics "num_topics" and others parameters.
#config_general - in this dictionary is defined the algorithm of semantic analysis and general parameters for all algorithms.
#“model_type” - type of semantic model. The following types of models are available: "TfIdf", "LSI", "LDA", "HDP", "RP". The individual settings for each model are specified in config_models.
#config_onehot - Here are the categories of news participating in the analysis and users defined as opinion leaders
#config_dqnn - DDQN parameters
#cross_validation - cross-validation parameters

#Filling the base for offline learning
#The database is populated from the csv file

import modeling 
from config import config
model = modeling.Model(config=config)
model.add_csv_data(path_to_csv)

#Where path_to_csv is the path to the csv file or directory containing csv files.


#Semantic analysis and preprocessing of data:
#Performing the transformation of texts using semantic analysis
model.train_model()

#For further analysis, it is necessary to prepare a one-shot matrix. These data will be required to determine the user part and categorie features. To do this, you must run the following command:
model.init_onehot()

#Algoritm cross validation (this part is not working)
#The code implements a class ModelDQNN (from dqnn.py) for learning DDQN, which is built into the cross-validation class. To run cross-validation, execute the following code:
cross_validation = cross_validation.CrossValidationDQNN(config=config)
 estimate = cross_validation.run()
