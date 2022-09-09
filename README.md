#  Eye-Tracking
The repository contains the Python code for our project “Early Identification of Autism Spectrum Disorder based on Machine Learning with Eye-tracking Data ”.

## Requirements
The code mainly relies on [scikit-learn](https://scikit-learn.org/stable/), [xgboost](https://pypi.org/project/xgboost/) for training and testing machine learning models, and [NumPy](https://numpy.org/install/), [pandas](https://pandas.pydata.org/getting_started.html) for data manipulation.

You can install all the required packages using pip: `pip install -r requirements.txt`


## Optimal model selection
Run `python Optimal model selection.py` to train and test on the model building dataset. Please note that the seven datasets based on different combinations of eye tracking paradigms can be selected and compared as needed. We suggest selecting the data with three paradigms combined for model development, which will lead to the best performence. In addition, since we set up 100 repeated experiments, it may take a long time to complete this task.. Better CPU performance could effectively shorten the experimental time.

Please also note that the model parameters will change according to the actual training data, and a single training does not necessarily produce satisfactory accuracy, which is only for illustrative purposes.

## Evaluate on temporal external validation
You can evaluate the model with a temporal external validation set. 

Simply run `python Evaluate on temporal external validation.py`. The Random Forest model will automatically optimize the parameters based on the model development dataset, and then evaluate and report the AUC value on the temporal external validation set.