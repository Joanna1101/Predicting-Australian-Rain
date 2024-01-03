# Predicting-Australian-Rain
CSCI 4622 Final Project

## Data Exploration and Preprocessing
- Periodic encoding of date values
- Handling missing values and outliers
- Label encoding categorical variables
- Adding temperature and time dependence
- Normalization
- Test/train split

## Feature Analysis
- 5-fold cross-validation with lasso regression

## Model Building
- Fully connected multilayer neural network (PyTorch)
- Two hidden layers with tanh activation function
- Sigmoid activation function for output

## Model Training and Validation
- Predictions without hyperparameter tuning: about 85% accuracy on test set
- Cross Entropy Validation of layer sizes and learning rate -- also about 85% accuracy on test set

## Model Evaluation
- Accuracy: 0.8506
- Precision: 0.7219
- Recall: 0.5367
- F1: 0.6156

## Further Work
- Better missing values handling using the periodicity of the data
- Testing more hidden layers
- Testing ReLU activation function instead of tanh
