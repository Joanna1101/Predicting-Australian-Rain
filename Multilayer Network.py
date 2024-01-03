# Libraries #################################################################################################################################################
#############################################################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from sklearn.linear_model import LassoCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Torch
import torch
import torch.nn as nn 
import torch.optim as optim




## Exploratory Analysis #####################################################################################################################################
#############################################################################################################################################################
# Loading the dataset
Rain = pd.read_csv('weatherAUS.csv')
print(f"Data: \n{Rain.columns}\n")

# Dropping rows with NA in 'RainTomorrow' (only 2% of total data)
# We don't want the model to be able to predict 'NA' for RainTomorrow 
Rain.dropna(subset = ['RainTomorrow'], inplace = True)

# Splitting into data and target
X = Rain.loc[:, Rain.columns != "RainTomorrow"]
y = Rain[['RainTomorrow']]
print(f'Features: \n{X.columns}\n')
print(f'Response: \n{y.columns}\n')

# Which columns have missing values
sad_cols = Rain.columns[Rain.isnull().any()]
happy_cols = Rain.columns[Rain.notna().all()]
print(f'No Missing Values: \n{happy_cols}\n')

# Which columns are categorical
categorical = Rain.select_dtypes(exclude = 'number').columns
print(f'Categorical Columns (need encoding):\n{categorical}\n')

# Categorical columns with missing data
c = [col for col in categorical if col not in happy_cols]

# Quantitative columns with missing data
q = [col for col in X if (col not in happy_cols) and (col not in categorical)]




## Date #####################################################################################################################################################
#############################################################################################################################################################
# Converting date into datetime format
X = X.copy()
X['Date'] = pd.to_datetime(X['Date'])

# Separating date into year, month, day
X['Year'] = X['Date'].dt.year
X['Month'] = X['Date'].dt.month
X['Day'] = X['Date'].dt.day

# Periodic encoding, using both sin and cos 
X['Year_sin'] = np.sin(2 * np.pi * X['Year'] / 12)
X['Year_cos'] = np.cos(2 * np.pi * X['Year'] / 12)
X['Month_sin'] = np.sin(2 * np.pi * X['Month'] / 12)
X['Month_cos'] = np.cos(2 * np.pi * X['Month'] / 12)
X['Day_sin'] = np.sin(2 * np.pi * X['Day'] / 12)
X['Day_cos'] = np.cos(2 * np.pi * X['Day'] / 12)

# Dropping extra columns
X = X.drop(['Date', 'Year', 'Month', 'Day'], axis = 1)
print(f"New columns:\n{X.columns}\n")




## Missing Values ###########################################################################################################################################
#############################################################################################################################################################
# Columns to fill, created in previous cell
print(f"Quantitative features with missing values: \n{q}\n")
print(f"Categorical features with missing values: \n{c}\n")

# Initial idea: estimate parameters based on distribution of data
# Issue: cyclical time-series data needs to be restructured 
# Temporary fix: fill in NA values with mean

# def gauss(x, mu, sigma, c, a):
#     topfrac = (-1) * (x - mu)**2
#     bottomfrac = 2 * (sigma**2)
#     ex = np.exp(topfrac / bottomfrac)
#     y = (a * ex) + c
#     return y

# fig, ax = plt.subplots(len(q), 1, figsize =(10, 5 * len(q)))

# Iterating through quantitative columns
for i, col in enumerate(q):
    X_notNA = X[col].dropna()

    # Estimating parameters
    mu_guess = X_notNA.mean()
    sigma_guess = X_notNA.std()
    c_guess = 0
    a_guess = 1

    # Assigning mean to NAs
    X[col] = X[col].fillna(mu_guess)
    X_NA = X[X[col] == mu_guess]

    # Curve fitting: learning parameters of this feature's distribution
    # This does not work
    # popt, _ = curve_fit(gauss, 
    #                     np.arange(len(X_notNA)), 
    #                     X_notNA,
    #                     p0 = [mu_guess, sigma_guess, c_guess, a_guess])
    
    # Visualizing distributions and missing values
    # ax[i].scatter(np.arange(len(X_notNA)), X_notNA, s = 1, color = 'lightseagreen', label = "Original Data")
    # ax[i].scatter(X_NA.index, X_NA[col], s = 1, color = 'violet', label = "Mean Value")
    # ax[i].set_xlabel('Time')
    # ax[i].set_ylabel(col)
    # ax[i].set_title(col)
    # ax[i].legend()

# plt.tight_layout()
# plt.show()

# Categorical features: randomly select a value 
# This adds noise, but hopefully it is noise the network can ignore
for col in c:
    possible_observations = X[col].dropna().unique().tolist()
    X[col] = X[col].apply(lambda x: np.random.choice(possible_observations) if pd.isna(x) else x)

# Double checking for NA values
emptycols = X.columns[X.isnull().any()]
print(f"Still Missing Values: {emptycols}")




## Encoding, Scaling ########################################################################################################################################
#############################################################################################################################################################
# Encoding binary variables
y = y.copy()
X = X.copy()
y['RainTomorrow10'] = y['RainTomorrow'].map({'Yes': 1, 'No': 0})
X['RainToday10'] = X['RainToday'].map({'Yes': 1, 'No': 0})

# Dropping original columns, renaming
y = y.drop(['RainTomorrow'], axis = 1)
X = X.drop(['RainToday'], axis = 1)
y.rename(columns = {'RainTomorrow10': 'RainTomorrow'}, inplace = True)
X.rename(columns = {'RainToday10': 'RainToday'}, inplace = True)

# Checking that those columns look right
# (They do, output is long)
# print(f"RainTomorrow: \n{y['RainTomorrow'].head(20)}")
# print(f"RainToday: \n{X['RainToday'].head(20)}")

# Encoding other categorical variables
cat_vars = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']

# I tried using OneHotEncoder but had *a lot* of data shape issues
# Using label encoder from sklearn instead
label_encoder = LabelEncoder()

X['Encoded_Location'] = label_encoder.fit_transform(X['Location'])
X['Encoded_WindGustDir'] = label_encoder.fit_transform(X['WindGustDir'])
X['Encoded_WindDir9am'] = label_encoder.fit_transform(X['WindDir9am'])
X['Encoded_WindDir3pm'] = label_encoder.fit_transform(X['WindDir3pm'])

# Dropping original columns, renaming
X = X.drop(['Location'], axis = 1)
X = X.drop(['WindGustDir'], axis = 1)
X = X.drop(['WindDir9am'], axis = 1)
X = X.drop(['WindDir3pm'], axis = 1)
X.rename(columns = {'Encoded_Location': 'Location'}, inplace = True)
X.rename(columns = {'Encoded_WindGustDir': 'WindGustDir'}, inplace = True)
X.rename(columns = {'Encoded_WindDir9am': 'WindDir9am'}, inplace = True)
X.rename(columns = {'Encoded_WindDir3pm': 'WindDir3pm'}, inplace = True)




## Additional Columns #######################################################################################################################################
#############################################################################################################################################################
# Capturing temporal aspect and temperature/humidity/evaporation relationship
X['Rain_Yesterday'] = X['Rainfall'].shift(1)
X['Avg_Rain_Last_Week'] = X['Rainfall'].rolling(window = 7).mean()
X['Max_Temp_Last_Month'] = X['MaxTemp'].rolling(window = 30).mean()
X['Min_Temp_Last_Month'] = X['MinTemp'].rolling(window = 30).mean()
X['temp_humidity_evap_9am'] = X['Temp9am'] * X['Humidity9am'] * X['Evaporation']
X['temp_humidity_evap_3pm'] = X['Temp3pm'] * X['Humidity3pm'] * X['Evaporation']
X['temp_humidity'] = X['Temp9am'] * X['Temp3pm'] * X['Humidity9am'] * X['Humidity3pm']

# Taking care of NaNs
X['Rain_Yesterday'].fillna(-1, inplace = True)
X['Avg_Rain_Last_Week'].fillna(-1, inplace = True)
X['Max_Temp_Last_Month'].fillna(-1, inplace = True)
X['Min_Temp_Last_Month'].fillna(-1, inplace = True)

# Scaling
scaler = StandardScaler()
normed_X = scaler.fit_transform(X)

#80-20 Train Test Split
X_train, X_test, y_train, y_test = train_test_split(normed_X, y, test_size = 0.2, random_state = 42)




## Feature Analysis #########################################################################################################################################
#############################################################################################################################################################
# Lasso Regression Model: 5 fold cross validation
lasso = LassoCV(cv = 5)
lasso.fit(X_train, y_train.values.ravel())

# Features
good_features = X.columns[abs(lasso.coef_) >= 0.01]
print(f'Most Important Features: {good_features}')
print(f'Coefficients: {lasso.coef_[lasso.coef_ >= 0.01]}')

# Visualization with correlation matrix
good_features = list(good_features)
good_features.append('RainTomorrow')
full = pd.concat([X, y], axis = 1)
correlation_matrix = full[good_features].corr()

plt.figure(figsize = (12, 8))
graph = plt.matshow(correlation_matrix, cmap = "cool")
plt.colorbar(graph)

plt.xticks(range(len(good_features)), good_features, rotation = 90, ha = 'left')
plt.yticks(range(len(good_features)), good_features)

plt.title("Correlation Matrix of Selected Features")
plt.show()




## Model Building ###########################################################################################################################################
#############################################################################################################################################################
class MultiLayerNN(nn.Module):
    ''' 
    Fully connected neural network
    Two hidden layers with tanh activation function
    Sigmoid activation function for output
    '''
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MultiLayerNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.tanh1 = nn.Tanh()

        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.tanh2 = nn.Tanh()

        self.layer3 = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.tanh1(self.layer1(x))
        x = self.tanh2(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x 

# Dimensions
input_size = normed_X.shape[1]
hidden_size1 = 64
hidden_size2 = 32

# Binary classification
output_size = 1

# Model instantiation
multi = MultiLayerNN(input_size, hidden_size1, hidden_size2, output_size)




## Model Training ###########################################################################################################################################
#############################################################################################################################################################
# Binary Cross Entropy loss function
criterion = nn.BCELoss()
optimizer = optim.Adam(multi.parameters(), lr = 0.001)

# Converting data to tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)

# Training
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = multi(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass, optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# Predictions without any hyperparameter tuning (not too bad!)
with torch.no_grad():
    multi.eval()
    predictions = multi(X_test_tensor)
    predictions_binary = (predictions >= 0.5).float()

    accuracy = torch.sum(predictions_binary == y_test_tensor).item() / len(y_test_tensor)
    print(f'Accuracy on the test set: {accuracy}')



## Cross Entropy Validation #################################################################################################################################
#############################################################################################################################################################
## Cross Entropy Validation #################################################################################################################################
# Hyperparameter Search Space
param_grid = {
    'hidden_size1': [32, 64, 128],
    'hidden_size2': [16, 32, 64],
    'learning_rate': [0.001, 0.01, 0.1],
}

# Grid to list of dicts
param_combos = list(ParameterGrid(param_grid))

# K-Fold cross-validation
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Results
results = []

# Iterating over hyperparameter combinations
for params in param_combos:
    print(f"Testing Hyperparameters: {params}")
    
    # Initialization with current hyperparameters
    model = MultiLayerNN(input_size, params['hidden_size1'], params['hidden_size2'], output_size)

    # Binary Cross Entropy loss function
    criterion = nn.BCELoss()
    
    # Adam optimizer with current learning rate
    optimizer = optim.Adam(model.parameters(), lr = params['learning_rate'])

    # Storing accuracy for each fold
    fold_accuracies = []

    # Iterating over folds
    for train_index, val_index in kf.split(normed_X):
        X_train_fold, X_val_fold = normed_X[train_index], normed_X[val_index]
        y_train_fold, y_val_fold = y.values[train_index], y.values[val_index]

        X_train_fold_tensor = torch.FloatTensor(X_train_fold)
        y_train_fold_tensor = torch.FloatTensor(y_train_fold).view(-1, 1)
        X_val_fold_tensor = torch.FloatTensor(X_val_fold)
        y_val_fold_tensor = torch.FloatTensor(y_val_fold).view(-1, 1)

        # Training
        num_epochs = 50
        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(X_train_fold_tensor)
            loss = criterion(outputs, y_train_fold_tensor)

            # Backward pass, optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluating on validation set
        with torch.no_grad():
            model.eval()
            predictions = model(X_val_fold_tensor)
            predictions_binary = (predictions >= 0.5).float()
            accuracy = torch.sum(predictions_binary == y_val_fold_tensor).item() / len(y_val_fold_tensor)
            fold_accuracies.append(accuracy)
    
    #  Printing accuracy
    print(f'Accuracy: {accuracy}\n')

    # Average accuracy across folds
    avg_accuracy = np.mean(fold_accuracies)
    results.append({
        'params': params,
        'avg_accuracy': avg_accuracy
    })

# Finding the best hyperparameters
best_result = max(results, key=lambda x: x['avg_accuracy'])
best_params = best_result['params']
print(f"Best Hyperparameters: {best_params}")
 



## Optimal Model ############################################################################################################################################
#############################################################################################################################################################
# Training a final version of the model with optimal hyperparameters
multi_optimal = MultiLayerNN(input_size, best_params['hidden_size1'], best_params['hidden_size2'], output_size)

criterion = nn.BCELoss()
optimizer = optim.Adam(multi_optimal.parameters(), lr = best_params['learning_rate'])

num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_test_tensor)
    loss = criterion(outputs, y_test_tensor)

    # Backward pass, optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# Final predictions
with torch.no_grad():
    multi.eval()
    predictions = multi(X_test_tensor)
    predictions_binary = (predictions >= 0.5).float()

    accuracy = torch.sum(predictions_binary == y_test_tensor).item() / len(y_test_tensor)
    print(f'Accuracy on the test set: {accuracy}')




## Performance Evaluation ###################################################################################################################################
#############################################################################################################################################################
# Accuracy
acc = torch.sum(predictions_binary == y_test_tensor).item() / len(y_test_tensor)

# Precision
prec = precision_score(y_test_tensor.numpy(), predictions_binary.numpy())

# Recall
recall = recall_score(y_test, predictions_binary)

# F1 Score
f1 = f1_score(y_test, predictions_binary)

# Printing scores
print(f'Accuracy: {acc}')
print(f'Precision: {prec}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}\n')

# Confusion matrix
conf_matrix = confusion_matrix(y_test, predictions_binary)
print(f'Confusion Matrix:\n{conf_matrix}')
