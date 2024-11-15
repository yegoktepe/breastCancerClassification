Algorithm 1. Pseudocode for UNNEBC
Input: breast cancer dataset
Output: classification report
Load and preprocess data:
X = all columns except the label column (features)
y = target labels
Feature engineering:
X['radius_texture_diff'] = difference between the first two columns
X['feature_std_dev'] = standard deviation across all features per row
X['feature_variance'] = variance across all features per row
y = transform y into numeric format using LabelEncoder
X = scale X to have zero mean and unit variance
Initialize PCA with n_components=nc
X_pca = reduce dimensions of X using PCA
Split data into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.25, random_state=42)
Reshape X_train and X_test to fit CNN input shape
Convert y_train and y_test to categorical format for multi-class classification
Define CNN model:
Add Conv2D layer, kernel size (3, 1), ReLU activation, input shape (X_train.shape[1], 1, 1)
Add MaxPooling2D layer, Flatten layer, Dense layer, Dropout layer 
Add output Dense layer with 2 neurons, softmax activation
Define Dense model:
Add Dense layer with 128 neurons, ReLU activation, input shape (X_train.shape[1],)
Add Dropout layer, Dense layer, Dropout layer
Add output Dense layer with 2 neurons, softmax activation
Train CNN and Dense models in the first layer:
Initialize cnn_models and dense_models arrays
for each model in cnn_models and dense_models
Compile model with optimizer=Adam(learning_rate=0.0001)
Fit model on X_train, y_train with early stopping and validation split of 0.25
end for
Obtain predictions from each model in the first layer:
for each model in cnn_models
cnn_predictions = [model.predict(X_train) for model in cnn_models]
end for
for each model in dense_models
dense_predictions = [model.predict(X_train) for model in dense_models]
end for
Prepare input for the second layer (meta-learning):
stacked_train_input = concatenate cnn_predictions and dense_predictions along axis=1
Reshape stacked_train_input to fit XGBoost model input requirements
Train XGBoost model as meta-learner:
y_train_binary = convert y_train to binary class format (0 and 1)
Initialize xgb_model = XGBClassifier with parameters 
Fit xgb_model on stacked_train_input and y_train_binary
Obtain predictions for test data in the first layer:
cnn_test_predictions = [model.predict(X_test) for model in cnn_models]
dense_test_predictions = [model.predict(X_test) for model in dense_models]
stacked_test_input = concatenate cnn_test_predictions and dense_test_predictions along axis=1
Reshape stacked_test_input to fit XGBoost model input requirements
Predict on test data using XGBoost model:
y_pred = xgb_model.predict(stacked_test_input)
Print classification report:
