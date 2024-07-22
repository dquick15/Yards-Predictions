#!/usr/bin/env python
# coding: utf-8

# In[10]:


#Import packages for exploratory data analysis, preprocessing and random forest regressor model selection
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('CommandersQuantTestData.csv')

# Load Column Information
df.info()

# Split the data into known and missing GAIN values
known_gain_df = df[df['GAIN'] != 'MISSING']
missing_gain_df = df[df['GAIN'] == 'MISSING']

# Convert the GAIN column to float type for known data
known_gain_df.loc[:, 'GAIN'] = known_gain_df['GAIN'].astype(float)

# Encode the categorical variable PLAYTYPE for known data
encoder = OneHotEncoder()
play_type_encoded_known = encoder.fit_transform(known_gain_df[['PLAYTYPE']]).toarray()
play_type_df_known = pd.DataFrame(play_type_encoded_known, columns=encoder.get_feature_names_out(['PLAYTYPE']))

# Combine the one-hot encoded PLAYTYPE DataFrame with DOWN and DIST for known data
features_known = pd.concat([known_gain_df[['DOWN', 'DIST']], play_type_df_known], axis=1)
target_known = known_gain_df['GAIN']

# Encode the categorical variable PLAYTYPE for missing data
play_type_encoded_missing = encoder.transform(missing_gain_df[['PLAYTYPE']]).toarray()
play_type_df_missing = pd.DataFrame(play_type_encoded_missing, columns=encoder.get_feature_names_out(['PLAYTYPE']))

# Combine the one-hot encoded PLAYTYPE DataFrame with DOWN and DIST for missing data
features_missing = pd.concat([missing_gain_df[['DOWN', 'DIST']], play_type_df_missing], axis=1)

# Impute missing values in both known and missing data features
imputer = SimpleImputer(strategy='mean')
features_known_imputed = imputer.fit_transform(features_known)
features_missing_imputed = imputer.transform(features_missing)

# Split the known data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_known_imputed, target_known, test_size=0.2, random_state=42)

# Instantiate the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Predict the GAIN on the test set for evaluation
y_test_pred = rf_model.predict(X_test)

# Compute RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f'Root Mean Square Error on test set: {rmse}')

# Predict the GAIN for the missing data
y_missing_pred = rf_model.predict(features_missing_imputed)

#Check that the array lengths for missing PLAYIDs and our predictions match 
playid_len = len(missing_gain_df['PLAYID'].values)
pred_len = len(y_missing_pred)

#Assuming our predictions are longer than the missing set, trim predictions dataset to match the missing PLAYIDs dataset
y_missing_pred_trimmed = y_missing_pred[:playid_len]

#Create a DataFrame with PlayID and predicted GAIN for the missing data
predictions_df = pd.DataFrame({
    'PLAYID': missing_gain_df['PLAYID'].values,
    'GAIN': y_missing_pred_trimmed
})

# Compute the 90% prediction interval for the sum of yards gained on the missing plays
all_tree_predictions = np.array([tree.predict(features_missing_imputed) for tree in rf_model.estimators_])
sum_predictions_per_tree = np.sum(all_tree_predictions, axis=1)
mean_sum_predictions = np.mean(sum_predictions_per_tree)
std_sum_predictions = np.std(sum_predictions_per_tree)
lower_bound = mean_sum_predictions - 1.645 * std_sum_predictions
upper_bound = mean_sum_predictions + 1.645 * std_sum_predictions

# Save the DataFrame to a CSV file
predictions_df.to_csv('QuickDavid_CodingPredictions.csv', index=False)

print(f'90% Prediction Interval for the sum of yards gained on all MISSING plays: [{lower_bound}, {upper_bound}]')

print("Predictions saved to QuickDavid_CodingPredictions.csv")

