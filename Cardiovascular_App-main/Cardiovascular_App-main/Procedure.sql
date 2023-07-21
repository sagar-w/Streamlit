CREATE OR REPLACE PROCEDURE HEART_DB.PUBLIC.TRAIN_AND_PREDICT()
RETURNS VARCHAR(16777216)
LANGUAGE PYTHON
RUNTIME_VERSION = '3.8'
PACKAGES = ('snowflake-snowpark-python','scikit-learn','pandas','numpy')
HANDLER = 'main'
EXECUTE AS OWNER
AS '
import pandas as pd
import numpy as nps
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main(session):

    # Load features
    pred_df = session.table(''PREDICTED_DATA'').to_pandas()

    x = pred_df.drop(''ORG_RESULT'', axis=1)
    Y = pred_df[''ORG_RESULT'']
    

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, Y, stratify = Y, test_size=0.2, random_state=2)

    model1 = LogisticRegression(max_iter=10000)
    model2 = GaussianNB()
    model3 = KNeighborsClassifier()
    model4 = DecisionTreeClassifier()
    model5 = RandomForestClassifier()

    #training model with training data
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    model3.fit(X_train, y_train)
    model4.fit(X_train, y_train)
    model5.fit(X_train, y_train)

    # input data
    input_df = session.table(''INPUT_DATA'').to_pandas()

    # Define the list of models
    models = [model1, model2, model3, model4, model5]

    # Define the list of model names
    model_names = [''LogisticRegression'', ''GaussianNB'', ''KNeighborsClassifier'', ''DecisionTreeClassifier'', ''RandomForestClassifier'']

    # Create an empty dictionary to store the predicted values for each model
    predictions = {}
    
    # Loop through each model and make predictions on the input DataFrame
    for model, name in zip(models, model_names):
        # Use the predict method of the model to generate predicted values
        y_pred = model.predict(input_df)
        # Store the predicted values in the dictionary
        predictions[name] = y_pred

    # Create a new DataFrame with the predicted values and model names
    results_df = pd.DataFrame(predictions)

    # Define empty lists to store the accuracy scores and confusion matrices
    accuracies = []

    # Loop through each model and evaluate its accuracy
    for model, name in zip(models, model_names):
        # Make predictions on the test data using the trained model
        y_pred = model.predict(X_test)

        # Compute the accuracy score
        accuracy = accuracy_score(y_test, y_pred)

        # Append the accuracy score to the lists
        accuracies.append(accuracy)

    # Create a DataFrame with the accuracy scores and model names
    accuracy_df = pd.DataFrame({''Model Name'': [name + ''_accuracy'' for name in model_names], ''Accuracy'': accuracies})


    # Reshape the data frame
    df_pivoted = accuracy_df.pivot(index=None, columns=''Model Name'', values=''Accuracy'')

    # Fill the NaN values in the data frame with the corresponding values from the other column
    df_combined = df_pivoted.fillna(method=''ffill'').fillna(method=''bfill'')
    # Drop all rows except the first one
    accuracy_df = df_combined.drop(index=df_combined.index[1:])

    # Print the combined data frame
    #accuracy_df = accuracy_df.set_index(''DecisionTreeClassifier_accuracy'', drop=True)

    # Concatenate the two DataFrames horizontally
    final_df = pd.concat([input_df, results_df], axis=1)
    final_df = pd.concat([final_df, accuracy_df], axis=1)
    # Fill the NaN values in the data frame with the corresponding values from the other column
    final_df = final_df.fillna(method=''ffill'').fillna(method=''bfill'')
    # Drop all rows except the first one
    final_df = final_df.drop(index=final_df.index[1:])

    for index, row in final_df.iterrows():
        insertStatement = f"INSERT INTO Final_Data \\
                  (AGE, SEX, CP, TRESTBPS, CHOL, FBS, RESTECG, THALACH, \\
                  EXANG, OLDPEAK, THAL, LogisticRegression, GaussianNB, \\
                  KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, \\
                  LogisticRegression_accuracy, GaussianNB_accuracy, KNeighborsClassifier_accuracy,  \\
                  DecisionTreeClassifier_accuracy, RandomForestClassifier_accuracy) \\
                  VALUES (''{row[''AGE'']}'', ''{row[''SEX'']}'', ''{row[''CP'']}'', ''{row[''TRESTBPS'']}'', \\
                  ''{row[''CHOL'']}'', ''{row[''FBS'']}'', ''{row[''RESTECG'']}'', ''{row[''THALACH'']}'', \\
                  ''{row[''EXANG'']}'', ''{row[''OLDPEAK'']}'', ''{row[''THAL'']}'', ''{row[''LogisticRegression'']}'', \\
                  ''{row[''GaussianNB'']}'', ''{row[''KNeighborsClassifier'']}'', ''{row[''DecisionTreeClassifier'']}'', \\
                  ''{row[''RandomForestClassifier'']}'', ''{row[''LogisticRegression_accuracy'']}'',''{row[''GaussianNB_accuracy'']}'', \\
                  ''{row[''KNeighborsClassifier_accuracy'']}'', ''{row[''DecisionTreeClassifier_accuracy'']}'', \\
                  ''{row[''RandomForestClassifier_accuracy'']}'')"
        session.sql(insertStatement).collect()
        
    trunc = ''TRUNCATE TABLE INPUT_DATA''
    session.sql(trunc).collect()

    #''The following features were found to be the most important in predicting heart health status:''
    feature_importance_df = pd.DataFrame({''Feature'': X_train.columns, ''Importance'': model5.feature_importances_}).sort_values(''Importance'', ascending=False)

    # store feature importance in snowflake table
    trunc2 = ''TRUNCATE TABLE Feature_Importance''
    session.sql(trunc2).collect() 
    for index, row in feature_importance_df.iterrows():
        query = f"INSERT INTO Feature_Importance (Feature, Importance) VALUES (''{row[''Feature'']}'', {row[''Importance'']})"
        session.sql(query).collect()
    return ''predicted''    
    ';