#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Load the data

# In[2]:


#load the data

df=pd.read_csv('illness_3.csv')
df=pd.DataFrame(df)


# # Explore data

# In[3]:


#print(df)

df.head()  #print 1st 5 rows


# In[4]:


df.info()    #print summary of dataframe


# In[5]:


df.describe()   #print summary statistics of dataframe


# In[6]:


df.shape   #prints shape of dataframe


# In[7]:


df.columns.tolist()  #to print all the columns as a list


# In[8]:


print(df['Illness'])    #printing only the 1st column values.Shows the list of the self diagonisable diseases


# # Handle Missing Values

# In[9]:


print(df.isnull().sum())


# In[10]:


# Drop rows with missing values
#df = df.dropna()


# # Handle Duplicates

# In[11]:


print(df.duplicated().sum())   #print if any duplicate values present 


# In[12]:


# Identify and handle outliers 
#z_scores = np.abs(stats.zscore(df.select_dtypes(include=np.number)))
#df = df[(z_scores < 3).all(axis=1)]
#df.shape


# # Data Visualization

# In[13]:


# Iterate over columns
for column in df.columns:
    # Count occurrences of "yes" in each column
    value_counts = df[column].value_counts()

    # Print the values
    print(f'Column: {column}')
    print(value_counts)
    print('-' * 30)    # used to print a line of dashes as a separator


# # Steps before training and testing different models

# In[14]:


# Encoding the specified columns
#Converting categorical labels into numerical labels.replacing 'No' with 0 and 'Yes' with 1

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
for col in df.columns:
    df[col] = label_encoder.fit_transform(df[col])

print(df)


# In[15]:


#feature extraction of x-features and y -labels [of train dataset]
y_train = df.iloc[:, 0]   # Labels (first column)
x_train = df.iloc[:, 1:]  # Features (all columns except the first one)


# In[16]:


#loading testing dataset
test_df=pd.read_csv('test_dataset.csv')


# In[17]:


#feature extraction of x-features and y -labels [of test data]
y_test = df.iloc[:, 0]   # Labels (first column)
x_test = df.iloc[:, 1:]  # Features (all columns except the first one)


# # Desicison Tree

# In[18]:


#importing descision tree models

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,f1_score


# In[19]:


dt_classifier = DecisionTreeClassifier(
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=2,
    random_state=2,
    criterion='entropy'
)


# In[20]:


# Train the model on the training set
dt_classifier.fit(x_train, y_train)

# Test the model on the testing set
y_test_pred1 = dt_classifier.predict(x_test)

y_train_pred1 = dt_classifier.predict(x_train)

# Evaluate training set accuracy
accuracy_train1 = accuracy_score(y_train, y_train_pred1)
print(f'Accuracy on the training set: {accuracy_train1 * 100:.2f}%')

# Evaluate testing set accuracy
accuracy_test1 = accuracy_score(y_test, y_test_pred1)
print(f'Accuracy on the testing set: {accuracy_test1 * 100:.2f}%')

# Additional Evaluation Metrics
print('\nClassification Report for Testing Set:')
print(classification_report(y_test, y_test_pred1))

print('\nConfusion Matrix for Testing Set:')
print(confusion_matrix(y_test, y_test_pred1))


# # K Nearest Neighbour(KNN)

# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[22]:


# Initialize the K-Nearest Neighbors Classifier 
knn_classifier = KNeighborsClassifier(n_neighbors=3,weights='distance',algorithm='kd_tree',p=1)

#weights= 'uniform' or 'distance'
#algorithm= 'auto', 'ball_tree', 'kd_tree', or 'brute'
#p=power parameter for the Minkowski distance.When p=1 ->Manhattan distance,p=2 ->Euclidean distance 


# In[23]:


# Train the model on the training set
knn_classifier.fit(x_train, y_train)


# In[24]:


#function ensures that the memory layout is C-contiguous

x_train2 = np.ascontiguousarray(x_train)
x_test2 = np.ascontiguousarray(x_test)


# In[25]:


# Test the model on the testing set
y_test_pred2 = knn_classifier.predict(x_test2)

y_train_pred2 = knn_classifier.predict(x_train2)

# Evaluate training set accuracy
accuracy_train2 = accuracy_score(y_train, y_train_pred2)
print(f'Accuracy on the training set: {accuracy_train2 * 100:.2f}%')

# Evaluate the model on the testing set
accuracy_test2 = accuracy_score(y_test, y_test_pred2)
print(f'Accuracy on the test set: {accuracy_test2*100 :.2f}%')

# Additional Evaluation Metrics
print('\nClassification Report:')
print(classification_report(y_test, y_test_pred2))

print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_test_pred2))


# # Logistic Regression

# In[26]:


from sklearn.linear_model import LogisticRegression


# In[27]:


# Initializing the Logistic Regression model
logreg_model = LogisticRegression(C=0.1,solver='liblinear',max_iter=10)

#penalty='l1' or 'l2 or 'none' regularization.It  prevents overfitting
#C= Inverse of regularization strength. Smaller values specify stronger regularization
#max_iter= Maximum number of iterations for the solver to converge. Increase it if you get a convergence warning.


# In[28]:


# Training the model
logreg_model.fit(x_train, y_train)


# In[29]:


# Test the model on the testing set
y_test_pred3 = knn_classifier.predict(x_test)

y_train_pred3 = knn_classifier.predict(x_train)

# Evaluate training set accuracy
accuracy_train3 = accuracy_score(y_train, y_train_pred3)
print(f'Accuracy on the training set: {accuracy_train3 * 100:.2f}%')

# Evaluate the model on the testing set
accuracy_test3 = accuracy_score(y_test, y_test_pred3)
print(f'Accuracy on the test set: {accuracy_test3*100 :.2f}%')

# Additional Evaluation Metrics
print('\nClassification Report:')
print(classification_report(y_test, y_test_pred3))

print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_test_pred3))


# # Logistic Regression tweaked 

# In[30]:


logreg_model_tweaked = LogisticRegression(
    C=1.0,  # Regularization strength (you can adjust this)
    solver='liblinear',  # Optimization algorithm (you can try different solvers)
    max_iter=1  # Maximum number of iterations for optimization (you can adjust this)
)


# In[31]:


# Training the tweaked model
logreg_model_tweaked.fit(x_train, y_train)


# In[32]:


# Making predictions on the training set
y_train_pred4 = logreg_model_tweaked.predict(x_train)

# Making predictions on the test set
y_test_pred4 = logreg_model_tweaked.predict(x_test)

# Evaluating the model on training set
accuracy_train4 = accuracy_score(y_train, y_train_pred4)
print(f'Training Set Accuracy: {accuracy_train4*100:.2f}%')

# Evaluating the model on test set
accuracy_test4 = accuracy_score(y_test, y_test_pred4)
print(f'Testing Set Accuracy: {accuracy_test4*100:.2f}%')

# Additional Evaluation Metrics
print('\nClassification Report (Test Set):')
print(classification_report(y_test, y_test_pred4))


# # Random Forest

# In[33]:


from sklearn.ensemble import RandomForestClassifier


# In[34]:


#rf_classifier = RandomForestClassifier(random_state=42)     #this gives 100% accuracy 

rf_classifier = RandomForestClassifier(n_estimators=10, random_state=42)


# In[35]:


# Train the model
rf_classifier.fit(x_train, y_train)


# In[36]:


# Making predictions on the training set
y_train_pred5 = logreg_model_tweaked.predict(x_train)

# Making predictions on the test set
y_test_pred5 = logreg_model_tweaked.predict(x_test)

# Evaluating the model on training set
accuracy_train5 = accuracy_score(y_train, y_train_pred4)
print(f'Training Set Accuracy: {accuracy_train5*100:.2f}%')

# Evaluating the model on test set
accuracy_test5 = accuracy_score(y_test, y_test_pred4)
print(f'Testing Set Accuracy: {accuracy_test5*100:.2f}%')

# Additional Evaluation Metrics
print('\nClassification Report (Test Set):')
print(classification_report(y_test, y_test_pred5))


# # Summport Vector Machine(SVM)

# In[37]:


from sklearn.svm import SVC


# In[38]:


# Initialize the SVM classifier
svm_classifier = SVC(kernel='poly', C=1.0 ,class_weight='balanced', random_state=42)


# In[39]:


# Train the SVM model
svm_classifier.fit(x_train, y_train)


# In[40]:


# Making predictions on the training set
y_train_pred6 = svm_classifier.predict(x_train)

# Making predictions on the test set
y_test_pred6 = svm_classifier.predict(x_test)

# Evaluating the model on training set
accuracy_train6 = accuracy_score(y_train, y_train_pred6)
print(f'Training Set Accuracy: {accuracy_train6*100:.2f}%')

# Evaluating the model on test set
accuracy_test6 = accuracy_score(y_test, y_test_pred6)
print(f'Testing Set Accuracy: {accuracy_test6*100:.2f}%')

# Additional Evaluation Metrics
print('\nClassification Report (Test Set):')
print(classification_report(y_test, y_test_pred6))


# # Naive Bayers

# In[41]:


from sklearn.naive_bayes import MultinomialNB


# In[42]:


# Initialize the Naive Bayes classifier (Multinomial Naive Bayes for text data)
naive_bayes_classifier = MultinomialNB(alpha=0.1)


# In[43]:


# Train the Naive Bayes model
naive_bayes_classifier.fit(x_train, y_train)


# In[44]:


# Making predictions on the training set
y_train_pred7 = naive_bayes_classifier.predict(x_train)

# Making predictions on the test set
y_test_pred7 = naive_bayes_classifier.predict(x_test)

# Evaluating the model on training set
accuracy_train7 = accuracy_score(y_train, y_train_pred7)
print(f'Training Set Accuracy: {accuracy_train7*100:.2f}%')

# Evaluating the model on test set
accuracy_test7 = accuracy_score(y_test, y_test_pred7)
print(f'Testing Set Accuracy: {accuracy_test7*100:.2f}%')

# Additional Evaluation Metrics
print('\nClassification Report (Test Set):')
print(classification_report(y_test, y_test_pred7))


# # Stochastic Gradient Descent

# In[45]:


from sklearn.linear_model import SGDClassifier


# In[46]:


# Initialize the stochastic gradient descent classfifier
sgd_classifier = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)


# In[47]:


# Train the model on the training set
sgd_classifier.fit(x_train, y_train)


# In[48]:


# Test the model on the testing set
y_test_pred7 = sgd_classifier.predict(x_test)
y_train_pred7 = sgd_classifier.predict(x_train)

# Evaluate training set accuracy
accuracy_train7 = accuracy_score(y_train, y_train_pred7)
print(f'Accuracy on the training set: {accuracy_train7 * 100:.2f}%')

# Evaluate testing set accuracy
accuracy_test7 = accuracy_score(y_test, y_test_pred7)
print(f'Accuracy on the testing set: {accuracy_test7 * 100:.2f}%')

# Additional Evaluation Metrics
print('\nClassification Report for Testing Set:')
print(classification_report(y_test, y_test_pred7))

print('\nConfusion Matrix for Testing Set:')
print(confusion_matrix(y_test, y_test_pred7))


# # tweak the sgd model

# # Ensemble Model

# In[ ]:


#to do


# #  Taking input from the user using check boxes

# In[49]:


import ipywidgets as widgets
from IPython.display import display

from sklearn.preprocessing import LabelEncoder


# In[50]:


symptoms_list = df.columns.tolist()
symptoms_list.remove('Illness')    #removes the target column so that the checkbox doesnt show illness values


# In[51]:


# Creating a dictionary to store the state of each checkbox
checkbox_states = {symptom: widgets.Checkbox(description=symptom, value=False) for symptom in symptoms_list}

#value=False : sets the initial state of the checkbox to False (unchecked)


# In[52]:


# Create a dictionary to store checkbox states
checkbox_states = {symptom: widgets.Checkbox(description=symptom, value=False) for symptom in symptoms_list}
#value=False : sets the initial state of the checkbox to False (unchecked)

symptom_values = {symptom: 0 for symptom in symptoms_list}  # Dictionary to store symptom values (0 or 1)

# Function to handle checkbox changes and update selected symptoms
def update_selected_symptoms(change):
    for symptom, checkbox in checkbox_states.items():
        symptom_values[symptom] = int(checkbox.value)
    print("Symptom Values:", symptom_values)

# Attach the update function to each checkbox
for checkbox in checkbox_states.values():
    checkbox.observe(update_selected_symptoms, names='value')

# Display checkboxes
checkboxes = widgets.VBox(list(checkbox_states.values()))
display(checkboxes)


# In[53]:


len(symptom_values)


# In[55]:


# Convert symptom_values to a 2D array
input_data = np.array([list(symptom_values.values())])

# Check if any symptom is selected
if np.any(input_data):
    # Use knn_classifier.predict()
    predicted_illness = knn_classifier.predict(input_data)

    # Get the predicted illness name
    predicted_illness_name = df.iloc[predicted_illness - 1]['Illness']

    # Print the predicted illness
    print("Predicted Illness:", predicted_illness_name)
else:
    # Handle the case when no symptoms are selected
    print("Nothing is selected for prediction.")



# # Linking the model to the remedy dataset
# 

# In[56]:


#load the data

df=pd.read_csv('remedies_1.csv')
df2=pd.DataFrame(df)


# In[57]:


df2.head()


# In[58]:


df2.info()    #print summary of dataframe|


# In[59]:


df2.shape  #to see if equal no.of entries are there wrt the illness dataset


# # check for printing the final result

# In[67]:


pip install gtts


# In[68]:


# Import required libraries
import pandas as pd
import numpy as np
from gtts import gTTS
import os

# Set pandas display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Convert symptom_values to a 2D array
input_data = np.array([list(symptom_values.values())])

# Check if any symptom is selected
if np.any(input_data):
    # Use knn_classifier.predict()
    predicted_illness = knn_classifier.predict(input_data)

    # Get the predicted illness name
    predicted_illness_name = df.iloc[predicted_illness - 1]['Illness']

    # Check if predicted illness matches 'Illness' column
    matching_rows = df[df['Illness'].eq(predicted_illness_name)]

    if not matching_rows.empty:
        # Print the row(s) corresponding to the matching illness in a readable format
        print("\nPredicted Illness:", predicted_illness_name)

        for idx, row in matching_rows.iterrows():
            print("\nRemedies:")
            print(f"1. {row['Remedy_1']}")
            print(f"2. {row['Remedy_2']}")
            print(f"3. {row['Remedy_3']}")
            print(f"4. {row['Remedy_4']}")
            
            print("\nNote/when to see a doctor:")
            print(f"{row['Note/when to see a doctor']}")

            print("\nFoods to eat:")
            for i, food in enumerate(row['Foods to eat'].split('\n'), start=1):
                print(f"{i}. {food}")

            print("\nFoods to avoid:")
            print(row['Foods to avoid'])

            print("\n" + "="*80)  # Separating each row with a line of equal signs

            # Convert the text to speech using gTTS
            text_to_speech = "\n".join([
                f"Predicted Illness: {predicted_illness_name}",
                f"Remedies: \n1. {row['Remedy_1']} \n2. {row['Remedy_2']} \n3. {row['Remedy_3']} \n4. {row['Remedy_4']}",
                f"Note/when to see a doctor: {row['Note/when to see a doctor']}",
                f"Foods to eat: \n" + "\n".join([f"{i}. {food}" for i, food in enumerate(row['Foods to eat'].split('\n'), start=1)]),
                f"Foods to avoid: {row['Foods to avoid']}"
            ])

            # Save the text to a temporary file
            temp_file_path = "output_text.txt"
            with open(temp_file_path, "w") as temp_file:
                temp_file.write(text_to_speech)

            # Convert the temporary file to speech
            tts = gTTS(text_to_speech, lang='en')
            tts.save("output_speech.mp3")

            # Play the generated speech
            os.system("start output_speech.mp3")

            # Remove temporary files
            os.remove(temp_file_path)
            os.remove("output_speech.mp3")
    else:
        print("\nNo matching rows found for the predicted illness.")
else:
    # Handle the case when no symptoms are selected
    print("\nNothing is selected for prediction.")


# In[97]:


#this is it 

import pandas as pd
import numpy as np
from gtts import gTTS
import os
import platform

# Set pandas display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Convert symptom_values to a 2D array
input_data = np.array([list(symptom_values.values())])

# Check if any symptom is selected
if np.any(input_data):
    # Use knn_classifier.predict()
    predicted_illness = knn_classifier.predict(input_data)

    # Get the predicted illness name
    predicted_illness_name = df.iloc[predicted_illness - 1]['Illness']

    # Check if predicted illness matches 'Illness' column
    matching_rows = df[df['Illness'].eq(predicted_illness_name)]

    if not matching_rows.empty:
        # Print the row(s) corresponding to the matching illness in a readable format
        print("\nPredicted Illness:", predicted_illness_name)

        for idx, row in matching_rows.iterrows():
            print("\nRemedies:")
            print(f"1. {row['Remedy_1']}")
            print(f"2. {row['Remedy_2']}")
            print(f"3. {row['Remedy_3']}")
            print(f"4. {row['Remedy_4']}")
            
            print("\nNote/when to see a doctor:")
            print(f"{row['Note/when to see a doctor']}")

            print("\nFoods to eat:")
            for i, food in enumerate(row['Foods to eat'].split('\n'), start=1):
                print(f"{i}. {food}")

            print("\nFoods to avoid:")
            print(row['Foods to avoid'])

            print("\n" + "="*80)  # Separating each row with a line of equal signs

            # Convert the text to speech using gTTS
            text_to_speech = "\n".join([
                f"Predicted Illness: {predicted_illness_name}",
                f"Remedies: \n1. {row['Remedy_1']} \n2. {row['Remedy_2']} \n3. {row['Remedy_3']} \n4. {row['Remedy_4']}",
                f"Note/when to see a doctor: {row['Note/when to see a doctor']}",
                f"Foods to eat: \n" + "\n".join([f"{i}. {food}" for i, food in enumerate(row['Foods to eat'].split('\n'), start=1)]),
                f"Foods to avoid: {row['Foods to avoid']}"
            ])

            # Save the text to a temporary file
            temp_file_path = "output_text.txt"
            with open(temp_file_path, "w") as temp_file:
                temp_file.write(text_to_speech)

            # Convert the temporary file to speech
            tts = gTTS(text_to_speech, lang='en')
            
            # Save the audio file with a unique name
            audio_file_path = f"output_speech_{idx}.mp3"
            tts.save(audio_file_path)

            # Play the generated speech
            if platform.system() == 'Windows':
                os.system(f"start {audio_file_path}")
            elif platform.system() == 'Linux':
                os.system(f"xdg-open {audio_file_path}")
            elif platform.system() == 'Darwin':  # macOS
                os.system(f"open {audio_file_path}")
            else:
                print("Unsupported operating system for opening audio file.")

            # Remove temporary files
            os.remove(temp_file_path)
    else:
        print("\nNo matching rows found for the predicted illness.")
else:
    # Handle the case when no symptoms are selected
    print("\nNothing is selected for prediction.")


# In[71]:


pip install googletrans==4.0.0-rc1


# In[75]:


from googletrans import Translator

# Set pandas display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Convert symptom_values to a 2D array
input_data = np.array([list(symptom_values.values())])

# Check if any symptom is selected
if np.any(input_data):
    # Use knn_classifier.predict()
    predicted_illness = knn_classifier.predict(input_data)

    # Get the predicted illness name
    predicted_illness_name = df.iloc[predicted_illness - 1]['Illness']

    # Check if predicted illness matches 'Illness' column
    matching_rows = df[df['Illness'].eq(predicted_illness_name)]

    if not matching_rows.empty:
        # Print the row(s) corresponding to the matching illness in a readable format
        print("\nPredicted Illness:", predicted_illness_name)

        for idx, row in matching_rows.iterrows():
            print("\nRemedies:")
            print(f"1. {row['Remedy_1']}")
            print(f"2. {row['Remedy_2']}")
            print(f"3. {row['Remedy_3']}")
            print(f"4. {row['Remedy_4']}")
            
            print("\nNote/when to see a doctor:")
            print(f"{row['Note/when to see a doctor']}")

            print("\nFoods to eat:")
            for i, food in enumerate(row['Foods to eat'].split('\n'), start=1):
                print(f"{i}. {food}")

            print("\nFoods to avoid:")
            print(row['Foods to avoid'])

            print("\n" + "="*80)  # Separating each row with a line of equal signs
            
            # Translate the output to Hindi
            translator = Translator()
            text_to_translate = "\n".join([
                f"Predicted Illness: {predicted_illness_name}",
                f"Remedies: \n1. {row['Remedy_1']} \n2. {row['Remedy_2']} \n3. {row['Remedy_3']} \n4. {row['Remedy_4']}",
                f"Note/when to see a doctor: {row['Note/when to see a doctor']}",
                f"Foods to eat: \n" + "\n".join([f"{i}. {food}" for i, food in enumerate(row['Foods to eat'].split('\n'), start=1)]),
                f"Foods to avoid: {row['Foods to avoid']}"
            ])
            
            translated_text = translator.translate(text_to_translate, dest='hi').text
            print("\nTranslated Output (Hindi):")
            print(translated_text)
    else:
        print("\nNo matching rows found for the predicted illness.")
else:
    # Handle the case when no symptoms are selected
    print("\nNothing is selected for prediction.")


# In[78]:


pip install pyttsx3


# In[ ]:


#this it is

from googletrans import Translator
import pandas as pd
import numpy as np

# Set pandas display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Ask the user to input the desired language
print("If you want the text translated to any of the below languages- enter the respective value")
print("Select the language:")
print("1. English ('en')")
print("2. Hindi ('hi')")
print("3. Marathi ('mr')")
print("4. Kannada ('kn')")
print("5. Bengali ('bn')")
print("6. Telugu ('te')")
print("7. Tamil ('ta')")
print("8. Malayalam ('ml')")
print("9. Odia ('or')")
print("10. Gujarati ('gu')")

selected_option = input("Enter the language option (1-10): ")

# Mapping user input to language code
language_mapping = {
    '1': 'en', '2': 'hi', '3': 'mr', '4': 'kn', '5': 'bn',
    '6': 'te', '7': 'ta', '8': 'ml', '9': 'or', '10': 'gu'
}

selected_language = language_mapping.get(selected_option)

if selected_language:
    # Convert symptom_values to a 2D array
    input_data = np.array([list(symptom_values.values())])

    # Check if any symptom is selected
    if np.any(input_data):
        # Use knn_classifier.predict()
        predicted_illness = knn_classifier.predict(input_data)

        # Get the predicted illness name
        predicted_illness_name = df.iloc[predicted_illness - 1]['Illness']

        # Check if predicted illness matches 'Illness' column
        matching_rows = df[df['Illness'].eq(predicted_illness_name)]

        if not matching_rows.empty:
            # Translate the output to the selected language
            translator = Translator()
            text_to_translate = "\n".join([
                f"Predicted Illness: {predicted_illness_name}",
                f"Remedies: \n1. {row['Remedy_1']} \n2. {row['Remedy_2']} \n3. {row['Remedy_3']} \n4. {row['Remedy_4']}",
                f"Note/when to see a doctor: {row['Note/when to see a doctor']}",
                f"Foods to eat: \n" + "\n".join([f"{i}. {food}" for i, food in enumerate(row['Foods to eat'].split('\n'), start=1)]),
                f"Foods to avoid: {row['Foods to avoid']}"
            ])

            translated_text = translator.translate(text_to_translate, dest=selected_language).text

            # Print the translated output in a readable format
            print(f"\nTranslated Output ({selected_language}):")
            print(translated_text)

        else:
            print("\nNo matching rows found for the predicted illness.")
    else:
        # Handle the case when no symptoms are selected
        print("\nNothing is selected for prediction.")
else:
    print("\nInvalid language option selected.")


# In[ ]:




