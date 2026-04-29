import streamlit as st
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
import sklearn.tree as tree
import sklearn.model_selection as sk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Creates the title for the application
st.title('COVID-19 Data Trends Visualizer')

# Creates the various tabs needed to show all the tools used for the project
tab1, tab2, tab3, tab4 = st.tabs(['Decision Tree Model Information', 'Model Prediction', 'Association Mining Results',
                                  'Cluster Analysis'])
# Imports the COVID 19 data from the .csv file
dataframe = pd.read_csv('COVID-19_Outcomes_by_Vaccination_Status_-_Historical_20260312.csv')

# Drops the columns that are not needed for analysis
dataframe = dataframe.drop(columns=['Week End', 'Crude Vaccinated Ratio',
                                    'Crude Boosted Ratio','Age-Adjusted Unvaccinated Rate',
                                    'Age-Adjusted Vaccinated Rate','Age-Adjusted Boosted Rate',
                                    'Age-Adjusted Vaccinated Ratio','Age-Adjusted Boosted Ratio',
                                    'Population Unvaccinated','Population Vaccinated','Population Boosted',
                                    'Outcome Unvaccinated','Outcome Vaccinated','Outcome Boosted',
                                    'Age Group Min','Age Group Max'])

# All age group is difficult to factor into to the other more concrete age groups, so we elected to remove those rows
dataframe = dataframe.drop(dataframe[dataframe['Age Group'] == 'All'].index)

# Some of the numbers in the different rates have commas (e.g. 2,300.00), which cannot be read
# Removes the comma from rates that have them and replaces the original value
dataframe['Unvaccinated Rate'] = dataframe['Unvaccinated Rate'].str.replace(',', '')
dataframe['Vaccinated Rate'] = dataframe['Vaccinated Rate'].str.replace(',', '')
dataframe['Boosted Rate'] = dataframe['Boosted Rate'].str.replace(',', '')

# Some values are missing, which would negatively affect the data
# Uses SimpleImputer to replace missing values with the median of the various vaccination rates we will be looking at
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(dataframe[['Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate']])
dataframe[['Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate']] = (
    imputer.transform(dataframe[['Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate']]))

# Used AI to assist with picking age ranges that would best fit the data
# Creates a dictionary that holds the encoding for the ages
age_dictionary = {'0-4': 0, '5-11': 1, '12-17': 2, '18-29': 3, '30-49': 4, '50-64': 5, '65-79': 6, '80+': 7}

# Creates the reversed age dictionary in {encoded_value, key} format
reversed_age_dictionary = {}
for key, value in age_dictionary.items():
    reversed_age_dictionary[value] = key

# Encodes the age groups for analysis
dataframe['Age Group Encoded'] = dataframe['Age Group'].map(age_dictionary)
dataframe['Age Group Unencoded'] = dataframe['Age Group Encoded'].map(reversed_age_dictionary)

# Used AI to assist with picking bins that would best fit these features
# Creates bins to fit the data into that can then be encoded for analysis
unvaccinated_bins = [-1, 0, 5, 30, 150, np.inf]
unvaccinated_labels = ['Zero', 'Very Low', 'Low', 'Medium', 'High']

# Creates bins to fit the data into that can then be encoded for analysis
vaccinated_bins = [-1, 0, 2, 15, 80, np.inf]
vaccinated_labels = ['Zero', 'Very Low', 'Low', 'Medium', 'High']

# Creates bins to fit the data into that can then be encoded for analysis
booster_bins = [-1, 0, 3, 10, 60, np.inf]
booster_labels = ['Zero', 'Very Low', 'Low', 'Medium', 'High']

# Places the rates into the created bins and replaces the value with the label
dataframe['Unvaccinated Rate'] = pd.cut(dataframe['Unvaccinated Rate'], bins=unvaccinated_bins, labels=unvaccinated_labels)
dataframe['Vaccinated Rate'] = pd.cut(dataframe['Vaccinated Rate'], bins=vaccinated_bins, labels=vaccinated_labels)
dataframe['Boosted Rate'] = pd.cut(dataframe['Boosted Rate'], bins=booster_bins, labels=booster_labels)

# Encodes the string value given into an integer value to be used for analysis
le = LabelEncoder()
for column in ['Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate']:
    dataframe[column] = le.fit_transform(dataframe[column])

# Creates a sidebar with information that is always displayed, no matter what page is open
with st.sidebar:
    st.header('About')
    st.write('The goal of this project is to create a visualization application that synthesizes COVID-19 information '
             'from the city of Chicago and outputs information regarding anomalies and the associations between '
             'different variables. Using association data mining and a decision tree classifier, we aim to visualize '
             'the relationships between the different variables and the possible historical reasons for said trends. '
             'The Generative Artificial Intelligence portion of the project aims to allow users to query the AI with '
             'questions regarding the graphs created by the different techniques.')
    st.header('Application Created By ')
    st.write('Group 2: Rachel Hussmann, Jeannine Elmasri, Sophia Milask, Anissa Serafine')

# Creates a tab for the decision tree classifier section
with tab1:

    # Information about what a decision tree classifier is and how that connects to our chosen dataset
    st.header('Decision Tree Classifier')
    st.write('We created a decision tree classifier from the COVID-19 dataset to create a way to predict '
                 'what class a record may fall into based on the following columns and information: Age group, '
                 'boosted rate, vaccinated rate, and unvaccinated rate. The prediction that the classifier is trying to '
                 'complete is deciding whether the combination of those features would result in a case, hospitalization or a death.')

    # Creates the columns and inputs some information about that data and the model
    column_one, column_two, column_three = st.columns(3)
    column_one.metric('Model Accuracy', '87%')
    column_two.metric('Total Records', dataframe.shape[0])
    column_three.metric('Classes', 3)

    # Creating the features and the outcome dataframes for training/testing
    x = dataframe[['Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate', 'Age Group Encoded']]
    y = dataframe['Outcome']

    # Creates the training and testing data for the decision tree
    xtraining_data, xtesting_data, ytraining_data, ytesting_data = sk.train_test_split(x, y, test_size=0.2, random_state=42)

    # Creates the decision tree and trains it on the training data
    decisionTree = tree.DecisionTreeClassifier(criterion='entropy', class_weight='balanced')
    decisionTree = decisionTree.fit(xtraining_data, ytraining_data)

    # Gathering the column names from the input features
    feature = []
    for column in x.columns:
        feature.append(column)

    # Title for this section of the application
    st.subheader('Visualization of the Decision Tree')

    # Explanation about the visual of the decision tree
    st.write('Below is the structure of the classifier and how it makes decisions based on the training set we have '
             'given it. The current makeup of the data is 80% training and 20% testing. The view that can be seen '
             'is halted at three levels, but the maximum depth of the tree is unlimited.')

    # Creates the visual of the decision tree
    decisionTreePlt, ax1 = plt.subplots(figsize=(80, 20), dpi=150)
    tree.plot_tree(decisionTree, feature_names=feature, filled=True, fontsize=25, max_depth=3, ax=ax1)
    st.pyplot(decisionTreePlt)
    plt.close(decisionTreePlt)

    # Gives the decision tree the testing data and saves the predictions
    y_predict = decisionTree.predict(xtesting_data)

    # Explanation for the classification report
    st.subheader('Classification Report')
    st.write('The classification report gives more specific performance metrics associated with the decision tree '
             'classifier. These metrics include precision, recall, f1-score, support and overall accuracy')

    # Creates the visual table of the classification report
    report = classification_report(ytesting_data, y_predict, output_dict=True)
    report_dataframe = pd.DataFrame(report).transpose()
    report_dataframe.index = ['Cases', 'Deaths', 'Hospitalizations', 'Accuracy', 'Macro Avg', 'Weighted Avg']
    st.dataframe(report_dataframe, width='stretch')

    # Creates the confusion matrix to see how the decision tree did
    confusion = confusion_matrix(ytesting_data, y_predict)

    # Explanation for the confusion matrix
    st.subheader('Confusion Matrix')
    st.write('The confusion matrix is responsible for showing us what the model classified correctly and incorrectly. '
             'This allowed us to tune specific attributes of the model in a way to improve those areas and in turn '
             'create an overall better performing model. ')

    # Creates the visual output of the confusion matrix
    cm, ax2 = plt.subplots()
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Reds')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(cm)
    plt.close(cm)

# Creates a tab for the decision tree prediction section
with tab2:
    # Information about the tab and instructions for the user on how to use it
    st.header('Decision Tree Prediction')
    st.write('Using our decision tree from the previous tab, we created an interactive format for users to see what '
             'different values for each of the features affect the prediction of the decision tree. Using the dropdowns '
             'below, choose the various values for each feature. When complete, click continue and the decision tree '
             'will show its prediction.')

    # Creates a container for user input
    with st.form('Prediction Form'):

        # Creates the organizational columns
        column_one, column_two, column_three, column_four = st.columns(4)

        # Within the columns, creates a dropdown box where users can pick from values that match the different bin
        # labels for each rate
        with column_one:
            unvaccinated_choice = st.selectbox('Unvaccinated Rate', unvaccinated_labels, placeholder='Zero')
        with column_two:
            vaccinated_choice = st.selectbox('Vaccinated Rate', vaccinated_labels, placeholder='Zero')
        with column_three:
            boosted_choice = st.selectbox('Boosted Rate', booster_labels, placeholder='Zero')
        with column_four:
            age_group_choice = st.selectbox('Age Group', age_dictionary.keys(), placeholder='0-4')

        # Required for the form to work, gives the user a button to submit their choices
        submit = st.form_submit_button('Predict')

    # Checks to see if the button has been clicked
    if submit:

        # If it has, compile the choices into 2d array of values and give it to the decision tree to predict the outcome
        # Display the prediction once it is done
        st.subheader('Model Prediction')
        st.write('Decision Tree Prediction: ', decisionTree.predict([[unvaccinated_labels.index(unvaccinated_choice),
                                                                     vaccinated_labels.index(vaccinated_choice),
                                                                     booster_labels.index(boosted_choice),
                                                                     age_dictionary[age_group_choice]]]))
# Creates a tab for the association mining section
with tab3:

    # Describes what the tab dopes and the instructions for the user
    st.header('Association Mining')
    st.write('For this part of the project, we created an interactive tool that allows you the user to choose what '
             'threshold values you want. We utilized the Apriori algorithm for this part of the project. Please input '
             'the minimum values for support, confidence, and lift. Once you submit your choice, the program will '
             'output what the rules it found were, any supporting values, and the plot at the bottom.')

    # Creates the organizational columns
    column_1, column_2, column_3 = st.columns(3)
    with st.form('Association Mining Form'):
        # Each column has a slider that allows the user to decide what the minimum value for various metrics should be
        with column_1:
            minimum_support = st.slider('Minimum Support', 0.01, 1.0, value=0.2, step=0.01)
        with column_2:
            minimum_confidence = st.slider('Minimum Confidence', 0.01, 1.0, value=0.2, step=0.01)
        with column_3:
            minimum_lift = st.slider('Minimum Lift', 0.0, 2.0, value=0.0, step=0.5)

        # Required for the form to work, gives the user a button to submit their choices
        submit = st.form_submit_button('Submit Minimum Values')

    # Check if the button to submit the values has been clicked
    if submit:

        # Created an empty list to hold the finalized transaction
        transactions = []

        # Used AI to generate the loop and the nice formatting for the values
        for _, row in dataframe.iterrows():
            transaction = [
                f'Outcome_{row["Outcome"]}',
                f'Age_Group_{row["Age Group"]}',
                f'Unvaccinated_Rate_{row["Unvaccinated Rate"]}',
                f'Boosted_Rate_{row["Boosted Rate"]}',
                f'Vaccinated_Rate_{row["Vaccinated Rate"]}'
            ]
            transactions.append(transaction)

        # Creates the transaction encoder that will take the transactions and turn them into something the apriori
        # algorithm can work with
        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        dataframe_encoded = pd.DataFrame(te_array, columns=te.columns_)

        # Uses the apriori algorithm to find the frequent itemsets from the encoded dataframe
        frequent_itemsets = apriori(dataframe_encoded, min_support=minimum_support, use_colnames=True)

        # Uses the association rules function to find the association rules based on confidence
        rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=minimum_confidence)

        # Gets a copy of the rules that are above the minimum_lift set by the user
        filtered_rules = rules[rules['lift'] >= minimum_lift].copy()

        # Removes the 'frozenset' from the beginning of the rules
        for x in filtered_rules.index:
            frozen_ant = filtered_rules['antecedents'].iloc[x]
            ant_format = ', '.join(list(frozen_ant))
            filtered_rules['antecendents'].iloc[x] = ant_format
            frozen_con = filtered_rules['consequents'].iloc[x]
            con_format = ', '.join(list(frozen_con))
            filtered_rules['consequents'].iloc[x] = con_format

        # Refactor (also buggy)
        column_1, column_2, column_3 = st.columns(3)
        column_1.metric('Total Rules Found', len(filtered_rules))
        column_2.metric('Average Confidence', f"{filtered_rules['confidence'].mean():.2f}")
        column_3.metric('Average Lift', f"{filtered_rules['lift'].mean():.2f}")

        # Refactor (also buggy)
        st.dataframe(filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']],
                     width='stretch')

        fig, ax = plt.subplots(figsize=(10, 6))

        # Refactor (also buggy)
        scatter = ax.scatter(filtered_rules['support'], filtered_rules['confidence'],
                             c=filtered_rules['lift'], cmap='RdYlGn', alpha=0.7, s=50)

        plt.colorbar(scatter, ax=ax, label='Lift')

        plt.xlabel('Support')
        plt.ylabel('Confidence')
        plt.title('Association Rules: Support vs Confidence (colored by Lift)')
        st.pyplot(fig)
        plt.close(fig)

with tab4:
    st.header('K-Means Clustering')
    st.write('Clustering helps us understand the natural relationships between data points. In this project, we used '
             'the k-means method of clustering to answer two separate questions: Which age group had the highest '
             'vaccination rate and which age group had the highest boosted rate. Before we were able to create the '
             'clusters, the number of clusters had to be discovered. To figure this out, we used the elbow method to '
             'find the best number of clusters to use based on the dataset. The graphs below displays the results of '
             'applying the elbow method.')

    labels1 = ['Vaccinated Rate', 'Age Group Encoded']
    x1 = dataframe[['Vaccinated Rate', 'Age Group Encoded']]
    scaler_x1 = StandardScaler()
    scaled_x1 = scaler_x1.fit_transform(x1)

    labels2 = ['Boosted Rate', 'Age Group Encoded']
    x2 = dataframe[['Boosted Rate', 'Age Group Encoded']]
    scaler_x2 = StandardScaler()
    scaled_x2 = scaler_x2.fit_transform(x2)

    sse_x1 = []
    for i in range(1, 9):
        k_mean_clusters = KMeans(n_clusters=i)
        k_mean_clusters.fit(scaled_x1)
        sse_x1.append(k_mean_clusters.inertia_)

    sse_x2 =[]
    for i in range(1, 9):
        k_mean_clusters = KMeans(n_clusters=i)
        k_mean_clusters.fit(scaled_x2)
        sse_x2.append(k_mean_clusters.inertia_)

    st.subheader('Elbow Method for Vaccination Rate and Age Group')
    fig, ax = plt.subplots(figsize=(10,6))
    plt.plot(range(1,9), sse_x1)
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('WCSS')
    st.pyplot(fig)
    plt.close(fig)

    st.write('In this graph, it shows that the best number of clusters to use is four, as any additional clusters '
             'would only provide a minimal increase in information.')

    st.subheader('Elbow Method for Boosted Rate and Age Group')
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(range(1, 9), sse_x2)
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('WCSS')
    st.pyplot(fig)
    plt.close(fig)

    st.write('In this graph, it shows that the best number of clusters to use is four, as any additional clusters '
             'would only provide a minimal increase in information. After the k-means clusters have been created, we '
             'want to visualize the data to see how much overlap occurs between the clusters. This will help us to '
             'better understand how separate the clusters are and if there are any gray areas where characteristics '
             'of two clusters may overlap.')

    k = 4
    km_x1 = KMeans(n_clusters=k, random_state=42)
    dataframe['Clusters_x1'] = km_x1.fit_predict(scaled_x1)
    centroids_x1 = km_x1.cluster_centers_

    st.subheader('Scatterplot Cluster Graph - Vaccinated Rate and Age Group')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(dataframe['Age Group Unencoded'], dataframe['Vaccinated Rate'], c=dataframe['Clusters_x1'])
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Vaccinated Rate')
    st.pyplot(fig)
    plt.close(fig)

    k = 4
    km_x2 = KMeans(n_clusters=k, random_state=42)
    dataframe['Clusters_x2'] = km_x2.fit_predict(scaled_x2)
    centroids_x2 = km_x2.cluster_centers_

    st.subheader('Scatterplot Cluster Graph - Boosted Rate and Age Group')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(dataframe['Age Group Unencoded'], dataframe['Boosted Rate'], c=dataframe['Clusters_x2'])
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Boosted Rate')
    st.pyplot(fig)
    plt.close(fig)

    st.subheader('Cluster Profiles - Vaccination Status and Age Group')
    st.write('The table below shows data from the centroids of each cluster:')
    st.table(pd.DataFrame(centroids_x1, columns=labels1, index=['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']))

    st.subheader('Cluster Profiles - Boosted Status and Age Group')
    st.write('The table below shows data from the centroids of each cluster:')
    st.table(pd.DataFrame(centroids_x2, columns=labels2, index=['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']))

    st.write('')