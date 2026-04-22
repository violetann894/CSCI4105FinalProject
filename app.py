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


st.title('COVID-19 Data Trends Visualizer')
tab1, tab2, tab3, tab4 = st.tabs(['Decision Tree Model Information', 'Model Prediction', 'Association Mining Results',
                                  'Cluster Analysis'])

pd.set_option('display.max_columns', None)

dataframe = pd.read_csv('COVID-19_Outcomes_by_Vaccination_Status_-_Historical_20260312.csv')

dataframe = dataframe.drop(columns=['Week End', 'Crude Vaccinated Ratio',
                                    'Crude Boosted Ratio','Age-Adjusted Unvaccinated Rate',
                                    'Age-Adjusted Vaccinated Rate','Age-Adjusted Boosted Rate',
                                    'Age-Adjusted Vaccinated Ratio','Age-Adjusted Boosted Ratio',
                                    'Population Unvaccinated','Population Vaccinated','Population Boosted',
                                    'Outcome Unvaccinated','Outcome Vaccinated','Outcome Boosted',
                                    'Age Group Min','Age Group Max'])

dataframe = dataframe.drop(dataframe[dataframe['Age Group'] == 'All'].index)

dataframe['Unvaccinated Rate'] = dataframe['Unvaccinated Rate'].str.replace(',', '').astype(float)
dataframe['Vaccinated Rate'] = dataframe['Vaccinated Rate'].str.replace(',', '').astype(float)
dataframe['Boosted Rate'] = dataframe['Boosted Rate'].str.replace(',', '').astype(float)

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(dataframe[['Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate']])
dataframe[['Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate']] = (
    imputer.transform(dataframe[['Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate']]))

unvaccinated_bins = [-1, 0, 5, 30, 150, np.inf]
unvaccinated_labels = ['Zero', 'Very Low', 'Low', 'Medium', 'High']

vaccinated_bins = [-1, 0, 2, 15, 80, np.inf]
vaccinated_labels = ['Zero', 'Very Low', 'Low', 'Medium', 'High']

booster_bins = [-1, 0, 3, 10, 60, np.inf]
booster_labels = ['Zero', 'Very Low', 'Low', 'Medium', 'High']

dataframe['Unvaccinated Rate'] = pd.cut(dataframe['Unvaccinated Rate'], bins=unvaccinated_bins, labels=unvaccinated_labels)
dataframe['Vaccinated Rate'] = pd.cut(dataframe['Vaccinated Rate'], bins=vaccinated_bins, labels=vaccinated_labels)
dataframe['Boosted Rate'] = pd.cut(dataframe['Boosted Rate'], bins=booster_bins, labels=booster_labels)

dataframe['Outcome_Age'] = dataframe['Outcome'] + '_' + dataframe['Age Group']
age_order = {'0-4': 0, '5-11': 1, '12-17': 2, '18-29': 3, '30-49': 4, '50-64': 5, '65-79': 6, '80+': 7}
dataframe['Age Group Encoded'] = dataframe['Age Group'].map(age_order)

le = LabelEncoder()
dataframe['Outcome_Age'] = le.fit_transform(dataframe['Outcome_Age'])

for col in ['Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate']:
    dataframe[col] = le.fit_transform(dataframe[col].astype(str))

with st.sidebar:
    st.header('About')
    st.write('The goal of this project is to create a visualization application that synthesizes COVID-19 information '
             'from the city of Chicago and outputs information regarding anomalies and the associations between '
             'different variables. Using association data mining and a decision tree classifier, we aim to visualize '
             'the relationships between the different variables and the possible historical reasons for said trends. '
             'The Generative Artificial Intelligence portion of the project aims to allow users to query the AI with '
             'questions regarding the graphs created by the different techniques.')

    st.header('Application Created By ')
    st.write('Group 2: Rachel Hussmann, Jeannine Elmasri , Sophia Milask, Anissa Serafine')

with tab1:

    st.header('Decision Tree Classifier')
    st.write('We created a decision tree classifier from the COVID-19 dataset to create a way to predict '
                 'what class a record may fall into based on the following columns and information: Age group, '
                 'boosted rate, vaccinated rate, and unvaccinated rate. The prediction that the classifier is trying to '
                 'complete is deciding whether the combination of those features would result in a case, hospitalization or a death.')

    col1, col2, col3 = st.columns(3)
    col1.metric('Model Accuracy', '87%')
    col2.metric('Total Records', '3,336')
    col3.metric('Classes', '3')

    x = dataframe[['Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate', 'Age Group Encoded']]
    y = dataframe['Outcome']

    xtraining_data, xtesting_data, ytraining_data, ytesting_data = sk.train_test_split(x, y, test_size=0.2,
                                                                                       random_state=42)

    decisionTree = tree.DecisionTreeClassifier(criterion='entropy', class_weight='balanced')
    decisionTree = decisionTree.fit(xtraining_data, ytraining_data)

    feature = [col for col in x.columns]

    st.subheader('Visualization of the Decision Tree')

    st.write('Below is the structure of the classifier and how it makes decisions based on the training set we have '
             'given it. The current makeup of the data is 80% training and 20% testing. The view that can be seen '
             'is halted at three levels, but the maximum depth of the tree is unlimited.')

    decisionTreePlt, ax1 = plt.subplots(figsize=(80, 20), dpi=150)
    tree.plot_tree(decisionTree, feature_names=feature, filled=True, fontsize=25, max_depth=3, ax=ax1)
    st.pyplot(decisionTreePlt)
    plt.close(decisionTreePlt)

    y_predict = decisionTree.predict(xtesting_data)

    st.subheader('Classification Report')
    st.write('The classification report gives more specific performance metrics associated with the decision tree '
             'classifier. These metrics include precision, recall, f1-score, support and overall accuracy')

    report = classification_report(ytesting_data, y_predict, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(2)
    report_df.index = ['Cases', 'Deaths', 'Hospitalizations', 'Accuracy', 'Macro Avg', 'Weighted Avg']
    st.dataframe(report_df, width= 'stretch', column_config={
        "_index": st.column_config.TextColumn("Class", width="medium")
    })

    confusion = confusion_matrix(ytesting_data, y_predict)

    st.subheader('Confusion Matrix')
    st.write('The confusion matrix is responsible for showing us what the model classified correctly and incorrectly. '
             'This allowed us to tune specific attributes of the model in a way to improve those areas and in turn '
             'create an overall better performing model. ')

    cm, ax2 = plt.subplots()
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Reds')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(cm)
    plt.close(cm)

# Rework this section (a bit buggy)
with tab2:
    st.header('Predict Outcome')
    st.write('Using the decision tree classifier, input the age group and vaccination rate levels to predict '
             'whether the profile is associated with a case, hospitalization or death outcome.')

    with st.form('prediction_form'):
        col1, col2 = st.columns(2)

        with col1:
            age_input = st.selectbox('Age Group', ['0-4', '5-11', '12-17', '18-29', '30-49', '50-64', '65-79', '80+'])
            unvax_input = st.selectbox('Unvaccinated Rate', ['Zero', 'Very Low', 'Low', 'Medium', 'High'])
        with col2:
            vax_input = st.selectbox('Vaccinated Rate', ['Zero', 'Very Low', 'Low', 'Medium', 'High'])
            boost_input = st.selectbox('Boosted Rate', ['Zero', 'Very Low', 'Low', 'Medium', 'High'])

        submitted = st.form_submit_button('Predict Outcome')

    if submitted:
        label_order = ['Zero', 'Very Low', 'Low', 'Medium', 'High']
        age_order = {'0-4': 0, '5-11': 1, '12-17': 2, '18-29': 3, '30-49': 4, '50-64': 5, '65-79': 6, '80+': 7}

        input_df = pd.DataFrame([[
            label_order.index(unvax_input),
            label_order.index(vax_input),
            label_order.index(boost_input),
            age_order[age_input]
        ]], columns=['Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate', 'Age Group Encoded'])

        prediction = decisionTree.predict(input_df)
        st.success(f'Predicted Outcome: {prediction[0]}')


# Refactor (also buggy)
with tab3:
    st.header('Association Mining')
    st.write('For this part of the project, we created an interactive tool that allows you the user to choose what '
             'threshold values you want. We utilized the Apriori algorithm for this part of the project. Please input '
             'the minimum values for support, confidence, and lift. Once you submit your choice, the program will '
             'output what the rules it found were, any supporting values, and the plot at the bottom.')

    col1, col2, col3 = st.columns(3)
    with col1:
        min_support = st.slider('Minimum Support', 0.01, 1.0, 0.20)
    with col2:
        min_confidence = st.slider('Minimum Confidence', 0.01, 1.0, 0.20)
    with col3:
        min_lift = st.slider('Minimum Lift', 0.0, 10.0, 1.0)

    if st.button('Mine Association Rules'):
        transactionsForAlgo = []
        for _, row in dataframe.iterrows():
            transaction = [
                f'Outcome_{row["Outcome"]}',
                f'Age_Group_{row["Age Group"]}',
                f'Unvaccinated_Rate_{row["Unvaccinated Rate"]}',
                f'Boosted_Rate_{row["Boosted Rate"]}',
                f'Vaccinated_Rate_{row["Vaccinated Rate"]}'
            ]
            transactionsForAlgo.append(transaction)

        te = TransactionEncoder()
        te_array = te.fit(transactionsForAlgo).transform(transactionsForAlgo)
        dataframe_encoded = pd.DataFrame(te_array, columns=te.columns_)

        frequent_itemsets = apriori(dataframe_encoded, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)

        filtered_rules = rules[rules['lift'] >= min_lift].copy()
        filtered_rules['antecedents'] = filtered_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        filtered_rules['consequents'] = filtered_rules['consequents'].apply(lambda x: ', '.join(list(x)))

        col1, col2, col3 = st.columns(3)
        col1.metric('Total Rules Found', len(filtered_rules))
        col2.metric('Avg Confidence', f"{filtered_rules['confidence'].mean():.2f}")
        col3.metric('Avg Lift', f"{filtered_rules['lift'].mean():.2f}")

        st.dataframe(filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']],
                     width='stretch')

        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(filtered_rules['support'], filtered_rules['confidence'],
                             c=filtered_rules['lift'], cmap='RdYlGn', alpha=0.7, s=50)
        plt.colorbar(scatter, ax=ax, label='Lift')
        ax.set_xlabel('Support')
        ax.set_ylabel('Confidence')
        ax.set_title('Association Rules: Support vs Confidence (colored by Lift)')
        st.pyplot(fig)
        plt.close(fig)

with tab4:

    st.header('K-Means Clustering')
    st.write('Clustering helps us understand the natural relationships between data points. In this project, we used '
             'the k-means method of clustering. Before we were able to create the clusters, the number of clusters had '
             'to be discovered. To figure this out, we used the elbow method to find the best number of clusters to '
             'used based on the dataset. The graph below displays the results of applying the elbow method.')

    labels = ['Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate', 'Age Group Encoded']
    x = dataframe[['Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate', 'Age Group Encoded']]
    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(x)

    wcss = []
    for i in range(1, 6):
        k_mean_clusters = KMeans(n_clusters=i)
        k_mean_clusters.fit(scaled_x)
        wcss.append(k_mean_clusters.inertia_)

    fig, ax = plt.subplots(figsize=(10,6))
    plt.plot(range(1,6), wcss)
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('WCSS')
    st.pyplot(fig)
    plt.close(fig)

    st.write('In this graph, it shows that the best number of clusters to use is three, as any additional clusters '
             'would only provide a minimal increase in information.')

    k = 3
    clusters = KMeans(n_clusters=k, random_state=42)
    clusters.fit_predict(scaled_x)
    centroids = clusters.cluster_centers_

    st.subheader('Cluster Profiles')
    st.write('The table below shows data from the centroids of each cluster:')
    st.table(pd.DataFrame(centroids, columns=labels, index=['Cluster 1', 'Cluster 2', 'Cluster 3']))

    st.write('Each of these clusters tell us something about the dataset. The first cluster represented the younger '
             'population, as defined by their average age group. Based on the data from the rates of vaccination, this '
             'group appeared to not follow through with the complete vaccination series. While their rates of unvaccination '
             'and vaccination were high, their boosted status was the lowest of the three clusters, informating us that '
             'they were a diversified age group that either: 1. did not get the vaccine at all or 2. received the first '
             'part of the vaccine but not the booster. The second cluster is based on our middle age group. The rates '
             'of vaccination for this group was a bit all over the place, which makes it a bit difficult to interpret '
             'specific behaviors for this group. However, comparing their rates to the other clusters, in can be '
             'inferred that this group was very polarized on the topic of vaccination. This cluster represents the '
             'working age citizen that most likely faced challenges regarding vaccination status. Based on information '
             'from the pandemic, this was most likely the group that faced vaccination requirements in the face of '
             'uncertainty with regard to the vaccine. This cluster perfectly represents that age group and the '
             'diversity in opinions that occurred during that timeframe. The last cluster represents the elderly '
             'community. Comparing their rates of vaccination to the other groups, they had the highest rate of '
             'vaccination, meaning that they were the age group that were the most protected during the pandemic.')