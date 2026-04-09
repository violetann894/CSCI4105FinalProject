import streamlit as st
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sklearn.tree as tree
import sklearn.model_selection as sk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

st.title('Analyzing COVID-19 Dataset from Chicago')
tab1, tab2, tab3 = st.tabs(['Decision Tree Model Information', 'Model Prediction', 'Association Mining Results'])

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
dataframe = dataframe[dataframe['Outcome'] != 'Cases']

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
             'The Generative Artificial Intelligence portion of the project aims to allow users to query information '
             'about specific weeks and have the model respond with summarized information. ')

with tab1:

    st.header('Decision Tree Classifier')
    st.write('We created a decision tree classifier from the COVID-19 dataset to create a way to predict '
                 'what class a record may fall into based on the following columns and information: Age group, '
                 'boosted rate, vaccinated rate, and unvaccinated rate. The prediction that the classifier is trying to '
                 'complete is deciding whether the record is a hospitalization or a death.')

    col1, col2, col3 = st.columns(3)
    col1.metric('Model Accuracy', '78%')
    col2.metric('Total Records', '3,336')
    col3.metric('Classes', '2')

    x = dataframe[['Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate']]
    y = dataframe['Outcome']

    xtraining_data, xtesting_data, ytraining_data, ytesting_data = sk.train_test_split(x, y, test_size=0.4,
                                                                                       random_state=42)

    decisionTree = tree.DecisionTreeClassifier(criterion='gini')
    decisionTree = decisionTree.fit(xtraining_data, ytraining_data)

    feature = [col for col in x.columns]

    st.subheader('Visualization of the Decision Tree')

    st.write('Below is the structure of the classifier and how it makes decisions based on the training set we have '
             'given it. The current makeup of the data is 60% training and 40% testing. The view that can be seen '
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
    report_df.index = ['Deaths', 'Hospitalizations', 'Accuracy', 'Macro Avg', 'Weighted Avg']
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
    st.write('Using the decision tree classifier, input the vaccination rate levels to predict '
             'whether the profile is associated with a hospitalization or death outcome.')

    col1, col2, col3 = st.columns(3)

    with col1:
        unvax_input = st.selectbox('Unvaccinated Rate', ['Zero', 'Very Low', 'Low', 'Medium', 'High'])
    with col2:
        vax_input = st.selectbox('Vaccinated Rate', ['Zero', 'Very Low', 'Low', 'Medium', 'High'])
    with col3:
        boost_input = st.selectbox('Boosted Rate', ['Zero', 'Very Low', 'Low', 'Medium', 'High'])

    if st.button('Predict Outcome'):
        label_order = ['Zero', 'Very Low', 'Low', 'Medium', 'High']

        input_df = pd.DataFrame([[
            label_order.index(unvax_input),
            label_order.index(vax_input),
            label_order.index(boost_input)
        ]], columns=['Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate'])

        prediction = decisionTree.predict(input_df)
        st.success(f'Predicted Outcome: {prediction[0]}')


# Refactor (also buggy)
with tab3:
    st.header('Association Mining')
    st.write('For this part of the project, we created an interactive tool that allows you the user to choose what '
             'threshold values you want. We utilized the Apriori algorithm for this part of the project. Please input '
             'the minimum values for support (prunes the infrequent itemsets in the rule generation phase), confidence '
             '(prunes the rules themselves), and lift (prunes the finalized rules). Once you submit your choice, the '
             'program will output what the rules it found were, any supporting values, and the plot at the bottom.')

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
                     use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(filtered_rules['support'], filtered_rules['confidence'],
                             c=filtered_rules['lift'], cmap='RdYlGn', alpha=0.7, s=50)
        plt.colorbar(scatter, ax=ax, label='Lift')
        ax.set_xlabel('Support')
        ax.set_ylabel('Confidence')
        ax.set_title('Association Rules: Support vs Confidence (colored by Lift)')
        st.pyplot(fig)
        plt.close(fig)
