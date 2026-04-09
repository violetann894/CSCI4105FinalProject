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

st.title('Analyzing COVID 19 Dataset from Chicago')
tab1, tab2 = st.tabs(["Decision Tree", "Association Mining"])

pd.set_option('display.max_columns', None)

dataframe = pd.read_csv('COVID-19_Outcomes_by_Vaccination_Status_-_Historical_20260312.csv')

dataframe = dataframe.drop(columns=['Week End', 'Crude Vaccinated Ratio',
                                    "Crude Boosted Ratio","Age-Adjusted Unvaccinated Rate",
                                    "Age-Adjusted Vaccinated Rate","Age-Adjusted Boosted Rate",
                                    "Age-Adjusted Vaccinated Ratio","Age-Adjusted Boosted Ratio",
                                    "Population Unvaccinated","Population Vaccinated","Population Boosted",
                                    "Outcome Unvaccinated","Outcome Vaccinated","Outcome Boosted",
                                    "Age Group Min","Age Group Max"])

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

le = LabelEncoder()
dataframe['Outcome_Age'] = le.fit_transform(dataframe['Outcome_Age'])

for col in ['Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate']:
    dataframe[col] = le.fit_transform(dataframe[col].astype(str))

with tab1:
    x = dataframe.drop(columns=['Outcome', 'Age Group', 'Outcome_Age'])
    y = dataframe['Outcome']

    xtraining_data, xtesting_data, ytraining_data, ytesting_data = sk.train_test_split(x, y, test_size=0.4,
                                                                                       random_state=42)

    decisionTree = tree.DecisionTreeClassifier(criterion='gini')
    decisionTree = decisionTree.fit(xtraining_data, ytraining_data)

    feature = [col for col in x.columns]

    plt.figure(figsize=(45, 20), dpi=200)

    tree.plot_tree(decisionTree, feature_names=feature, filled=True, fontsize=20, max_depth=3)
    plt.show()

    y_predict = decisionTree.predict(xtesting_data)

    print('Classification Report for Decision Tree')
    print(classification_report(ytesting_data, y_predict))
    print('Accuracy score for the decision tree:', accuracy_score(ytesting_data, y_predict), '\n')

    confusion = confusion_matrix(ytesting_data, y_predict)
    sns.heatmap(confusion, annot=True, cmap='Reds')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

with tab2:
    transactionsForAlgo = []

    for _, row in dataframe.iterrows():
        transaction = [
            f"{row['Outcome_Age']}",
            f"Unvaccinated Rate {row['Unvaccinated Rate']}",
            f"Boosted Rate {row['Boosted Rate']}",
            f"Vaccinated Rate {row['Vaccinated Rate']}"
        ]
        transactionsForAlgo.append(transaction)

    te = TransactionEncoder()
    array = te.fit(transactionsForAlgo).transform(transactionsForAlgo)
    dataframe_encoded = pd.DataFrame(array, columns=te.columns_)

    frequent_sets = apriori(dataframe_encoded, min_support=0.01, use_colnames=True)

    rules = association_rules(frequent_sets, metric='confidence', min_threshold=0.1)
    rules = rules[
        rules['antecedents'].apply(lambda x: any('Deaths' in item or 'Hospitalizations' in item for item in x))]
    rules = rules[rules['consequents'].apply(lambda x: len(x) >= 1)]
    print('Association Rules: ', rules.shape)
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
