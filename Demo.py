import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

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

bins = [-1, 0, 10, 20, np.inf]
labels = ['Zero', 'Low', 'Medium', 'High']
dataframe['Unvaccinated Rate'] = pd.cut(dataframe['Unvaccinated Rate'], bins=bins, labels=labels)
dataframe['Vaccinated Rate'] = pd.cut(dataframe['Vaccinated Rate'], bins=bins, labels=labels)
dataframe['Boosted Rate'] = pd.cut(dataframe['Boosted Rate'], bins=bins, labels=labels)

dataframe['Outcome_Age'] = dataframe['Outcome'] + '_' + dataframe['Age Group']

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
rules = rules[rules['antecedents'].apply(lambda x: any('Deaths' in item or 'Hospitalizations' in item for item in x))]
rules = rules[rules['consequents'].apply(lambda x: len(x) >= 1)]
print('Association Rules: ', rules.shape)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
