import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

# Load dataset
dataset = pd.read_csv("D:\\UberDataset.csv")

# Fill missing PURPOSE values
dataset['PURPOSE'] = dataset['PURPOSE'].fillna("NOT")


# Convert date columns to datetime
dataset['START_DATE'] = pd.to_datetime(dataset['START_DATE'], errors='coerce')
dataset['END_DATE'] = pd.to_datetime(dataset['END_DATE'], errors='coerce')

# Extract date and hour
dataset['date'] = pd.DatetimeIndex(dataset['START_DATE']).date
dataset['time'] = pd.DatetimeIndex(dataset['START_DATE']).hour

# Create time of day categories
dataset['day-night'] = pd.cut(
    x=dataset['time'],
    bins=[0, 10, 15, 19, 24],
    labels=['Morning', 'Afternoon', 'Evening', 'Night']
)

# Clean data
dataset.dropna(inplace=True)
dataset.drop_duplicates(inplace=True)

# Categorical column identification
obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)

# Unique values count (optional for analysis)
unique_values = {}
for col in object_cols:
    unique_values[col] = dataset[col].unique().size

# Plot category and purpose counts
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.countplot(data=dataset, x='CATEGORY')
plt.xticks(rotation=90)

plt.subplot(1, 2, 2)
sns.countplot(data=dataset, x='PURPOSE')
plt.xticks(rotation=90)

sns.countplot(data=dataset, x='day-night')
plt.xticks(rotation=90)

plt.figure(figsize=(15, 5))
sns.countplot(data=dataset, x='PURPOSE', hue='CATEGORY')
plt.xticks(rotation=90)
plt.show()

# Apply OneHotEncoding
object_cols = ['CATEGORY', 'PURPOSE']
OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
OH_cols = pd.DataFrame(OH_encoder.fit_transform(dataset[object_cols]))
OH_cols.index = dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out(object_cols)

# Merge encoded columns
df_final = dataset.drop(object_cols, axis=1)
dataset = pd.concat([df_final, OH_cols], axis=1)

# Correlation Heatmap
numeric_dataset = dataset.select_dtypes(include=['number'])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_dataset.corr(), cmap='BrBG', fmt='.2f', linewidths=2, annot=True)
plt.show()

# Month column and mapping
dataset['MONTH'] = pd.DatetimeIndex(dataset['START_DATE']).month
month_label = {
    1.0: 'Jan', 2.0: 'Feb', 3.0: 'Mar', 4.0: 'April',
    5.0: 'May', 6.0: 'June', 7.0: 'July', 8.0: 'Aug',
    9.0: 'Sep', 10.0: 'Oct', 11.0: 'Nov', 12.0: 'Dec'
}
dataset["MONTH"] = dataset.MONTH.map(month_label)

# Monthly analysis
mon = dataset.MONTH.value_counts(sort=False)
df = pd.DataFrame({
    "MONTHS": mon.index,
    "VALUE COUNT": dataset.groupby('MONTH', sort=False)['MILES'].max().values
})
p = sns.lineplot(data=df, x="MONTHS", y="VALUE COUNT")
p.set(xlabel="MONTHS", ylabel="VALUE COUNT")
plt.show()

# Day of week analysis
dataset['DAY'] = dataset.START_DATE.dt.weekday
day_label = {
    0: 'Mon', 1: 'Tues', 2: 'Wed', 3: 'Thurs', 4: 'Fri', 5: 'Sat', 6: 'Sun'
}
dataset['DAY'] = dataset['DAY'].map(day_label)
day_counts = dataset.DAY.value_counts()
sns.barplot(x=day_counts.index, y=day_counts.values)
plt.xlabel('DAY')
plt.ylabel('COUNT')
plt.show()

# Box and distribution plots for MILES
sns.boxplot(x=dataset['MILES'])
plt.title("Miles Boxplot")
plt.show()

sns.boxplot(x=dataset[dataset['MILES'] < 100]['MILES'])
plt.title("Miles Boxplot (< 100 miles)")
plt.show()

sns.histplot(dataset[dataset['MILES'] < 40]['MILES'], kde=True, bins=30)
plt.title("Distribution of Miles (< 40 miles)")
plt.show()