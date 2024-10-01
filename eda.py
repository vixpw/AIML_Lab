import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the dataset
df = pd.read_csv('data.csv')

# Display the first five rows
print("\nFirst Five Rows:\n", df.head())

# Get a summary of the dataset
print("\nSummary Statistics:\n", df.describe())

# Visualizations
# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Distribution of numerical columns
df.hist(figsize=(10, 10), bins=30)
plt.suptitle('Distribution of Numerical Columns')
plt.show()

# Countplot for categorical columns (if any)
if 'category_column' in df.columns:  # Replace with your actual categorical column name
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='category_column')
    plt.title('Countplot of Category Column')
    plt.xticks(rotation=45)
    plt.show()
