import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from google.colab import files

# Step 1: Upload the CSV file
uploaded = files.upload()
# Note: Upload widget is only available when the cell has been executed in the current browser session.

# Step 2: Read the uploaded CSV file into a DataFrame
df = pd.read_csv(io.BytesIO(uploaded['customers-100.csv']))

# Step 3: Display first five rows
print("First Five Rows:")
print(df.head(5))

# Step 4: Display last five rows
print("\nLast Five Rows:")
print(df.tail(5))

# Step 5: Check for missing values
print("\nMissing Values:")
print(df.isna().sum())

# Step 6: Get basic statistics
print("\nSummary Statistics:")
print(df.describe())

# Step 7: Display the size of the DataFrame
print("\nSize of the DataFrame:", df.size)

# Step 8: Summarize numerical columns
print("\nSum of Numeric Columns:")
print(df.sum(numeric_only=True))
print("\nMean of Numeric Columns:")
print(df.mean(numeric_only=True))
print("\nMinimum Value of Numeric Columns:")
print(df.min(numeric_only=True))
print("\nMaximum Value of Numeric Columns:")
print(df.max(numeric_only=True))

# Step 9: Visualization of statistics

# 9.1 Histogram of numeric columns
plt.figure(figsize=(10, 6))
df.hist(bins=30, edgecolor='black', figsize=(10, 6))
plt.suptitle('Histogram of Numeric Columns', fontsize=16)
plt.show()

# 9.2 Box Plot for each numeric column
plt.figure(figsize=(10, 6))
sns.boxplot(data=df.select_dtypes(include=['float64', 'int64']), orient='h')
plt.title('Box Plot of Numeric Columns')
plt.show()

# 9.3 Bar Plot for sum of numeric columns
plt.figure(figsize=(10, 6))
sum_values = df.sum(numeric_only=True)
sum_values.plot(kind='bar', color='skyblue')
plt.title('Sum of Numeric Columns')
plt.xlabel('Columns')
plt.ylabel('Sum')
plt.xticks(rotation=45)
plt.show()

# 9.4 Bar Plot for average of numeric columns
plt.figure(figsize=(10, 6))
mean_values = df.mean(numeric_only=True)
mean_values.plot(kind='bar', color='lightgreen')
plt.title('Mean of Numeric Columns')
plt.xlabel('Columns')
plt.ylabel('Mean')
plt.xticks(rotation=45)
plt.show()

# 9.5 Heatmap of correlation
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Step 10: Export DataFrame to a new CSV file
df.to_csv('exported_data.csv', index=False)
print("\nData exported to 'exported_data.csv'.")

