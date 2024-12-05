import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/kaggle/input/best-selling-books/best-selling-books.csv')
df.head(10)

df.tail(10)
df.shape

df.describe()
df.info()

df.dtypes

sales_stats = df['Approximate sales in millions'].describe()
print("Data analysis of sales statistics:\n", sales_stats)

top_selling_books = df.sort_values('Approximate sales in millions', ascending=False).head(5)
print("Top Selling Books:\n", top_selling_books)


plt.figure(figsize=(8, 6))
plt.hist(df['Approximate sales in millions'], bins=10, edgecolor='black')
plt.xlabel('Sales (millions)')
plt.ylabel('Frequency')
plt.title('Distribution of sales data')
plt.show()


sales_by_genre = df.groupby('Genre')['Approximate sales in millions'].sum().reset_index()

top_10_genres = sales_by_genre.nlargest(10, 'Approximate sales in millions')

plt.figure(figsize=(8, 6))
sns.barplot(data=top_10_genres, x='Genre', y='Approximate sales in millions')
plt.xlabel('Genre')
plt.ylabel('Total Sales (millions)')
plt.title('Top 10 Sales by Genre')
plt.xticks(rotation=90)
plt.show()


sales_by_language = df.groupby('Original language')['Approximate sales in millions'].sum().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(data=sales_by_language, x='Original language', y='Approximate sales in millions')
plt.xlabel('Original Language')
plt.ylabel('Total Sales (millions)')
plt.title('Total Sales by Language of Origin')
plt.xticks(rotation=90)
plt.show()

df['First published'] = pd.to_datetime(df['First published'])
df = df.sort_values('First published')
plt.figure(figsize=(10, 6))
plt.plot(df['First published'], df['Approximate sales in millions'], color= 'red')
plt.xlabel('Year')
plt.ylabel('Sales (millions)')
plt.title('Revenue Trends Over Time')
plt.xticks(rotation=90)
plt.show()

top_authors = df.groupby('Author(s)')['Approximate sales in millions'].sum().nlargest(10)
plt.figure(figsize=(10, 6))
top_authors.plot(kind='bar')
plt.xlabel('Author')
plt.ylabel('Total Sales (millions)')
plt.title('Authors with the Highest Sales')
plt.xticks(rotation=90)
plt.show()