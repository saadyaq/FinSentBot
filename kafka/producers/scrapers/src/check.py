import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('../data/articles.db')

# Read the table into a DataFrame
df = pd.read_sql_query('SELECT * FROM cleaned_articles', conn)

# Close the connection
conn.close()

# Display the first 5 rows
print(df.head())

# Optional: save as CSV for inspection
df.to_csv('../data/articles_cleaned.csv', index=False)
