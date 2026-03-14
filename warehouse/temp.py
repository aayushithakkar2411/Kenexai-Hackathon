import sqlite3
db_path = "data/gold/warehouse.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(dim_customer)")
columns = cursor.fetchall()
print(columns)
conn.close()