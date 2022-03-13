import sqlite3

import pandas as pd

conn = sqlite3.connect('project_database\wafer.db')

cursor = conn.cursor()

with conn:
    cursor.execute(
        f"""SELECT * from train""")
    res = cursor.fetchall()
    df = pd.DataFrame(res)
    print(df)


conn.close()
