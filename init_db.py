import duckdb
import os

DB_NAME = os.getenv("DATABASE_NAME", "nutriscan.duckdb")

def setup_database():
    con = duckdb.connect(DB_NAME)

    print("--- Initializing NutriScan Warehouse ---")

    con.execute('''
        CREATE TABLE IF NOT EXISTS extractions (
            id VARCHAR PRIMARY KEY,
            created_at TIMESTAMP,
            s3_url TEXT,
            raw_json JSON
        )
    ''')

    tables = con.execute("SHOW TABLES").fetchall()
    print(f"✅ Tables: {tables}")

    con.close()

if __name__ == "__main__":
    setup_database()