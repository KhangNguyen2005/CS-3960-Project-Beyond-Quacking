import duckdb

def get_connection():
    return duckdb.connect("project.duckdb")

def setup():
    con = get_connection()

    con.execute("DROP TABLE IF EXISTS documents")

    con.execute("""
    CREATE TABLE documents (
        id INTEGER,
        title TEXT,
        content TEXT,
        embedding FLOAT[]
    )
    """)

    return con