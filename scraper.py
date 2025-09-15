import requests
from bs4 import BeautifulSoup
import csv
import time
from datetime import datetime
import pandas as pd
import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "dbname": "nepal_stock",
    "user": "postgres",
    "password": "amir",
    "port": 5432
}

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

COMPANIES = [
    ("NIC Asia Bank Limited", "NICA"),
    ("Unilever Nepal Limited", "UNL"),
    ("Nepal Telecom", "NTC"),
    ("Chilime Hydropower Company Limited", "CHCL"),
    ("Bottlers Nepal Limited", "BNL")
]

def connect_db():
    return psycopg2.connect(**DB_CONFIG)

def ensure_schema():
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS company (
        id SERIAL PRIMARY KEY,
        company_name TEXT NOT NULL,
        company_tag TEXT NOT NULL UNIQUE
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS price_history (
        id SERIAL PRIMARY KEY,
        company_id INT NOT NULL REFERENCES company(id) ON DELETE CASCADE,
        date DATE NOT NULL,
        open_price NUMERIC,
        high_price NUMERIC,
        low_price NUMERIC,
        close_price NUMERIC
    );
    """)
    conn.commit()
    conn.close()

def parse_price(p):
    if not p:
        return None
    p = p.strip().replace(",", "").replace("—", "").replace("-", "")
    return float(p) if p else None

def parse_date(ds):
    ds = ds.strip()
    for fmt in ["%Y/%m/%d", "%d/%m/%Y", "%Y-%m-%d", "%d-%b-%Y", "%d %b %Y"]:
        try:
            return datetime.strptime(ds, fmt).date()
        except:
            pass
    return None

def get_company_id(conn, name, tag):
    cur = conn.cursor()
    cur.execute("SELECT id FROM company WHERE company_tag = %s", (tag,))
    r = cur.fetchone()
    if r:
        return r[0]
    cur.execute("INSERT INTO company (company_name, company_tag) VALUES (%s, %s) RETURNING id", (name, tag))
    company_id = cur.fetchone()[0]
    conn.commit()
    return company_id

def insert_price_history(conn, company_id, rows):
    cur = conn.cursor()
    inserted = 0
    for r in rows:
        try:
            cur.execute("""
            INSERT INTO price_history
            (company_id, date, open_price, high_price, low_price, close_price)
            VALUES (%s,%s,%s,%s,%s,%s)
            ON CONFLICT (company_id, date) DO NOTHING
            """, (company_id, r["date"], r["open"], r["high"], r["low"], r["close"]))
            inserted += cur.rowcount
        except Exception as e:
            print("Insert error:", e)
    conn.commit()
    return inserted


def main():
    ensure_schema()
    conn = connect_db()

if __name__ == "__main__":
    main()