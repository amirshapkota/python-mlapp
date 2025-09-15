from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import psycopg2
from datetime import datetime
import time

DB_CONFIG = {
    "host": "localhost",
    "dbname": "nepal_stock",
    "user": "postgres",
    "password": "amir",
    "port": 5432
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

def ensure_schema(conn):
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
        company_id INT NOT NULL REFERENCES company(id),
        date DATE NOT NULL,
        open_price NUMERIC,
        high_price NUMERIC,
        low_price NUMERIC,
        close_price NUMERIC,
        volume NUMERIC,
        source TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (company_id, date)
    );
    """)
    conn.commit()

def get_company_id(conn, name, tag):
    cur = conn.cursor()
    cur.execute("SELECT id FROM company WHERE company_tag = %s", (tag,))
    r = cur.fetchone()
    if r:
        return r[0]
    cur.execute("INSERT INTO company (company_name, company_tag) VALUES (%s, %s) RETURNING id", (name, tag))
    cid = cur.fetchone()[0]
    conn.commit()
    return cid

def save_rows(conn, company_id, rows):
    cur = conn.cursor()
    inserted = 0
    for r in rows:
        if r["date"] is None:
            continue
        cur.execute("""
        INSERT INTO price_history
        (company_id, date, open_price, high_price, low_price, close_price, volume, source)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (company_id, date) DO NOTHING
        """, (company_id, r["date"], r["open"], r["high"], r["low"], r["close"], r.get("volume"), "nepalipaisa"))
        inserted += cur.rowcount
    conn.commit()
    return inserted

def parse_date(ds):
    for fmt in ["%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y"]:
        try:
            return datetime.strptime(ds.strip(), fmt).date()
        except:
            continue
    return None

def parse_price(val):
    try:
        return float(val.replace(",", "").strip())
    except:
        return None

def scrape_price_history(symbol, max_rows=20):
    url = f"https://nepalipaisa.com/company/{symbol}"

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=options)
    driver.get(url)

    tab = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, '[data-tab="price-history"]'))
    )

    driver.execute_script("arguments[0].scrollIntoView(true);", tab)
    time.sleep(1)
    driver.execute_script("arguments[0].click();", tab)

    table = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, '#price-history table'))
    )

    # Extract table rows
    rows_data = []
    rows = table.find_elements(By.TAG_NAME, "tr")[1:max_rows+1]  # skip header
    for row in rows:
        cols = [c.text.strip() for c in row.find_elements(By.TAG_NAME, "td")]
        if len(cols) < 5:
            continue
        rows_data.append({
            "date": parse_date(cols[0]),
            "open": parse_price(cols[1]),
            "high": parse_price(cols[2]),
            "low": parse_price(cols[3]),
            "close": parse_price(cols[4]),
            "volume": None
        })

    driver.quit()
    return rows_data

def main():
    conn = connect_db()
    ensure_schema(conn)
    total = 0

    for name, tag in COMPANIES:
        print(f"Scraping {tag} from NepalPaisa...")
        cid = get_company_id(conn, name, tag)
        rows = scrape_price_history(tag, max_rows=20)
        if not rows:
            print(f"No data returned for {tag}")
            continue
        saved = save_rows(conn, cid, rows)
        print(f"Saved {saved} rows for {tag}")
        total += saved

    conn.close()
    print("Done! Total rows saved:", total)

if __name__ == "__main__":
    main()
