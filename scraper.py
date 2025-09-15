import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import pandas as pd
import psycopg2
import re
import random

DB_CONFIG = {
    "host": "localhost",
    "dbname": "nepal_stock",
    "user": "postgres",
    "password": "amir",
    "port": 5432
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
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
        close_price NUMERIC,
        volume NUMERIC,
        source TEXT,
        UNIQUE(company_id, date)
    );
    """)
    
    # Add missing columns if they don't exist
    try:
        cur.execute("ALTER TABLE price_history ADD COLUMN IF NOT EXISTS volume NUMERIC;")
        cur.execute("ALTER TABLE price_history ADD COLUMN IF NOT EXISTS source TEXT;")
    except Exception as e:
        print(f"Schema update note: {e}")
    
    conn.commit()
    conn.close()

def parse_price(price_str):
    if not price_str:
        return None
    price_str = str(price_str).strip().replace(',', '').replace('—', '').replace('-', '')
    if not price_str or price_str in ['-', '—', '--']:
        return None
    try:
        return float(price_str)
    except ValueError:
        return None

def parse_date(date_str):
    if not date_str:
        return None
    date_str = str(date_str).strip()
    date_formats = ["%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y"]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None

def get_company_id(conn, name, tag):
    cur = conn.cursor()
    cur.execute("SELECT id FROM company WHERE company_tag = %s", (tag,))
    result = cur.fetchone()
    
    if result:
        return result[0]
    
    cur.execute("INSERT INTO company (company_name, company_tag) VALUES (%s, %s) RETURNING id", (name, tag))
    company_id = cur.fetchone()[0]
    conn.commit()
    return company_id

def insert_price_history(conn, company_id, rows):
    if not rows:
        return 0
    
    cur = conn.cursor()
    inserted = 0
    
    for row in rows:
        try:
            cur.execute("""
                INSERT INTO price_history 
                (company_id, date, open_price, high_price, low_price, close_price, volume, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (company_id, date) DO NOTHING
            """, (
                company_id, row.get('date'), row.get('open'), row.get('high'), 
                row.get('low'), row.get('close'), row.get('volume'), row.get('source', 'sharesansar')
            ))
            
            if cur.rowcount > 0:
                inserted += 1
                
        except Exception as e:
            print(f"Insert error: {e}")
            continue
    
    conn.commit()
    return inserted

def scrape_sharesansar_selenium(symbol, max_rows=25):
    """Scrape ShareSansar using Selenium"""
    results = []
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(options=chrome_options)
        
        try:
            url = f"https://www.sharesansar.com/company/{symbol.lower()}"
            print(f"Loading {url} with browser...")
            
            driver.get(url)
            time.sleep(2)
            
            # Click Price History tab
            try:
                price_tab = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.ID, "btn_cpricehistory"))
                )
                price_tab.click()
                print("Clicked Price History tab")
                time.sleep(3)
            except:
                print("Could not find Price History tab")
                return []
            
            # Get the price history table
            try:
                table = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "myTableCPriceHistory"))
                )
                
                rows = table.find_elements(By.TAG_NAME, "tr")
                print(f"Found {len(rows)-1} data rows")
                
                # Process each row (skip header)
                for i, row in enumerate(rows[1:max_rows+1]):
                    try:
                        cells = row.find_elements(By.TAG_NAME, "td")
                        
                        if len(cells) >= 6:
                            # Columns
                            date_val = parse_date(cells[1].text.strip())
                            
                            if date_val:
                                price_data = {
                                    'date': date_val,
                                    'open': parse_price(cells[2].text),
                                    'high': parse_price(cells[3].text),
                                    'low': parse_price(cells[4].text),
                                    'close': parse_price(cells[5].text),  # LTP = Close
                                    'volume': parse_price(cells[7].text) if len(cells) > 7 else None,
                                    'source': 'sharesansar'
                                }
                                results.append(price_data)
                                
                    except Exception as e:
                        print(f"Error parsing row {i+1}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Could not find price table: {e}")
                
        finally:
            driver.quit()
            
    except ImportError:
        print("Selenium not installed. Install with: pip install selenium")
        return []
    except Exception as e:
        print(f"Selenium error: {e}")
    
    return results

def scrape_sharesansar_simple(symbol, max_rows=25):
    """Simple ShareSansar scraper without Selenium"""
    results = []
    
    try:
        url = f"https://www.sharesansar.com/company/{symbol.lower()}"
        print(f"Trying simple method: {url}")
        
        response = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for price history table
        table = soup.find('table', {'id': 'myTableCPriceHistory'})
        
        if table:
            rows = table.find_all('tr')
            print(f"Found table with {len(rows)-1} rows")
            
            for row in rows[1:max_rows+1]:  # Skip header
                cols = [td.get_text(strip=True) for td in row.find_all('td')]
                
                if len(cols) >= 6:
                    date_val = parse_date(cols[1])  # Date column
                    
                    if date_val:
                        price_data = {
                            'date': date_val,
                            'open': parse_price(cols[2]),
                            'high': parse_price(cols[3]),
                            'low': parse_price(cols[4]),
                            'close': parse_price(cols[5]),
                            'volume': parse_price(cols[7]) if len(cols) > 7 else None,
                            'source': 'sharesansar'
                        }
                        results.append(price_data)
        else:
            print("Price history table not found in HTML")
            
    except Exception as e:
        print(f"Simple scraping error: {e}")
    
    return results

def scrape_company(symbol, company_name, target_records=20):
    """Scrape data for one company"""
    print(f"\nScraping {company_name} ({symbol})")
    print("-" * 40)
    
    results = scrape_sharesansar_selenium(symbol, target_records)
    
    # Try simple method
    if len(results) < 5:
        print("Trying simple method...")
        results = scrape_sharesansar_simple(symbol, target_records)
    
    # If no data, create sample data
    if len(results) == 0:
        print("No real data found, creating sample data...")
        from datetime import timedelta
        
        base_date = datetime.now().date()
        base_price = random.uniform(500, 2000)
        
        for i in range(target_records):
            date = base_date - timedelta(days=i)
            
            if date.weekday() >= 5:
                continue
                
            change = random.uniform(-0.05, 0.05)
            open_price = base_price * (1 + change)
            high_price = open_price * (1 + random.uniform(0, 0.03))
            low_price = open_price * (1 - random.uniform(0, 0.03))
            close_price = random.uniform(low_price, high_price)
            
            results.append({
                'date': date,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': random.randint(1000, 50000),
                'source': 'generated'
            })
            
            base_price = close_price
    
    return results[:target_records]

def main():
    
    ensure_schema()
    conn = connect_db()
    total_saved = 0
    
    for company_name, symbol in COMPANIES:
        try:
            company_id = get_company_id(conn, company_name, symbol)
            
            # Scrape data
            price_data = scrape_company(symbol, company_name, target_records=25)
            
            if price_data:
                # Save to database
                saved_count = insert_price_history(conn, company_id, price_data)
                
                # Save to CSV
                df = pd.DataFrame(price_data)
                csv_file = f"{symbol}_history.csv"
                df.to_csv(csv_file, index=False)
                
                print(f"Saved {saved_count} records to database")
                print(f"Saved to {csv_file}")
                print(f"Latest: {price_data[0]['date']} - Close: {price_data[0]['close']}")
                
                total_saved += saved_count
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
        
        time.sleep(2)
    
    conn.close()
    
    print(f"Total records saved: {total_saved}")

if __name__ == "__main__":
    main()