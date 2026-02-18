import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import pandas as pd
import time
import os
from datetime import datetime
import tempfile
import platform
import subprocess
import sys

st.set_page_config(
    page_title="Stock Analysis Scraper",
    layout="wide"
)

def find_chromedriver():
    """Find chromedriver in common locations"""
    possible_paths = [
        '/usr/bin/chromedriver',
        '/usr/lib/chromium/chromedriver',
        '/usr/lib/chromium-browser/chromedriver',
        '/snap/bin/chromium.chromedriver',
        '/usr/local/bin/chromedriver'
    ]
    
    # Try common paths
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Try using 'which' command
    try:
        result = subprocess.run(['which', 'chromedriver'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    
    return None

class StockAnalysisScraper:
    def __init__(self):
        self.driver = None
        self.required_columns = ['symbol', 'company_name', 'current_price', 'market_cap', 
                                 'pe_ratio', 'revenue', 'industry']
        
        # Initialize session state for storing scraped data
        if 'scraped_data' not in st.session_state:
            st.session_state.scraped_data = []
        if 'current_csv_filename' not in st.session_state:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.session_state.current_csv_filename = f"stock_data_{timestamp}.csv"
    
    def setup_driver(self):
        """Initialize the Chrome WebDriver with cloud-compatible options"""
        try:
            chrome_options = Options()
            
            # Essential options for cloud environment
            chrome_options.add_argument('--headless=new')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            
            # Set binary location for Chromium
            system = platform.system()
            
            if system == 'Linux':
                # For Streamlit Cloud - try multiple binary locations
                chromium_paths = [
                    '/usr/bin/chromium',
                    '/usr/bin/chromium-browser',
                    '/snap/bin/chromium',
                    '/usr/bin/google-chrome',
                    '/usr/bin/google-chrome-stable'
                ]
                
                binary_found = False
                for path in chromium_paths:
                    if os.path.exists(path):
                        chrome_options.binary_location = path
                        binary_found = True
                        break
                
                if not binary_found:
                    st.error("Chromium/Chrome browser not found. Please ensure it's installed.")
                    return False
                
                # Find chromedriver
                chromedriver_path = find_chromedriver()
                
                if chromedriver_path:
                    try:
                        service = Service(executable_path=chromedriver_path)
                        self.driver = webdriver.Chrome(service=service, options=chrome_options)
                        return True
                    except Exception as e:
                        st.error(f"Error creating driver with found chromedriver: {str(e)}")
                        return False
                else:
                    st.error("ChromeDriver not found. Please ensure chromium-driver is installed.")
                    return False
            
            else:
                # For local Windows/Mac development
                try:
                    from webdriver_manager.chrome import ChromeDriverManager
                    service = Service(ChromeDriverManager().install())
                    self.driver = webdriver.Chrome(service=service, options=chrome_options)
                    return True
                except:
                    # Fallback to selenium-manager
                    self.driver = webdriver.Chrome(options=chrome_options)
                    return True
            
        except Exception as e:
            st.error(f"Error setting up WebDriver: {str(e)}")
            return False

    def load_website_first(self):
        """Load the website"""
        try:    
            self.driver.get("https://stockanalysis.com/")
            wait = WebDriverWait(self.driver, 20)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
            time.sleep(2)
            return True
        except Exception as e:
            st.error(f"Error loading website: {e}")
            return False

    def is_valid_symbol(self, symbol):
        """Check if the stock symbol is valid"""
        try:
            stock_url = f"https://stockanalysis.com/stocks/{symbol.lower()}/"
            self.driver.get(stock_url)
            time.sleep(3)
            
            try:
                company_name = self.driver.find_element(By.TAG_NAME, 'h1').text.strip()
                if company_name and len(company_name) > 0:
                    return True
            except:
                pass
            
            page_text = self.driver.page_source.lower()
            error_indicators = ["404", "page not found", "does not exist", "no results"]
            
            for indicator in error_indicators:
                if indicator in page_text:
                    return False
            
            return False
        except Exception as e:
            return False

    def clean_column_name(self, key):
        """Clean and standardize column names"""
        clean_key = (key.lower()
                   .replace(' ', '_')
                   .replace('-', '_')
                   .replace('(', '')
                   .replace(')', '')
                   .replace('#', 'num')
                   .replace(':', '')
                   .replace('/', '_')
                   .replace('&', 'and')
                   .replace('%', 'percent')
                   .replace('.', ''))
        return clean_key

    def extract_financial_metrics(self):
        """Extract specific financial metrics from tables"""
        metrics = {
            'market_cap': 'N/A',
            'pe_ratio': 'N/A', 
            'revenue': 'N/A'
        }
        
        try:
            tables = self.driver.find_elements(By.CSS_SELECTOR, 'table')
            
            for table in tables:
                rows = table.find_elements(By.CSS_SELECTOR, 'tr')
                for row in rows:
                    cells = row.find_elements(By.CSS_SELECTOR, 'td')
                    if len(cells) >= 2:
                        label = cells[0].text.strip().lower()
                        value = cells[1].text.strip()
                        
                        if 'market cap' in label and metrics['market_cap'] == 'N/A':
                            metrics['market_cap'] = value
                        elif 'pe ratio' in label and metrics['pe_ratio'] == 'N/A':
                            metrics['pe_ratio'] = value
                        elif 'revenue' in label and metrics['revenue'] == 'N/A':
                            metrics['revenue'] = value
            
            return metrics
        except Exception as e:
            return metrics

    def extract_industry(self):
        """Extract industry"""
        try:
            industry = self.driver.find_element(By.CSS_SELECTOR, '.col-span-1 a')
            return industry.text.strip()
        except Exception as e:
            return 'N/A'

    def scrape_stock_data(self, symbol):
        """Scrape all data for a single stock"""
        stock_data = {
            'symbol': symbol,
            'company_name': 'N/A',
            'current_price': 'N/A',
            'market_cap': 'N/A',
            'pe_ratio': 'N/A',
            'revenue': 'N/A',
            'industry': 'N/A'
        }
        
        try:
            stock_url = f"https://stockanalysis.com/stocks/{symbol.lower()}/"
            self.driver.get(stock_url)
            time.sleep(4)
            
            # Extract company name
            try:
                company_name = self.driver.find_element(By.TAG_NAME, 'h1').text.strip()
                stock_data['company_name'] = company_name
            except:
                pass
            
            # Extract current price
            try:
                price = self.driver.find_element(By.XPATH, '//*[@id="main"]/div[1]/div[2]/div[1]/div[1]')
                stock_data['current_price'] = price.text.strip()
            except:
                pass
            
            # Extract industry
            stock_data['industry'] = self.extract_industry()
            
            # Extract financial metrics
            financial_metrics = self.extract_financial_metrics()
            stock_data.update(financial_metrics)
            
            # Extract additional table data
            try:
                tables = self.driver.find_elements(By.CSS_SELECTOR, 'table')
                
                for table in tables:
                    rows = table.find_elements(By.CSS_SELECTOR, 'tr')
                    for row in rows:
                        cells = row.find_elements(By.CSS_SELECTOR, 'td') 
                        if len(cells) >= 2:
                            label = cells[0].text.strip()
                            value = cells[1].text.strip()
                            
                            if label and value:
                                clean_key = self.clean_column_name(label)
                                if clean_key not in self.required_columns:
                                    stock_data[clean_key] = value
            except:
                pass
            
            return stock_data
            
        except Exception as e:
            st.error(f"Error scraping {symbol}: {e}")
            return stock_data

    def save_to_csv(self):
        """Save all scraped data to CSV"""
        try:
            if st.session_state.scraped_data:
                df = pd.DataFrame(st.session_state.scraped_data)
                df.to_csv(st.session_state.current_csv_filename, index=False)
                return True
            return False
        except Exception as e:
            st.error(f"Error saving to CSV: {e}")
            return False

    def close_driver(self):
        """Close the WebDriver"""
        if self.driver:
            self.driver.quit()

def main():
    st.title("Stock Analysis Scraper")
    
    # Initialize scraper in session state
    if 'scraper' not in st.session_state:
        st.session_state.scraper = StockAnalysisScraper()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Scrape Stock Data", "View Scraped Data", "Export Data"])
    
    # Tab 1: Scrape Stock Data
    with tab1:
        st.header("Scrape Stock Data")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            symbol = st.text_input("Enter Stock Symbol", placeholder="e.g., AAPL, TSLA, MSFT").strip().upper()
        
        with col2:
            st.write("")
            st.write("")
            scrape_button = st.button("Scrape Stock", type="primary")
        
        if scrape_button:
            if not symbol:
                st.warning("Please enter a stock symbol.")
            else:
                with st.spinner(f"Scraping data for {symbol}..."):
                    if st.session_state.scraper.setup_driver():
                        if st.session_state.scraper.load_website_first():
                            if st.session_state.scraper.is_valid_symbol(symbol):
                                stock_data = st.session_state.scraper.scrape_stock_data(symbol)
                                st.session_state.scraped_data.append(stock_data)
                                
                                if st.session_state.scraper.save_to_csv():
                                    st.success(f"Successfully scraped data for {symbol}")
                                    
                                    # Display scraped data
                                    st.subheader("Scraped Data")
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write(f"**Company:** {stock_data['company_name']}")
                                        st.write(f"**Symbol:** {stock_data['symbol']}")
                                        st.write(f"**Current Price:** {stock_data['current_price']}")
                                        st.write(f"**Industry:** {stock_data['industry']}")
                                    
                                    with col2:
                                        st.write(f"**Market Cap:** {stock_data['market_cap']}")
                                        st.write(f"**P/E Ratio:** {stock_data['pe_ratio']}")
                                        st.write(f"**Revenue:** {stock_data['revenue']}")
                                    
                                    # Show additional data
                                    additional_keys = [k for k in stock_data.keys() 
                                                     if k not in st.session_state.scraper.required_columns]
                                    if additional_keys:
                                        st.subheader("Additional Data")
                                        for key in additional_keys:
                                            st.write(f"**{key.replace('_', ' ').title()}:** {stock_data[key]}")
                                else:
                                    st.error("Failed to save data to CSV")
                            else:
                                st.error(f"'{symbol}' is not a valid stock symbol.")
                        else:
                            st.error("Failed to load website.")
                        
                        st.session_state.scraper.close_driver()
                    else:
                        st.error("Failed to initialize WebDriver. Please check if Chromium is installed.")
        
        st.info(f"Data will be saved to: {st.session_state.current_csv_filename}")
    
    # Tab 2: View Scraped Data
    with tab2:
        st.header("View Scraped Data")
        
        if not st.session_state.scraped_data:
            st.info("No data has been scraped yet. Please scrape some stocks first.")
        else:
            st.subheader("Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Stocks", len(st.session_state.scraped_data))
            
            with col2:
                all_columns = set()
                for data in st.session_state.scraped_data:
                    all_columns.update(data.keys())
                st.metric("Total Data Fields", len(all_columns))
            
            with col3:
                symbols = [data['symbol'] for data in st.session_state.scraped_data]
                st.metric("Symbols", ", ".join(symbols))
            
            st.subheader("Data Table")
            df = pd.DataFrame(st.session_state.scraped_data)
            st.dataframe(df, use_container_width=True)
    
    # Tab 3: Export Data
    with tab3:
        st.header("Export Data")
        
        if not st.session_state.scraped_data:
            st.info("No data to export. Please scrape some stocks first.")
        else:
            st.subheader("Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                df = pd.DataFrame(st.session_state.scraped_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name=st.session_state.current_csv_filename,
                    mime="text/csv"
                )
            
            with col2:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                        df.to_excel(tmp.name, index=False)
                        with open(tmp.name, 'rb') as f:
                            excel_data = f.read()
                    
                    st.download_button(
                        label="Download as Excel",
                        data=excel_data,
                        file_name=st.session_state.current_csv_filename.replace('.csv', '.xlsx'),
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.warning("Excel export unavailable. Using CSV only.")
            
            if st.button("Clear All Scraped Data"):
                st.session_state.scraped_data = []
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                st.session_state.current_csv_filename = f"stock_data_{timestamp}.csv"
                st.success("All data cleared successfully!")
                st.rerun()

if __name__ == "__main__":
    main()
