import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Scraper",
    layout="wide"
)

class StockAnalysisScraper:
    def __init__(self):
        self.driver = None
        self.required_columns = ['symbol', 'company_name', 'current_price', 'market_cap', 
                                 'pe_ratio', 'revenue', 'industry']
        self.additional_columns = set()
        
        # Initialize session state for storing scraped data
        if 'scraped_data' not in st.session_state:
            st.session_state.scraped_data = []
        if 'current_csv_filename' not in st.session_state:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.session_state.current_csv_filename = f"stock_data_{timestamp}.csv"
    
    def setup_driver(self):
        """Initialize the Chrome WebDriver"""
        try:
            options = webdriver.ChromeOptions()
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            
            self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            return True
        except Exception as e:
            st.error(f"Error setting up WebDriver: {e}")
            return False

    def load_website_first(self):
        """Load the website FIRST before any user interaction"""
        try:    
            self.driver.get("https://stockanalysis.com/")
            
            # Wait for the page to load completely
            wait = WebDriverWait(self.driver, 20)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
            
            # Additional wait to ensure everything is loaded
            time.sleep(2)
            
            return True
            
        except Exception as e:
            st.error(f"Error loading website: {e}")
            return False

    def is_valid_symbol(self, symbol):
        """Simple check if the stock symbol is valid by looking for company name"""
        try:
            # Navigate to the stock's page
            stock_url = f"https://stockanalysis.com/stocks/{symbol.lower()}/"
            self.driver.get(stock_url)
            
            # Wait for page to load
            time.sleep(3)
            
            # Try to find the company name - if we can find it, the symbol is valid
            try:
                company_name = self.driver.find_element(By.TAG_NAME, 'h1').text.strip()
                # If we found a company name and it's not empty, it's valid
                if company_name and len(company_name) > 0:
                    return True
            except:
                pass
            
            # If company name cannot be found, check for clear error messages
            page_text = self.driver.page_source.lower()
            error_indicators = ["404", "page not found", "does not exist", "no results"]
            
            for indicator in error_indicators:
                if indicator in page_text:
                    return False
            
            # If no clear errors but no company name either, assume invalid
            return False
            
        except Exception as e:
            st.error(f"Error checking symbol validity: {e}")
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
            # Find all tables on the page
            tables = self.driver.find_elements(By.CSS_SELECTOR, 'table')
            
            for table in tables:
                rows = table.find_elements(By.CSS_SELECTOR, 'tr')
                for row in rows:
                    cells = row.find_elements(By.CSS_SELECTOR, 'td')
                    if len(cells) >= 2:
                        label = cells[0].text.strip().lower()
                        value = cells[1].text.strip()
                        
                        # Extract market cap
                        if 'market cap' in label and metrics['market_cap'] == 'N/A':
                            metrics['market_cap'] = value
                        
                        # Extract P/E ratio
                        elif 'pe ratio' in label and metrics['pe_ratio'] == 'N/A':
                            metrics['pe_ratio'] = value
                        
                        # Extract revenue
                        elif 'revenue' in label and metrics['revenue'] == 'N/A':
                            metrics['revenue'] = value
            
            return metrics
            
        except Exception as e:
            st.error(f"Error extracting financial metrics: {e}")
            return metrics

    def extract_industry(self):
        """Extract industry using the CSS selector"""
        try:
            industry = self.driver.find_element(By.CSS_SELECTOR, '.col-span-1 a')
            industry_text = industry.text.strip()
            return industry_text
        except Exception as e:
            return 'N/A'

    def scrape_stock_data(self, symbol):
        """Scrape all data for a single stock and return as dictionary"""
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
            # Navigate to the stock's page (we already validated it's valid)
            stock_url = f"https://stockanalysis.com/stocks/{symbol.lower()}/"
            self.driver.get(stock_url)
            
            # Wait for page to load
            time.sleep(4)
            
            # 1. EXTRACT COMPANY NAME
            try:
                company_name = self.driver.find_element(By.TAG_NAME, 'h1').text.strip()
                stock_data['company_name'] = company_name
            except Exception as e:
                pass
            
            # 2. EXTRACT CURRENT PRICE
            try:
                price = self.driver.find_element(By.XPATH, '//*[@id="main"]/div[1]/div[2]/div[1]/div[1]')
                stock_data['current_price'] = price.text.strip()
            except Exception as e:
                pass
            
            # 3. EXTRACT INDUSTRY
            stock_data['industry'] = self.extract_industry()
            
            # 4. EXTRACT FINANCIAL METRICS FROM TABLES
            financial_metrics = self.extract_financial_metrics()
            stock_data.update(financial_metrics)
            
            # 5. EXTRACT ADDITIONAL TABLE DATA
            try:
                tables = self.driver.find_elements(By.CSS_SELECTOR, 'table')
                additional_data_count = 0

                for table_index, table in enumerate(tables, 1):
                    rows = table.find_elements(By.CSS_SELECTOR, 'tr')
           
                    for row in rows:
                        cells = row.find_elements(By.CSS_SELECTOR, 'td') 
                        if len(cells) >= 2:
                            label = cells[0].text.strip()
                            value = cells[1].text.strip()
                            
                            if label and value:
                                clean_key = self.clean_column_name(label)
                                
                                # Only add if it's not one of the required columns
                                if clean_key not in self.required_columns:
                                    stock_data[clean_key] = value
                                    self.additional_columns.add(clean_key)
                                    additional_data_count += 1
                
            except Exception as e:
                pass
            
            return stock_data
            
        except Exception as e:
            st.error(f"Error scraping {symbol}: {e}")
            return stock_data

    def save_to_csv(self, stock_data):
        """Save stock data to CSV file"""
        try:
            # Convert session data to DataFrame
            if st.session_state.scraped_data:
                df = pd.DataFrame(st.session_state.scraped_data)
                # Reorder columns to put required columns first
                all_columns = self.required_columns + [col for col in df.columns if col not in self.required_columns]
                df = df[all_columns]
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
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Scrape Stock Data", "View Scraped Data", "Export Data"])
    
    # Tab 1: Scrape Stock Data
    with tab1:
        st.header("Scrape Stock Data")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Input for stock symbol
            symbol = st.text_input("Enter Stock Symbol", placeholder="e.g., AAPL, TSLA, MSFT").strip().upper()
        
        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            scrape_button = st.button("Scrape Stock", type="primary")
        
        if scrape_button:
            if not symbol:
                st.warning("Please enter a stock symbol.")
            else:
                with st.spinner(f"Scraping data for {symbol}..."):
                    # Setup driver
                    if st.session_state.scraper.setup_driver():
                        # Load website
                        if st.session_state.scraper.load_website_first():
                            # Check if symbol is valid
                            if st.session_state.scraper.is_valid_symbol(symbol):
                                # Scrape data
                                stock_data = st.session_state.scraper.scrape_stock_data(symbol)
                                
                                # Add to session state
                                st.session_state.scraped_data.append(stock_data)
                                
                                # Save to CSV
                                if st.session_state.scraper.save_to_csv(stock_data):
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
                                    
                                    # Show additional data if any
                                    additional_keys = [k for k in stock_data.keys() if k not in st.session_state.scraper.required_columns]
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
                        
                        # Close driver
                        st.session_state.scraper.close_driver()
                    else:
                        st.error("Failed to initialize WebDriver.")
        
        # Display current CSV filename
        st.info(f"Data will be saved to: {st.session_state.current_csv_filename}")
    
    # Tab 2: View Scraped Data
    with tab2:
        st.header("View Scraped Data")
        
        if not st.session_state.scraped_data:
            st.info("No data has been scraped yet. Please scrape some stocks first.")
        else:
            # Display summary statistics
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
            
            # Display data in table format
            st.subheader("Data Table")
            df = pd.DataFrame(st.session_state.scraped_data)
            
            # Reorder columns to show required columns first
            required_cols = [col for col in st.session_state.scraper.required_columns if col in df.columns]
            other_cols = [col for col in df.columns if col not in required_cols]
            df = df[required_cols + other_cols]
            
            st.dataframe(df, use_container_width=True)
    
    # Tab 3: Export Data
    with tab3:
        st.header("Export Data")
        
        if not st.session_state.scraped_data:
            st.info("No data to export. Please scrape some stocks first.")
        else:
            # Export options
            st.subheader("Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download as CSV
                df = pd.DataFrame(st.session_state.scraped_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name=st.session_state.current_csv_filename,
                    mime="text/csv"
                )
            
            with col2:
                # Download as Excel
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
                    st.warning("Excel export requires openpyxl. Install with: pip install openpyxl")
            
            # Clear data button
            if st.button("Clear All Scraped Data"):
                st.session_state.scraped_data = []
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                st.session_state.current_csv_filename = f"stock_data_{timestamp}.csv"
                st.success("All data cleared successfully!")
                st.rerun()

if __name__ == "__main__":
    main()
