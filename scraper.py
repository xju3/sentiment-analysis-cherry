import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import json
import logging
import os

class ChineseSocialMediaScraper:
    def __init__(self):
        self.setup_logging()
        self.load_config()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='scraping.log'
        )
        
    def load_config(self):
        # Load API keys and credentials from environment variables
        self.weibo_api_key = os.getenv('WEIBO_API_KEY')
        self.weibo_api_secret = os.getenv('WEIBO_API_SECRET')
        self.xiaohongshu_token = os.getenv('XIAOHONGSHU_TOKEN')
        
    def get_weibo_access_token(self):
        """Get Weibo access token using API credentials"""
        url = "https://api.weibo.com/oauth2/access_token"
        data = {
            "client_id": self.weibo_api_key,
            "client_secret": self.weibo_api_secret,
            "grant_type": "client_credentials"
        }
        response = requests.post(url, data=data)
        return response.json().get('access_token')

    def search_weibo_api(self, keyword, count=100, page=1):
        """Search Weibo using official API"""
        access_token = self.get_weibo_access_token()
        url = "https://api.weibo.com/2/search/topics.json"
        params = {
            "access_token": access_token,
            "q": keyword,
            "count": count,
            "page": page
        }
        try:
            response = requests.get(url, params=params)
            return response.json()
        except Exception as e:
            logging.error(f"Error searching Weibo API: {str(e)}")
            return None

    def scrape_weibo_selenium(self, keyword, pages=5):
        """Scrape Weibo using Selenium (as backup method)"""
        driver = webdriver.Chrome()  # Make sure to have ChromeDriver installed
        posts = []
        
        try:
            # Login page (you'll need to implement login logic)
            driver.get("https://weibo.com/login.php")
            time.sleep(5)  # Allow time for manual login if needed
            
            # Search for keyword
            search_url = f"https://s.weibo.com/weibo?q={keyword}"
            driver.get(search_url)
            
            for page in range(pages):
                # Wait for posts to load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "card-feed"))
                )
                
                # Extract posts
                post_elements = driver.find_elements(By.CLASS_NAME, "card-feed")
                
                for post in post_elements:
                    try:
                        content = post.find_element(By.CLASS_NAME, "txt").text
                        timestamp = post.find_element(By.CLASS_NAME, "from").text
                        user = post.find_element(By.CLASS_NAME, "name").text
                        
                        posts.append({
                            "platform": "weibo",
                            "content": content,
                            "timestamp": timestamp,
                            "user": user,
                            "keyword": keyword
                        })
                    except Exception as e:
                        logging.error(f"Error extracting post: {str(e)}")
                
                # Click next page
                next_button = driver.find_element(By.CLASS_NAME, "next")
                next_button.click()
                time.sleep(3)
                
        except Exception as e:
            logging.error(f"Error in Selenium scraping: {str(e)}")
        finally:
            driver.quit()
            
        return posts

    def scrape_xiaohongshu(self, keyword, num_posts=100):
        """Scrape Xiaohongshu (Red Book) data"""
        headers = {
            "Authorization": f"Bearer {self.xiaohongshu_token}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        posts = []
        url = f"https://www.xiaohongshu.com/search_result/{keyword}"
        
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract posts (implementation depends on website structure)
            post_elements = soup.find_all("div", class_="note-item")
            
            for post in post_elements[:num_posts]:
                content = post.find("div", class_="content").text
                timestamp = post.find("div", class_="time").text
                user = post.find("div", class_="username").text
                
                posts.append({
                    "platform": "xiaohongshu",
                    "content": content,
                    "timestamp": timestamp,
                    "user": user,
                    "keyword": keyword
                })
                
        except Exception as e:
            logging.error(f"Error scraping Xiaohongshu: {str(e)}")
            
        return posts

    def collect_data(self, keyword, platforms=['weibo', 'xiaohongshu'], days_back=7):
        """Collect data from multiple platforms"""
        all_data = []
        
        for platform in platforms:
            try:
                if platform == 'weibo':
                    # Try API first, fall back to Selenium
                    data = self.search_weibo_api(keyword)
                    if not data:
                        data = self.scrape_weibo_selenium(keyword)
                    all_data.extend(data)
                    
                elif platform == 'xiaohongshu':
                    data = self.scrape_xiaohongshu(keyword)
                    all_data.extend(data)
                    
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                logging.error(f"Error collecting data from {platform}: {str(e)}")
                
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Basic cleaning
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df[df['timestamp'] >= datetime.now() - timedelta(days=days_back)]
        
        return df

    def save_data(self, df, filename):
        """Save collected data to file"""
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        logging.info(f"Data saved to {filename}")

def main():
    # Example usage
    scraper = ChineseSocialMediaScraper()
    
    # Set your search parameters
    keyword = "你的品牌名称"  # Your brand name
    platforms = ['weibo', 'xiaohongshu']
    days_back = 30
    
    # Collect data
    data = scraper.collect_data(
        keyword=keyword,
        platforms=platforms,
        days_back=days_back
    )
    
    # Save data
    scraper.save_data(data, f"social_media_data_{datetime.now().strftime('%Y%m%d')}.csv")
    
    return data

if __name__ == "__main__":
    data = main()