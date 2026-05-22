# =============================================================================
# FAKE JOB DETECTOR — WEB SCRAPER
# File: scraper.py
# Author: Divya Virkud
# Version: 2.0 | March 2026
# Description: Scrape real Indian job postings from Naukri and Indeed
# =============================================================================
# INSTALL REQUIREMENTS:
#   pip install requests beautifulsoup4 pandas lxml
# =============================================================================
# RUN: python scraper.py
# =============================================================================

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# Headers to mimic real browser (avoids bot detection)
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
}

# Job search keywords
SEARCH_KEYWORDS = [
    'data+analyst',
    'data+scientist',
    'business+analyst',
    'machine+learning',
    'python+developer'
]

# =============================================================================
# SCRAPER 1: NAUKRI.COM (BeautifulSoup - Static Scraping)
# =============================================================================

def scrape_naukri(keyword, max_pages=3):
    """
    Scrape job postings from Naukri.com for a given keyword.
    
    Args:
        keyword: search term (e.g., 'data+analyst')
        max_pages: number of pages to scrape (each page = ~20 jobs)
    
    Returns:
        List of job dictionaries
    """
    print(f"\n{'='*60}")
    print(f"SCRAPING NAUKRI.COM — Keyword: {keyword.replace('+', ' ')}")
    print(f"{'='*60}")
    
    jobs = []
    
    for page in range(1, max_pages + 1):
        # Naukri URL structure
        url = f"https://www.naukri.com/{keyword}-jobs-{page}"
        
        print(f"  Page {page}/{max_pages}...", end=" ")
        
        try:
            # Send request
            response = requests.get(url, headers=HEADERS, timeout=10)
            
            if response.status_code != 200:
                print(f"Failed (Status {response.status_code})")
                continue
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Find all job cards (Naukri's structure as of 2026)
            # Note: HTML structure may change - inspect element to verify
            job_cards = soup.find_all('article', class_='jobTuple')
            
            if not job_cards:
                print("No jobs found (site structure may have changed)")
                break
            
            # Extract data from each job card
            for card in job_cards:
                try:
                    # Job title
                    title_tag = card.find('a', class_='title')
                    title = title_tag.text.strip() if title_tag else 'Not specified'
                    
                    # Company name
                    company_tag = card.find('a', class_='subTitle')
                    company = company_tag.text.strip() if company_tag else 'Not specified'
                    
                    # Experience required
                    exp_tag = card.find('span', class_='expwdth')
                    experience = exp_tag.text.strip() if exp_tag else 'Not specified'
                    
                    # Salary
                    salary_tag = card.find('span', class_='salary')
                    salary = salary_tag.text.strip() if salary_tag else 'Not disclosed'
                    
                    # Location
                    loc_tag = card.find('span', class_='locWdth')
                    location = loc_tag.text.strip() if loc_tag else 'Not specified'
                    
                    # Job description (snippet)
                    desc_tag = card.find('div', class_='jobDescription')
                    description = desc_tag.text.strip() if desc_tag else 'Not available'
                    
                    # Job URL
                    job_url = title_tag['href'] if title_tag and 'href' in title_tag.attrs else ''
                    
                    # Create job dictionary
                    job = {
                        'title': title,
                        'company': company,
                        'location': location,
                        'description': description,
                        'requirements': 'Not available',  # Naukri listings show minimal details
                        'benefits': 'Not available',
                        'employment_type': 'Full-time',   # Default assumption
                        'required_experience': experience,
                        'salary_range': salary,
                        'company_profile': 'Not available',
                        'has_company_logo': 0,
                        'has_questions': 0,
                        'telecommuting': 0,
                        'fraudulent': 0,  # Unknown - manual labeling needed
                        'source': 'Naukri.com',
                        'url': job_url,
                        'scraped_date': datetime.now().strftime('%Y-%m-%d')
                    }
                    
                    jobs.append(job)
                    
                except Exception as e:
                    continue  # Skip problematic job cards
            
            print(f"✓ {len(job_cards)} jobs")
            
            # Random delay to avoid rate limiting (2-4 seconds)
            time.sleep(random.uniform(2, 4))
            
        except Exception as e:
            print(f"Error: {str(e)}")
            continue
    
    print(f"\n  Total jobs scraped: {len(jobs)}")
    return jobs

# =============================================================================
# SCRAPER 2: INDEED INDIA (BeautifulSoup - Static Scraping)
# =============================================================================

def scrape_indeed(keyword, max_pages=3):
    """
    Scrape job postings from Indeed India.
    
    Args:
        keyword: search term (e.g., 'data+analyst')
        max_pages: number of pages to scrape
    
    Returns:
        List of job dictionaries
    """
    print(f"\n{'='*60}")
    print(f"SCRAPING INDEED INDIA — Keyword: {keyword.replace('+', ' ')}")
    print(f"{'='*60}")
    
    jobs = []
    
    for page in range(0, max_pages * 10, 10):  # Indeed uses 0, 10, 20, 30...
        # Indeed India URL structure
        url = f"https://in.indeed.com/jobs?q={keyword}&l=India&start={page}"
        
        print(f"  Page {page//10 + 1}/{max_pages}...", end=" ")
        
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            
            if response.status_code != 200:
                print(f"Failed (Status {response.status_code})")
                continue
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Find all job cards (Indeed's structure as of 2026)
            job_cards = soup.find_all('div', class_='job_seen_beacon')
            
            if not job_cards:
                print("No jobs found")
                break
            
            for card in job_cards:
                try:
                    # Job title
                    title_tag = card.find('h2', class_='jobTitle')
                    title = title_tag.text.strip() if title_tag else 'Not specified'
                    
                    # Company name
                    company_tag = card.find('span', class_='companyName')
                    company = company_tag.text.strip() if company_tag else 'Not specified'
                    
                    # Location
                    loc_tag = card.find('div', class_='companyLocation')
                    location = loc_tag.text.strip() if loc_tag else 'Not specified'
                    
                    # Salary
                    salary_tag = card.find('span', class_='salary-snippet')
                    salary = salary_tag.text.strip() if salary_tag else 'Not disclosed'
                    
                    # Job snippet (description preview)
                    snippet_tag = card.find('div', class_='job-snippet')
                    description = snippet_tag.text.strip() if snippet_tag else 'Not available'
                    
                    # Job URL
                    link_tag = card.find('a', class_='jcs-JobTitle')
                    job_url = f"https://in.indeed.com{link_tag['href']}" if link_tag and 'href' in link_tag.attrs else ''
                    
                    job = {
                        'title': title,
                        'company': company,
                        'location': location,
                        'description': description,
                        'requirements': 'Not available',
                        'benefits': 'Not available',
                        'employment_type': 'Full-time',
                        'required_experience': 'Not specified',
                        'salary_range': salary,
                        'company_profile': 'Not available',
                        'has_company_logo': 0,
                        'has_questions': 0,
                        'telecommuting': 0,
                        'fraudulent': 0,
                        'source': 'Indeed India',
                        'url': job_url,
                        'scraped_date': datetime.now().strftime('%Y-%m-%d')
                    }
                    
                    jobs.append(job)
                    
                except Exception as e:
                    continue
            
            print(f"✓ {len(job_cards)} jobs")
            time.sleep(random.uniform(2, 4))
            
        except Exception as e:
            print(f"Error: {str(e)}")
            continue
    
    print(f"\n  Total jobs scraped: {len(jobs)}")
    return jobs

# =============================================================================
# DATA CLEANING
# =============================================================================

def clean_scraped_data(jobs):
    """Clean and deduplicate scraped job data."""
    print(f"\n{'='*60}")
    print("CLEANING DATA")
    print(f"{'='*60}")
    
    print(f"  Jobs before cleaning: {len(jobs)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(jobs)
    
    # Remove duplicates based on title + company
    df = df.drop_duplicates(subset=['title', 'company'], keep='first')
    print(f"  Jobs after deduplication: {len(df)}")
    
    # Remove rows with missing critical fields
    df = df[df['title'] != 'Not specified']
    df = df[df['company'] != 'Not specified']
    print(f"  Jobs after removing incomplete: {len(df)}")
    
    # Clean text fields
    text_cols = ['title', 'company', 'location', 'description', 
                 'requirements', 'benefits', 'employment_type',
                 'required_experience', 'salary_range', 'company_profile']
    
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].str.strip()
            df[col] = df[col].replace('Not available', '')
            df[col] = df[col].replace('Not specified', '')
    
    return df

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run the full scraping pipeline."""
    print("\n" + "="*60)
    print("  FAKE JOB DETECTOR — WEB SCRAPER")
    print("  By Divya Virkud | v2.0")
    print("="*60)
    
    all_jobs = []
    
    # Scrape from multiple keywords
    for keyword in SEARCH_KEYWORDS:
        
        # Naukri scraping
        try:
            naukri_jobs = scrape_naukri(keyword, max_pages=2)
            all_jobs.extend(naukri_jobs)
        except Exception as e:
            print(f"  Naukri scraping failed: {str(e)}")
        
        # Indeed scraping
        try:
            indeed_jobs = scrape_indeed(keyword, max_pages=2)
            all_jobs.extend(indeed_jobs)
        except Exception as e:
            print(f"  Indeed scraping failed: {str(e)}")
        
        # Delay between keywords
        time.sleep(random.uniform(3, 5))
    
    # Clean data
    if all_jobs:
        df = clean_scraped_data(all_jobs)
        
        # Save to CSV
        output_file = f"scraped_jobs_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\n{'='*60}")
        print(f"  SCRAPING COMPLETE!")
        print(f"  Total jobs collected: {len(df)}")
        print(f"  Saved to: {output_file}")
        print(f"{'='*60}\n")
        
        # Summary statistics
        print("\n📊 SUMMARY:")
        print(f"  Naukri jobs  : {len(df[df['source'] == 'Naukri.com'])}")
        print(f"  Indeed jobs  : {len(df[df['source'] == 'Indeed India'])}")
        print(f"\n  Top companies:")
        print(df['company'].value_counts().head(5).to_string())
        print(f"\n  Top locations:")
        print(df['location'].value_counts().head(5).to_string())
    else:
        print("\n⚠️ No jobs scraped. Check your internet connection or site structure may have changed.")

# =============================================================================
# INSTRUCTIONS FOR USE
# =============================================================================

"""
HOW TO USE THIS SCRAPER:

1. INSTALL REQUIREMENTS:
   pip install requests beautifulsoup4 pandas lxml

2. RUN THE SCRAPER:
   python scraper.py
   
3. OUTPUT:
   - Creates CSV file: scraped_jobs_YYYYMMDD.csv
   - Contains columns matching your training data format
   
4. MERGE WITH EXISTING DATA:
   - Open scraped CSV in Excel
   - Manually label fraudulent column (0 = real, 1 = fake)
   - Append to your original fake_job_postings.csv
   - Rerun preprocessing + model training pipeline
   
5. ETHICS & LEGALITY:
   - Respect robots.txt (we use delays between requests)
   - Use data for education/research only
   - Do not scrape at scale (>1000 jobs/day)
   - Some sites may block scrapers - this is a learning tool
   
6. TROUBLESHOOTING:
   - If no jobs scraped: site structure changed
   - Inspect element on Naukri/Indeed to see current HTML classes
   - Update class names in code accordingly
   
7. LIMITATIONS:
   - Only scrapes job listings (not full job pages)
   - Cannot bypass login walls or CAPTCHAs
   - For full descriptions, consider Selenium (slower but more powerful)
"""

if __name__ == '__main__':
    main()
