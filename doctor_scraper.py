import requests
from bs4 import BeautifulSoup
import csv
import json
import time
import random
from urllib.parse import urljoin
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DoctorScraper:
    def __init__(self):
        self.base_url = "https://www.vezeeta.com/en/doctor/dermatology"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        # Governorate mapping
        self.governorates = {
            "cairo": "cairo",
            "giza": "giza", 
            "alexandria": "alexandria",
            "qalyubia": "qalyubia",
            "dakahlia": "dakahlia",
            "gharbia": "gharbia",
            "kafr el sheikh": "kafr-el-sheikh",
            "beheira": "beheira",
            "menoufia": "menoufia",
            "sharqia": "sharqia",
            "ismailia": "ismailia",
            "port said": "port-said",
            "suez": "suez",
            "north sinai": "north-sinai",
            "south sinai": "south-sinai"
        }
    
    def scrape_doctors(self, governorate="cairo", max_pages=3):
        """
        Scrape doctors from Vezeeta for a specific governorate
        
        Args:
            governorate (str): Name of the governorate
            max_pages (int): Maximum number of pages to scrape
            
        Returns:
            list: List of doctor dictionaries
        """
        governorate = governorate.lower().strip()
        if governorate not in self.governorates:
            logger.error(f"Governorate '{governorate}' not supported")
            return []
        
        doctors_data = []
        governorate_code = self.governorates[governorate]
        
        try:
            for page in range(1, max_pages + 1):
                logger.info(f"Scraping page {page} for {governorate}")
                
                # Construct URL with page parameter
                url = f"{self.base_url}/{governorate_code}"
                if page > 1:
                    url += f"?page={page}"
                
                # Make request with retry logic
                page_doctors = self._scrape_page(url)
                if not page_doctors:
                    logger.warning(f"No doctors found on page {page}, stopping")
                    break
                
                doctors_data.extend(page_doctors)
                
                # Add random delay to avoid being blocked
                time.sleep(random.uniform(1, 3))
            
            logger.info(f"Successfully scraped {len(doctors_data)} doctors from {governorate}")
            return doctors_data
            
        except Exception as e:
            logger.error(f"Error scraping doctors: {e}")
            return []
    
    def _scrape_page(self, url, retries=3):
        """Scrape a single page with retry logic"""
        for attempt in range(retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, "lxml")
                
                # Updated selectors - these might need adjustment based on current HTML structure
                doctor_cards = soup.find_all('div', class_=lambda x: x and 'doctor-card' in x.lower() if x else False)
                
                # If above doesn't work, try alternative selectors
                if not doctor_cards:
                    doctor_cards = soup.find_all('div', class_=lambda x: x and any(keyword in x.lower() for keyword in ['card', 'doctor', 'profile']) if x else False)
                
                # Fallback to your original selectors
                if not doctor_cards:
                    return self._scrape_with_original_selectors(soup)
                
                doctors = []
                for card in doctor_cards[:20]:  # Limit to 20 per page
                    doctor_data = self._extract_doctor_info(card)
                    if doctor_data:
                        doctors.append(doctor_data)
                
                return doctors
                
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
            except Exception as e:
                logger.error(f"Error parsing page: {e}")
                break
        
        return []
    
    def _scrape_with_original_selectors(self, soup):
        """Use your original selectors as fallback"""
        try:
            names = soup.find_all('a', {'class': 'CommonStylesstyle__TransparentA-sc-1vkcu2o-2 cTFrlk'})
            specialties = soup.find_all('p', {'class': 'DoctorCardSubComponentsstyle__Text-sc-1vq3h7c-14 DoctorCardSubComponentsstyle__DescText-sc-1vq3h7c-17 fuBVZG esZVig'})
            locations = soup.find_all('span', {'class': 'DoctorCardstyle__Text-sc-uptab2-4 blwPZf'})
            
            doctors = []
            min_length = min(len(names), len(specialties), len(locations))
            
            for i in range(min_length):
                try:
                    doctor = {
                        'name': names[i].text.strip(),
                        'specialty': specialties[i].text.strip(),
                        'location': locations[i].text.strip(),
                        'rating': 'N/A',
                        'experience': 'N/A',
                        'fees': 'Contact for details'
                    }
                    doctors.append(doctor)
                except (AttributeError, IndexError) as e:
                    logger.debug(f"Error extracting doctor {i}: {e}")
                    continue
            
            return doctors
            
        except Exception as e:
            logger.error(f"Error with original selectors: {e}")
            return []
    
    def _extract_doctor_info(self, card):
        """Extract doctor information from a card element"""
        try:
            # Try multiple selectors for name
            name_elem = (card.find('h3') or 
                        card.find('h4') or 
                        card.find('a', class_=lambda x: x and 'name' in x.lower() if x else False) or
                        card.find('div', class_=lambda x: x and 'name' in x.lower() if x else False))
            
            name = name_elem.text.strip() if name_elem else "Unknown Doctor"
            
            # Try to find specialty
            specialty_elem = (card.find('p', class_=lambda x: x and 'specialty' in x.lower() if x else False) or
                            card.find('span', class_=lambda x: x and 'specialty' in x.lower() if x else False) or
                            card.find_all('p')[1] if len(card.find_all('p')) > 1 else None)
            
            specialty = specialty_elem.text.strip() if specialty_elem else "Dermatologist"
            
            # Try to find location
            location_elem = (card.find('span', class_=lambda x: x and 'location' in x.lower() if x else False) or
                           card.find('div', class_=lambda x: x and 'address' in x.lower() if x else False) or
                           card.find('p', class_=lambda x: x and 'address' in x.lower() if x else False))
            
            location = location_elem.text.strip() if location_elem else "Location not specified"
            
            # Try to find rating
            rating_elem = card.find(class_=lambda x: x and 'rating' in x.lower() if x