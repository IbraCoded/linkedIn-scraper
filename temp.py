import asyncio
import json
import os
import time
import urllib.parse
from datetime import datetime
from playwright.async_api import async_playwright
from typing import Dict, Any, Optional
from schemas import  ProfileData
import requests
import time
import base64
import json


from config import settings
from cv_processor import process_downloaded_cv

API_URL = "https://15fg-sa.teleows.com/adc-intg/api/rest/v1/HR_RESUME_APP/HR_AI_API/hr_ai_api_cv_snaps/store_create"
ACCESS_ID = "15fg|zjc81H5MF4XirbGVNduS"
SECRET_KEY = "00J(Y8*%zi#2#;wW=+,2I)N4fK2jHhltRszcynoH"


async def search_linkedin_profiles(page, search_query, location="Nigeria", max_profiles=10):
    """
    Search for LinkedIn profiles based on query and location
    Returns list of profile dictionaries with name and url
    """
    print(f"🔍 Searching LinkedIn for: '{search_query}' in {location}")
    
    # Construct LinkedIn people search URL
    encoded_query = urllib.parse.quote(search_query)
    encoded_location = urllib.parse.quote(location)
    
    # LinkedIn people search URL structure
    search_url = f"https://www.linkedin.com/search/results/people/?geoUrn=%5B%22105365761%22%5D&keywords={encoded_query}&origin=FACETED_SEARCH"
    
    # If location is specified, we can add it to the search
    if location.lower() != "global":
        # For Nigeria specifically, you might need to adjust the geoUrn
        search_url = f"https://www.linkedin.com/search/results/people/?geoUrn=%5B%22105365761%22%5D&keywords={encoded_query}&origin=FACETED_SEARCH"
    
    print(f"📍 Search URL: {search_url}")
    
    try:
        await page.goto(search_url)
        await page.wait_for_load_state('domcontentloaded')
        await page.wait_for_timeout(3000)  # Let the page load
        
        # Take screenshot of search results
        # await page.screenshot(path=f"{settings.SCREENSHOTS_PATH}/search_results.png")
        # print(f"📸 Screenshot: {settings.SCREENSHOTS_PATH}/search_results.png")
        
        profiles = []
        
        # Look for profile links in search results
        profile_selectors = [
            'a[href*="/in/"]',  # Profile links contain /in/
            '.search-result__wrapper a[href*="/in/"]',
            '.entity-result__title-text a[href*="/in/"]'
        ]
        
        found_links = set()  # Use set to avoid duplicates
        
        for selector in profile_selectors:
            try:
                elements = await page.query_selector_all(selector)
                print(f"   Found {len(elements)} elements with selector: {selector}")
                
                for element in elements:
                    href = await element.get_attribute('href')
                    if href and '/in/' in href and href not in found_links:
                        # Clean up the URL (remove query parameters)
                        clean_url = href.split('?')[0]
                        if not clean_url.startswith('http'):
                            clean_url = 'https://www.linkedin.com' + clean_url
                        
                        # Extract name from the link text or nearby elements
                        try:
                            # Try to get name from the link text
                            name_text = await element.inner_text()
                            name_text = name_text.strip()
                            
                            # If name is empty, try to find it in parent elements
                            if not name_text:
                                parent = await element.query_selector('xpath=..')
                                if parent:
                                    name_text = await parent.inner_text()
                                    name_text = name_text.strip().split('\n')[0]  # Take first line
                            
                            # Clean up the name (remove extra whitespace, newlines)
                            name_text = ' '.join(name_text.split()) if name_text else "Unknown"
                            
                            if name_text and name_text != "Unknown" and len(name_text) > 2:
                                profiles.append({
                                    "name": name_text,
                                    "url": clean_url
                                })
                                found_links.add(href)
                                print(f"   ✅ Found profile: {name_text} -> {clean_url}")
                                
                                if len(profiles) >= max_profiles:
                                    break
                        
                        except Exception as e:
                            print(f"   ⚠️  Error extracting name for {clean_url}: {e}")
                            # Add with URL as name if we can't get the actual name
                            profile_id = clean_url.split('/in/')[-1].rstrip('/')
                            profiles.append({
                                "name": f"Profile_{profile_id}",
                                "url": clean_url
                            })
                            found_links.add(href)
                
                if len(profiles) >= max_profiles:
                    break
                    
            except Exception as e:
                print(f"   ❌ Error with selector {selector}: {e}")
        
        # If we didn't find enough profiles, try scrolling and searching more
        if len(profiles) < max_profiles:
            print(f"   📜 Found {len(profiles)} profiles, trying to load more...")
            try:
                # Scroll down to potentially load more results
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(3000)
                
                # Try the search again after scrolling
                # (You could implement additional logic here to find more profiles)
                
            except Exception as e:
                print(f"   ⚠️  Error while scrolling: {e}")
        
        print(f"🎯 Found {len(profiles)} profiles total")
        return profiles[:max_profiles]  # Return only the requested number
        
    except Exception as e:
        print(f"❌ Error during LinkedIn search: {e}")
        return []

async def login_to_linkedin(page):
    """Login to LinkedIn"""
    print("🔐 Logging into LinkedIn...")
    await page.goto('https://www.linkedin.com/login')
    await page.wait_for_load_state('domcontentloaded')
    
    await page.fill('input[name="session_key"]', settings.LINKEDIN_EMAIL)
    await page.fill('input[name="session_password"]', settings.LINKEDIN_PASSWORD)
    await page.click('button[type="submit"]')
    
    print("⏳ Waiting for login...")
    await page.wait_for_timeout(5000)
    
    current_url = page.url
    if 'login' in current_url:
        print("⚠️  Manual intervention may be required for 2FA/CAPTCHA")
        # In production, you might want to handle this differently
        await page.wait_for_timeout(10000)  # Give time for manual intervention
    
    print("✅ Login completed!")

async def download_and_process_cv(page, profile: Dict, job_spec:dict) -> Optional[Dict]:
    """Download CV from profile and process it"""
    try:
        print(f"\n🔍 Processing: {profile['name']}")
        
        await page.goto(profile['url'])
        await page.wait_for_load_state('domcontentloaded')
        await page.wait_for_timeout(3000)
        
        # # Take screenshot
        safe_name = profile['name'].replace(' ', '_').replace('/', '_')
        # await page.screenshot(path=f"{settings.SCREENSHOTS_PATH}/{safe_name}_profile.png")
        
        # Monitor downloads folder for new PDF files
        downloads_folder = os.path.expanduser("~/Downloads")
        before_files = []
        if os.path.exists(downloads_folder):
            before_files = [f for f in os.listdir(downloads_folder) if f.lower().endswith('.pdf')]
        
        script_before_files = []
        if os.path.exists(settings.DOWNLOADS_PATH):
            script_before_files = [f for f in os.listdir(settings.DOWNLOADS_PATH) if f.lower().endswith('.pdf')]
        
        # Find and click More button
        more_selectors = [
            'button:has-text("More")',
            'button[aria-label*="More"]',
            '.pv-top-card-v2-ctas button:has-text("More")'
        ]
        
        more_button = None
        for selector in more_selectors:
            try:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    text = await element.inner_text()
                    if "More" in text and await element.is_visible() and await element.is_enabled():
                        more_button = element
                        break
                if more_button:
                    break
            except Exception:
                continue
        
        if not more_button:
            print("   ❌ More button not found!")
            return None
        
        # Click More button
        await more_button.click()
        await page.wait_for_timeout(2000)
        
        # Find Save to PDF option
        pdf_selectors = [
            'div[aria-label="Save to PDF"]',
            '[aria-label="Save to PDF"]',
            'div:has-text("Save to PDF")',
            '.artdeco-dropdown__item:has-text("Save to PDF")'
        ]
        
        pdf_element = None
        for selector in pdf_selectors:
            try:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    if await element.is_visible():
                        text = await element.inner_text()
                        aria_label = await element.get_attribute('aria-label')
                        if "Save to PDF" in text or (aria_label and "Save to PDF" in aria_label):
                            pdf_element = element
                            break
                if pdf_element:
                    break
            except Exception:
                continue
        
        if not pdf_element:
            print("   ❌ Save to PDF option not found!")
            return None
        
        # Setup download handling
        download_path = None
        download_complete = asyncio.Event()
        
        def handle_download(download):
            async def save_download():
                nonlocal download_path
                try:
                    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # filename = f"{safe_name}_CV_{timestamp}.pdf"
                    filename = f"{safe_name}.pdf"
                    download_path = os.path.join(settings.DOWNLOADS_PATH, filename)
                    await download.save_as(download_path)
                    print(f"   ✅ PDF downloaded: {download_path}")
                    download_complete.set()
                except Exception as e:
                    print(f"   ❌ Download failed: {e}")
                    download_complete.set()
            
            asyncio.create_task(save_download())
        
        page.on("download", handle_download)
        
        # Click Save to PDF
        await pdf_element.click()
        print("   ✅ Clicked 'Save to PDF'!")
        
        # Monitor for file creation as backup
        async def monitor_downloads():
            start_time = time.time()
            timeout = 60
            
            while time.time() - start_time < timeout:
                # Check ~/Downloads folder
                if os.path.exists(downloads_folder):
                    current_pdfs = [f for f in os.listdir(downloads_folder) if f.lower().endswith('.pdf')]
                    new_pdfs = [pdf for pdf in current_pdfs if pdf not in before_files]
                    
                    if new_pdfs:
                        source_path = os.path.join(downloads_folder, new_pdfs[0])
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{safe_name}_CV_{timestamp}.pdf"
                        dest_path = os.path.join(settings.DOWNLOADS_PATH, filename)
                        
                        # Copy file to our downloads folder
                        import shutil
                        shutil.copy2(source_path, dest_path)
                        
                        nonlocal download_path
                        download_path = dest_path
                        download_complete.set()
                        return dest_path
                
                await asyncio.sleep(2)
            
            return None
        
        # Wait for download with both methods
        monitor_task = asyncio.create_task(monitor_downloads())
        
        try:
            await asyncio.wait_for(download_complete.wait(), timeout=70.0)
        except asyncio.TimeoutError:
            print("   ⏰ Download timeout")
            monitor_task.cancel()
            return None
        
        if not download_path or not os.path.exists(download_path):
            print("   ❌ Download failed")
            return None
        
        # Process the downloaded CV
        cv_data = await process_downloaded_cv(download_path, profile['name'], job_spec)
        if cv_data:
            cv_data['source_pdf'] = download_path
            cv_data['download_timestamp'] = datetime.now().isoformat()
            return cv_data
        
        return None
        
    except Exception as e:
        print(f"   ❌ Error processing profile {profile['name']}: {e}")
        return None





def convert_cv_data_format(cv_data: dict) -> dict:
    """
    Convert CV data from ultra-robust parser format to API response format
    Maps uppercase field names to lowercase for consistency
    """
    converted = {}
    
    # Field name mapping
    field_mapping = {
        "Name": "name",
        "Location": "location", 
        "Contact_Information": "contact_information",
        "Top_Skills": "top_skills",
        "Certifications": "certifications", 
        "Honors_Awards": "honors_awards",
        "Summary": "summary",
        "Experience": "experience",
        "Education": "education",
        "Languages": "languages"
    }
    
    # Convert field names
    for old_key, new_key in field_mapping.items():
        if old_key in cv_data:
            converted[new_key] = cv_data[old_key]
        else:
            # Set default values for missing fields
            if new_key == "contact_information":
                converted[new_key] = {
                    "email": "",
                    "phone": "", 
                    "linkedin": "",
                    "website": "",
                    "portfolio": ""
                }
            elif new_key in ["top_skills", "certifications", "honors_awards", "experience", "education", "languages"]:
                converted[new_key] = []
            else:
                converted[new_key] = ""
    
    # Handle contact information field name conversion
    if "Contact_Information" in cv_data:
        contact_info = cv_data["Contact_Information"]
        converted["contact_information"] = {
            "email": contact_info.get("Email", ""),
            "phone": contact_info.get("Phone", ""),
            "linkedin": contact_info.get("LinkedIn", ""),
            "website": contact_info.get("Website", ""),
            "portfolio": contact_info.get("Portfolio", "")
        }
    
    # Preserve metadata fields
    converted["source_pdf"] = cv_data.get("source_pdf")
    converted["download_timestamp"] = cv_data.get("download_timestamp")
    
    return converted


async def send_response_data(resume_data: Dict[str, Any]):

    request_body = {
      **resume_data,
        "headers": {
            "http_method": "POST",
            "content_type": "application/json"
        }
    }
    
    credentials = f"{ACCESS_ID}:{SECRET_KEY}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Basic {encoded_credentials}"
    }
    
    try:
        response = requests.post(
            url=API_URL,
            headers=headers,
            json=request_body,
            timeout=30
        )
        
        print(response.status_code, response.text)
        print(response.json())

        
        response.raise_for_status()
        
        return {
            "success": True,
            "status_code": response.status_code,
            "response": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
        }
    
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e),
            "response": getattr(e, 'response', {}).text if hasattr(e, 'response') else None
        }   



async def scrape_linkedin_profiles(search_query: str, location: str, max_profiles: int, job_spec: dict) -> Dict:
    """
    Main scraping function - your original debug_single_profile_download logic
    """
    print(f"🎯 Search Configuration:")
    print(f"   Query: {search_query}")
    print(f"   Location: {location}")
    print(f"   Max profiles: {max_profiles}")
    print(f"   Job Sppec: {job_spec}")
    
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=settings.PLAYWRIGHT_HEADLESS,
            args=[
                '--no-sandbox', 
                '--disable-dev-shm-usage',
                '--disable-blink-features=AutomationControlled'
            ]
        )
        
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080},
            accept_downloads=True
        )
        
        page = await context.new_page()
        page.set_default_timeout(60000)
        
        try:
            # Login to LinkedIn
            await login_to_linkedin(page)
            
            # Search for profiles
            print("\n" + "="*50)
            print("🔍 SEARCHING FOR PROFILES")
            print("="*50)
            
            profiles = await search_linkedin_profiles(page, search_query, location, max_profiles)
            
            print(f"\n📋 Found profiles to download:")
            for i, profile in enumerate(profiles, 1):
                print(f"   {i}. {profile['name']}")
                print(f"      URL: {profile['url']}")
            
            # Save search results to JSON for reference
            # search_results_file = f"{settings.PROCESSED_PATH}/search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            # with open(search_results_file, 'w', encoding='utf-8') as f:
            #     json.dump({
            #         "search_query": search_query,
            #         "location": location,
            #         "search_date": datetime.now().isoformat(),
            #         "profiles": profiles
            #     }, f, indent=2, ensure_ascii=False)
            # print(f"💾 Search results saved to: {search_results_file}")
            
            # Process each profile
            print("\n" + "="*50)
            print("📥 DOWNLOADING CVS")
            print("="*50)
            
            successful_downloads = 0
            processed_cvs = []
            complete_profiles = []
            
            for profile in profiles:
                processed_cvs = []
                complete_profiles = []
            
                try:
                    print(f"   Job Sppec: {job_spec}")
                    cv_data = await download_and_process_cv(page, profile, job_spec)
                    
                    #check if the similarity is less than 30%
                    
                    # if cv_data:
                    if cv_data and (cv_data.get('Job_Match', {}).get('similarity_score', 0.0) >= 0.12):
                        #debugging output
                        #print(f"🔍 DEBUG: download_and_process_cv returned: {cv_data}")
                        
                        # NEW: Convert PDF to base64
                        pdf_base64 = ""
                        if cv_data and 'source_pdf' in cv_data and cv_data['source_pdf']:
                            pdf_base64 = await convert_pdf_to_base64(cv_data['source_pdf']) 
                            # Add base64 to cv_data
                            cv_data['pdf_base64'] = pdf_base64
                        
                        converted_data = convert_cv_data_format(cv_data)
                        try:
                            profile_obj = ProfileData(**converted_data)
                            complete_profiles.append(profile_obj.dict())
                        except Exception as validation_error:
                            print(f"⚠️  Validation warning for {cv_data.get('Name', 'Unknown')}: {validation_error}")
                            # Use raw data if validation fails
                            complete_profiles.append(cv_data)
                    
                        if cv_data:
                            response_data = {
                                "status": "success",
                                "message": f"Successfully processed 1 profiles",
                                "search_query": search_query,
                                "location": location,
                                "total_profiles_found": 1,
                                "successful_downloads": 1,
                                "successfully_processed": 1,
                                "processing_date": datetime.now().isoformat(),
                                
                                # MAIN RESPONSE: Converted format (lowercase field names)
                                "profiles": complete_profiles,
                        
                                # NEW: PDF Base64 data
                                "pdf_data": {
                                    "profile_name": cv_data.get('Name', 'Unknown'),
                                    "profile_url": profile.get('url', ''),
                                    "pdf_base64": pdf_base64,
                                    "pdf_available": bool(pdf_base64),
                                    "pdf_size_kb": round(len(pdf_base64) * 3/4 / 1024, 2) if pdf_base64 else 0
                                },
                                
                                # METADATA
                                "parsing_info": {
                                    "parser_version": "ultra_robust_v1",
                                    "field_mapping_applied": True,
                                    "original_format_preserved": True,
                                    "pdf_base64_included": bool(pdf_base64)
                                }
                            }
                            # print(response_data)
            
                            await send_response_data(response_data)
                            successful_downloads += 1
                            print(f"   ✅ CV processed successfully for {profile['name']}!")
                            print(f"   ✅ CV processed successfully for {cv_data}!")
                            processed_cvs.append(cv_data)
                        else:
                            print(f"   ❌ Failed to process CV for {profile['name']}")
                        
                        # Small delay between profiles
                        await page.wait_for_timeout(2000)
                        
                        # delete the downloaded PDF to save space
                        if 'source_pdf' in cv_data and os.path.exists(cv_data['source_pdf']):
                            os.remove(cv_data['source_pdf'])
                            print(f"   🗑️  Deleted temporary file: {cv_data['source_pdf']}")
                except Exception as e:
                    print(f"Error processing profile {profile['name']}: {e}")
                    continue
            
            # Final summary
            print("\n" + "="*50)
            print("📊 DOWNLOAD SUMMARY")
            print("="*50)
            print(f"✅ Successful downloads: {successful_downloads}")
            print(f"❌ Failed downloads: {len(profiles) - successful_downloads}")
            print(f"📊 Success rate: {(successful_downloads/len(profiles)*100):.1f}%")
            print(f"🔄 Successfully processed CVs: {len(processed_cvs)}")
            
        except Exception as e:
            print(f"❌ Error during scraping: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "search_query": search_query,
                "location": location,
                "processing_date": datetime.now().isoformat()
            }
            
        finally:
            await browser.close()
            
            
# convert PDF to base64

async def convert_pdf_to_base64(pdf_path: str) -> str:
    """
    Convert PDF file to base64 string
    """
    print(f"🔄 Converting to base64: {os.path.basename(pdf_path)}")
    
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_data = pdf_file.read()
            base64_encoded = base64.b64encode(pdf_data).decode('utf-8')
        
        print(f"   ✅ Successfully converted to base64 ({len(base64_encoded)} characters)")
        return base64_encoded
    
    except Exception as e:
        print(f"   ❌ Failed to convert to base64: {e}")
        return ""

if __name__ == "__main__":
    import asyncio
    asyncio.run(scrape_linkedin_profiles("RAN Engineer/ Wireless Engineer", "Nigeria", 2))
