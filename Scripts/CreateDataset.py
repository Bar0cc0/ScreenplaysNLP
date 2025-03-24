#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CreateDataset.py
author: Michael Garancher
date: 2021-09-01
Description:
	This script collects and processes subtitle and script data for the movie "Back to the Future III".
	It downloads subtitles from a website, extracts them, and aligns them with the movie script.
	The aligned data is then stored in an Excel file.
Notes:
	- Requires Firefox browser for Selenium web scraping
	- Uses temporary directories for staging downloaded files
	- Falls back to archived files if web scraping fails
	- Uses Jaccard similarity for text alignment
	- Configurable via CONFIG dictionary
"""

# Std library
import os, sys, warnings, shutil, tempfile
import re, queue, logging, time
from pathlib import Path
from typing import Dict, List, Union

# Third party
import pandas as pd
from requests_html import HTMLSession
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions 



# Suppress warnings
sys.tracebacklimit = 0
warnings.filterwarnings('ignore')

# Pandas display options
pd.set_option("display.max_colwidth", 100)


# Configuration
ROOT_DIR:Path = Path(__file__).parents[1].resolve()
CONFIG:Dict[str, Union[str,Path]] = {
	"SRC_SRT": "https://yifysubtitles.ch/subtitles/back-to-the-future-part-iii-1990-english-yify-129087",
	"SRC_SCRIPT": "https://movies.fandom.com/wiki/Back_to_the_Future_Part_III/Transcript",
	"OUTPUT_DIR": ROOT_DIR / 'Data',
	"SIMILARITY_THRESHOLD": 0.2 # Minimum similarity score for alignment
}

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s',
	handlers=[
		logging.StreamHandler(),
		logging.FileHandler(CONFIG['OUTPUT_DIR'] / 'topic_extraction.log', mode='w')
	]
)


class Toolbox(object):
	"""
	Utility functions for file operations and data processing.
	This class provides static utility methods for managing temporary directories,
	processing files, and calculating text similarity.
	"""
	@staticmethod
	def staging_tmpDir_create() -> Path:
		"""Create a temporary directory for processing files."""
		staging_path = ROOT_DIR.joinpath('staging')
		if not staging_path.exists():
			staging_path.mkdir(parents=True, exist_ok=True)
		staging_dir = tempfile.mkdtemp(dir=staging_path)
		return Path(staging_dir)

	@staticmethod
	def staging_tmpDir_clear(staging_dir:Path) -> None:
		"""Remove the temporary directory and its parent staging directory."""
		shutil.rmtree(staging_dir)
		shutil.rmtree(ROOT_DIR.joinpath('staging'))

	@staticmethod
	def unzip_archive(staging_dir:Path) -> None:
		"""Extract zip archives and rename SRT files in the staging directory."""
		for archive in staging_dir.glob('*.zip'):
			shutil.unpack_archive(archive, staging_dir)
			os.remove(archive)
		for srt in staging_dir.glob('*.srt'):
			os.rename(srt, staging_dir.joinpath('_srt.txt')) #NOTE renamed from bttf_srt.txt
	
	@staticmethod
	def get_txt_files(staging_dir:Path) -> List[Path]:
		"""Get a list of all text files in the staging directory."""
		files = []
		for txt in staging_dir.glob('*.txt'):
			files.append(txt)
		return files
	
	@staticmethod
	def numbered_file(format:str) -> Path:
		"""Generate a unique numbered filename in the output directory."""
		counter = 0
		while True:
			counter += 1
			path = CONFIG['OUTPUT_DIR'].joinpath(f'_({counter}).{format}') #NOTE renamed from bttf_({counter}).{format}
			if not path.exists():
				return path

	@staticmethod
	def jaccard_similarity(str1, str2):
		"""Calculate Jaccard similarity between two strings (word overlap)."""
		if pd.isna(str1) or pd.isna(str2):
			return 0
		
		# Preprocess
		str1 = str1.lower()
		str2 = str2.lower()
		
		# Remove punctuation except apostrophes
		str1 = re.sub(r'[^\w\s\']', ' ', str1)
		str2 = re.sub(r'[^\w\s\']', ' ', str2)
		
		# Convert to sets of words
		set1 = set(str1.split())
		set2 = set(str2.split())
		
		# Compute Jaccard similarity
		intersection = len(set1.intersection(set2))
		union = len(set1.union(set2))
		
		return intersection / union if union > 0 else 0



class CollectDataFromWeb(object):
	"""
	Handles web scraping to collect subtitle and script data.
	This class is responsible for downloading subtitle archives and script text
	from specified web sources and saving them to the staging directory.
	"""
	def __init__(self, staging_dir:Path):
		self.staging_dir = staging_dir
	
	def setup_scraping_environment() -> webdriver:
		"""Configure scraping environment with reliability features."""
		options = webdriver.FirefoxOptions()
		options.headless = True
		options.add_argument("--no-sandbox")
		
		# Create browser profile with anti-detection measures
		profile = webdriver.FirefoxProfile()
		profile.set_preference("javascript.enabled", True)
		
		driver = webdriver.Firefox(options=options, firefox_profile=profile)
		driver.set_page_load_timeout(30)
		driver.implicitly_wait(10)
		return driver

	def timout(self) -> None:
		"""Wait for download to complete or timeout after a specified period."""
		try:
			# Wait for download to complete by checking if file appears in directory
			max_wait = 30
			wait_time = 0
			while wait_time < max_wait:
				time.sleep(1)
				wait_time += 1
				if any(file.endswith('.zip') for file in os.listdir(self.staging_dir)):
					logging.info(f"Download completed in {wait_time} seconds")
					break
				elif wait_time % 5 == 0:
					logging.info(f"Waiting for download... ({wait_time}s)")
			if wait_time >= max_wait:
				logging.error("Download timed out after 30 seconds. Check if URL is valid.")
				#TODO: Implement a fallback strategy (e.g. retry, other URL, local zip file)
		except TimeoutError as e:
			logging.error(f"Timeout error: {e}")

	def find_and_click_download_buttons(self, driver:webdriver, max_depth:int=2, current_depth:int=0, visited_elements:int=None) -> bool:
		"""Recursively find and click potential download buttons until a download starts."""
		if visited_elements is None:
			visited_elements = set()
			
		if current_depth >= max_depth:
			logging.info(f"Reached maximum recursion depth ({max_depth})")
			return False
		
		# Find all potential download buttons
		potential_buttons = []
		
		# By ID 
		button_ids = [
			'download-en',
			'btn-download-subtitle', 
			'downloadButton',
			'bt-dwl-bt', 
			'bt-dwl', 
			'download-tigger', 
			'download-button', 
			'dl-button'
		]
		for button_id in button_ids:
			try:
				elements = driver.find_elements(By.ID, button_id)
				if elements:
					potential_buttons.extend(elements)
			except Exception:
				pass

		# By href text partial match
		link_texts = [
			'download', 'Download', 'DOWNLOAD',
			'subtitle', 'subtitles', 'Subtitles',
			'srt', 'SRT', 
			'zip', 'ZIP',
			'english', 'eng', 'en'
		]
		for text in link_texts:
			try:
				elements = driver.find_elements(By.PARTIAL_LINK_TEXT, text)
				if elements:
					potential_buttons.extend(elements)
			except Exception:
				pass

		
		if not potential_buttons:
			logging.info(f"No potential download buttons found at depth {current_depth}")
			return False
		
		# Filter out already visited elements and create element signatures
		new_buttons = []
		for element in potential_buttons:
			try:
				# Create a signature based on attributes to identify unique elements
				attrs = {
					'tag': element.tag_name,
					'text': element.text[:20] if element.text else '',
					'href': element.get_attribute('href')[:30] if element.get_attribute('href') else '',
					'class': element.get_attribute('class') if element.get_attribute('class') else ''
				}
				element_sig = f"{attrs['tag']}:{attrs['text']}:{attrs['href']}:{attrs['class']}"
				
				if (element_sig not in visited_elements and 
					element.is_displayed() and 
					element.is_enabled()):
					new_buttons.append((element, element_sig))
					visited_elements.add(element_sig)
			except Exception:
				pass
		
		if not new_buttons:
			logging.info(f"No new download buttons to try at depth {current_depth}")
			return False
		
		logging.info(f"Found {len(new_buttons)} potential download buttons at depth {current_depth}")
		
		# Try clicking each button
		for i, (button, sig) in enumerate(new_buttons):
			try:
				# Log button info
				button_text = button.text.strip() if button.text else "[No text]"
				button_href = button.get_attribute('href') if button.get_attribute('href') else "[No href]"
				
				# Ensure element is visible
				driver.execute_script("arguments[0].scrollIntoView(true);", button)
				time.sleep(0.5)
				
				# Click the button
				driver.execute_script("arguments[0].click();", button)
				logging.info(f"Clicked button {i+1}/{len(new_buttons)}")
				
				# Check if download started
				start_time = time.time()
				while time.time() - start_time < 5:  # 5 second check
					if any(file.endswith(('.zip', '.srt')) for file in os.listdir(self.staging_dir)):
						logging.info(f"Download started after clicking button at depth {current_depth}")
						return True
					time.sleep(0.5)
				
				# Check for new windows/tabs
				if len(driver.window_handles) > 1:
					# Switch to new window
					driver.switch_to.window(driver.window_handles[-1])
					
					# Recursive call in new window
					if self.find_and_click_download_buttons(driver, max_depth, current_depth + 1, visited_elements):
						return True
					
					# Close and switch back if no download
					driver.close()
					driver.switch_to.window(driver.window_handles[0])
				else:
					# Try recursively on same page (new elements might have appeared)
					if self.find_and_click_download_buttons(driver, max_depth, current_depth + 1, visited_elements):
						return True
					
			except Exception as e:
				logging.warning(f"Error clicking button {i+1} at depth {current_depth}: {str(e)}")
				# Try to recover in case of navigation or stale element
				try:
					if len(driver.window_handles) > 1:
						driver.switch_to.window(driver.window_handles[0])
				except:
					pass
		
		return False

	def download_srt_archive(self, url:str=CONFIG['SRC_SRT']) -> None:
		"""Download subtitle archive from specified URL."""
		options = Options()
		options.set_preference("browser.download.folderList", 2)
		options.set_preference('browser.download.dir', str(self.staging_dir))
		options.set_preference("browser.download.manager.showWhenStarting", False)
		options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/zip")
	
		options.accept_insecure_certs = True
		options.set_preference("security.ssl.enable_ocsp_stapling", False)
		options.set_preference("network.stricttransportsecurity.preloadlist", False)
		options.set_preference("security.cert_pinning.enforcement_level", 0)
		
		with webdriver.Firefox(options=options) as driver:
			try:
				driver.get(url)
				logging.info(f"Navigated to {url}")
				
				# Wait for page to load
				WebDriverWait(driver, 10).until(
					expected_conditions.presence_of_element_located((By.TAG_NAME, 'body'))
				)
				
				# Use recursive search and click download buttons
				download_found = self.find_and_click_download_buttons(driver)
				if not download_found:
					logging.warning("No download started after clicking all buttons")
				else:
					# Wait for download to complete
					self.timout()
				
			except Exception as e:
				logging.error(f"Error downloading subtitle: {e}")
				screenshot_path = os.path.join(self.staging_dir, "error_screenshot.png")
				driver.save_screenshot(screenshot_path)
				logging.info(f"Screenshot saved to {screenshot_path}")
				raise
				
		# Check for downloaded files and process them
		zip_files = list(Path(self.staging_dir).glob('*.zip'))
		srt_files = list(Path(self.staging_dir).glob('*.srt'))

		if zip_files:
			logging.info(f"Found {len(zip_files)} zip files")
			Toolbox.unzip_archive(self.staging_dir)
			os.rename(self.staging_dir.joinpath('_srt.txt'), self.staging_dir.joinpath('_srt.txt'))
			logging.info(f"Unzipped file saved to {self.staging_dir.joinpath('_srt.txt')}")
		elif srt_files:
			logging.info(f"Found {len(srt_files)} srt files")
			os.rename(srt_files[0], self.staging_dir.joinpath('_srt.txt'))
			logging.info(f"SRT file saved to {self.staging_dir.joinpath('_srt.txt')}")
		else:
			logging.warning("No files downloaded. Falling back to archived files.")
			shutil.copyfile(CONFIG['OUTPUT_DIR'].joinpath('archive/_srt.txt'), self.staging_dir.joinpath('_srt.txt'))	
			logging.info(f"Archived file copied to {self.staging_dir.joinpath('_srt.txt')}")
	
	def download_script_txt(self, url:str=CONFIG['SRC_SCRIPT']) -> None:
		"""Download screenplay script from specified URL."""
		try:
			session = HTMLSession()
			response = session.get(url)
			response.raise_for_status() # raise exception 4xx/5xx responses
			parsed_html = BeautifulSoup(response.content, 'html5lib')\
								.find('div', class_="mw-parser-output")\
								.text
			if not parsed_html:
				logging.error('No data found in the parsed HTML')
				return
			with open(self.staging_dir.joinpath('_script.txt'), 'w+', encoding='utf-8') as f:
				f.write(parsed_html)
			logging.info(f"Downloaded script text saved to {self.staging_dir.joinpath('_script.txt')}")
		except Exception as e:
			logging.error(f'{e}. Failed to download script text. Falling back to archived file.')
			shutil.copyfile(CONFIG['OUTPUT_DIR'].joinpath('archive/_script.txt'), self.staging_dir.joinpath('_script.txt'))
			logging.info(f"Archived file copied to {self.staging_dir.joinpath('_script.txt')}")
				
	def collect(self) -> None:
		self.download_srt_archive()
		self.download_script_txt()



class ProcessData(object):
	"""
	Processes subtitle and script data to create an aligned dataset.
	This class extracts dialogue and metadata from subtitle and script files,
	aligns them based on text similarity, and exports the results.
	"""
	def __init__(self, staging_dir:Path):
		self.srt = staging_dir.joinpath('_srt.txt')
		self.script = staging_dir.joinpath('_script.txt')
		self.result:pd.DataFrame = None

	def extract_srt(self) -> pd.DataFrame:
		"""Extract timecodes and dialogues from subtitle file."""
		subtitle_queue = queue.Queue()
		OFFSET = 0 # Number of lines to skip at the beginning of the file

		with open(self.srt, 'r+', encoding='utf-8') as f:
			content = f.read()

			# Pre-process content to standardize newlines and clean up
			content = re.sub(r'\r\n', '\n', content)  # Normalize line endings
			content = re.sub(r'\n{3,}', '\n\n', content)  # Remove excessive newlines
			
			# Split content by timecodes
			TIMECODE_PATTERN = re.compile(r'^(\d{2}:\d{2}:\d{2}(?:,\d{3})?)', re.MULTILINE) # Format: (Timecode,idx --> Timecode,idx) but only captures the first part
			sections = re.split(TIMECODE_PATTERN, content)

			# Process odd-indexed sections as dialogue (section = 'timecode\n dialogue \n')
			# Skip first section (before first timecode)
			for i in range(1, len(sections), 2):
				timecode = sections[i]
				dialogue = sections[i+1] if i+1 < len(sections) else "" # Last section may not have dialogue

				# Clean and segment dialogue
				C1_PATTERN = re.compile(r'([\s]?\-\-\>)[\s\d\:\,]+', re.MULTILINE) # Remove timecode separator at the begining (e.g. --> 00:00:00,000)
				C2_PATTERN = re.compile(r'(\\n[\d]*)', re.MULTILINE) # Remove harcoded '\n' followed by digits
				C3_PATTERN = re.compile(r'[\(]{1}[A-Z\s]+[\)]{1}', re.MULTILINE) # Remove scene instructions located in brackets
				C4_PATTERN = re.compile(r'[\d]+(?=\n)', re.MULTILINE) # Remove digits at the beginning of a line
				dialogue = re.sub(C1_PATTERN, '', dialogue)
				dialogue = re.sub(C2_PATTERN, ' ', dialogue)
				dialogue = re.sub(C3_PATTERN, '', dialogue)
				dialogue = re.sub(C4_PATTERN, '', dialogue)
				dialogue = re.sub(r'[\n]+', ' ', dialogue) # Remove excessive newlines
				dialogue = re.sub(r'^[\"]+', '', dialogue) # Remove quotation marks at the beginning of a line
				dialogue = re.sub(r'[A-Z]+(\:){1}','', dialogue) # Remove character names at the beginning of a line
				dialogue = re.sub(r'\-\s', ' ', dialogue) # Remove hyphens at the beginning of a line
				
				if len(dialogue) > 3:
					subtitle_queue.put(timecode)
					dialogue = subtitle_queue.put(dialogue.strip())
				else:
					continue
				
		
			
		subtitle_entries = []
		timecodes, dialogues = [], []
		dialogue_count = 1

		while not subtitle_queue.empty(): 
			queue_item = subtitle_queue.get()
	
			is_timecode = re.search(TIMECODE_PATTERN, queue_item)
			
			if not is_timecode:
				if len(queue_item) > 3:				
					dialogues.append(queue_item)
					dialogue_count += 1
			else:
				timecodes.append(queue_item)
			
			if len(timecodes) > 1:
				subtitle_entries.append({timecodes[0] : ' '.join(dialogues[-dialogue_count:])})
				timecodes.pop(0)
				dialogues.clear()

		#subtitle_entries.append({timecodes[0] : ' '.join(dialogues[-dialogue_count:])})
		df = pd.concat([pd.DataFrame.from_dict(_, columns=['dialogue'], orient='index') for _ in subtitle_entries])   
		df = df.reset_index().rename(columns={'index':'timecode'})

		logging.info(f"Extracted {len(df)} subtitle entries")
		
		return df[OFFSET:] #.reset_index(drop=True)
	
	def extract_script(self) -> pd.DataFrame:
		"""Extract character names and dialogues from script file."""
		script_queue = queue.Queue() 
		OFFSET = 0 # Number of lines to skip at the beginning of the file

		with open(self.script, 'r+', encoding='utf-8') as f:
			content = f.read()

			# Pre-process content to standardize newlines and clean up
			content = re.sub(r'\r\n', '\n', content)  # Normalize line endings
			content = re.sub(r'\n{3,}', '\n\n', content)  # Remove excessive newlines
			
			# Split content by character markers
			CHARACTER_PATTERN = re.compile(r'^([A-Z\d]+[A-Za-z\s\.]+[\d]?)(?=:)', re.MULTILINE) # Format: Character name
			sections = re.split(CHARACTER_PATTERN, content)
			
			# Process odd-indexed sections as dialogue (section = 'character name : dialogue \n')
			# Skip first section (before first character name)
			for i in range(1, len(sections), 2):
				character = sections[i]
				dialogue = sections[i+1] if i+1 < len(sections) else "" # Last section may not have dialogue
				
				script_queue.put(character + ":")
				
				# Clean and segment dialogue
				C1_PATTERN = re.compile(r'[\[][\w\d\s\.\,\-\'\"\’\:\(\)]+[\]]', re.MULTILINE) 			# Remove scene instructions located in brackets
				C2_PATTERN = re.compile(r'(?<=\:)[\w\d\s\.\!\?\,\'\"\“\”\’]+(?=[\n]+)', re.MULTILINE) 	# Define dialogue line starting with a colon and ending with a newline
				C3_PATTERN = re.compile(r'(?<=\d)[\:]{1}(?=\d)', re.MULTILINE) 							# Replace time separator with a period (e.g. 12:30 -> 12.30) 
				dialogue = re.sub(C1_PATTERN, '', dialogue)
				dialogue = re.sub(C3_PATTERN, '.', dialogue)
				dialogue = re.findall(C2_PATTERN, dialogue) 
								
				for sentence_group in dialogue:
					sentence_group = [ 
						sentence.strip() for sentence in re.split(r'\n', sentence_group)
						if sentence.strip() and len(sentence.strip()) > 3
					]
						
					for sentence in sentence_group:
						sentence = re.sub(r'(?<=[\d])([.]+)(?=[\d]{2}[\s][A-Z])',':', sentence) 		# Replace time separator with a colon (e.g. 12.30 -> 12:30)
						script_queue.put(sentence.strip())

		script_entries = []
		characters, dialogues = [], []
		dialogue_count = 1

		while not script_queue.empty(): 
			queue_item = script_queue.get()

			is_character_name = re.search(CHARACTER_PATTERN, queue_item)
			
			if not is_character_name:
				dialogues.append(queue_item)
				dialogue_count += 1
			else:
				characters.append(queue_item.strip(':'))
				
			if len(characters) > 1:
				for dialogue_line in dialogues[-dialogue_count:]:
					script_entries.append({characters[0] : dialogue_line})
				characters.pop(0)
				dialogues.clear()

		#script_entries.append({characters[0] : ' '.join(dialogues[-dialogue_count:])})
		df = pd.concat([pd.DataFrame.from_dict(_, columns=['dialogue'], orient='index') for _ in script_entries])   
		df = df.reset_index().rename(columns={'index':'part'})
		
		logging.info(f"Extracted {len(df)} script entries")
		return df[OFFSET:] #.reset_index(drop=True)
	
	def align(self) -> pd.DataFrame:
		"""Align subtitle and script dialogues based on text similarity."""
		#FIXME 	Problem: script lines often span over multiple srt lines, i.e. multiple timecodes
		# 		Example: "Marty, you're going to have to do something about those clothes. You walk around town dressed like that, you're liable to get shot." 
		# 				  00:35:26,541 		 											   00:35:28,585
		# 		Solution: split script lines into muliplte sentences... but how to know when to split?

		df_srt = self.extract_srt()
		df_script = self.extract_script()
		
		logging.info("Aligning entries now. Might take a while...")

		matches = []
		matched_script_indices = set()  
		last_match_idx = 0  
		window_size = 10
		
		# For each subtitle dialogue, find best matching script dialogue
		for i, srt_row in df_srt.iterrows():			
			best_match = None
			best_score = CONFIG['SIMILARITY_THRESHOLD']
			
			# Define search window (centered around last position)
			window_start = max(0, last_match_idx - window_size//4)  
			window_end = min(len(df_script), last_match_idx + window_size) 
			
			# Search within window and not yet matched
			for j in range(window_start, window_end):
				if j in matched_script_indices:
					continue 
					
				score = Toolbox.jaccard_similarity(srt_row['dialogue'], df_script.loc[j, 'dialogue'])
				if score > best_score:
					best_score = score
					best_match = j
			
			if best_match is not None:
				matched_script_indices.add(best_match)  
				last_match_idx = best_match + 1 
				
				matches.append({
					'timecode': srt_row['timecode'],
					'part': df_script.loc[best_match, 'part'],
					'srt_dialogue': srt_row['dialogue'],
					'script_dialogue': df_script.loc[best_match, 'dialogue'],
					'similarity': best_score
				})
		
		# Sort by timecode to maintain chronological order
		result_df = pd.DataFrame(matches)
		
		if not result_df.empty:
			result_df = result_df.sort_values('timecode').reset_index(drop=True)

		logging.info(
			f"""Aligned {len(matches)} subtitle dialogues with script dialogues.""")
		return pd.DataFrame(matches)

	def to_excel(self, path_out:Path) -> None:
		"""Export aligned data to Excel file."""
		df = self.align()

		if df is not None:
			df = df.dropna().reset_index() 	
			
			if not path_out.exists(): 
				path = path_out
			else: 
				path = Toolbox.numbered_file('xlsx')
			
			with pd.ExcelWriter(path, engine='openpyxl') as w:
				df.to_excel(w, sheet_name='bttf')
				logging.info(f"Data saved to {path}")
		else:
			logging.error("No data to save yet.")
			return



def main() -> None:
	
	STAGING_DIRECTORY = Toolbox.staging_tmpDir_create()
	
	logging.info('Starting data collection and processing')

	try:
		if not CONFIG['OUTPUT_DIR'].exists():
			CONFIG['OUTPUT_DIR'].mkdir(parents=True, exist_ok=True)
			CONFIG['OUTPUT_DIR'].joinpath('archive').mkdir(exist_ok=True)

		CollectDataFromWeb(staging_dir=STAGING_DIRECTORY).collect()
		ProcessData(staging_dir=STAGING_DIRECTORY).to_excel(path_out=CONFIG['OUTPUT_DIR'] / 'bttf.xlsx')
		logging.info('Data collection and processing completed.')

	except Exception as e:
		logging.error(f'Error: {e}', exc_info=True)  # w/ stack trace
		logging.info('Data collection and processing failed.')
		sys.exit(1)

	finally:
		if STAGING_DIRECTORY.exists():
			Toolbox.staging_tmpDir_clear(staging_dir=STAGING_DIRECTORY)
		logging.shutdown()
		sys.exit(0)



if __name__ == "__main__":
	main()
