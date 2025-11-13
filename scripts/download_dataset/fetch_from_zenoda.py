#!/usr/bin/env python3
"""
Download datasets from Zenodo using their API.
Usage: python fetch_from_zenoda.py --record_id RECORD_ID --output_folder OUTPUT_FOLDER
"""

import os
import requests
import argparse
import sys
import dotenv
dotenv.load_dotenv()
ACCESS_TOKEN = os.getenv('ZENODO_ACCESS_TOKEN')

def main():
    parser = argparse.ArgumentParser(description='Download datasets from Zenodo')
    parser.add_argument('--record_id', required=True, help='Zenodo record ID')
    parser.add_argument('--output_folder', required=True, help='Output folder for downloaded files')
    
    args = parser.parse_args()
    
    record_id = args.record_id
    output_folder = args.output_folder
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Downloading from Zenodo record: {record_id}")
    print(f"Output folder: {output_folder}")
    
    # Get the metadata of the Zenodo record
    r = requests.get(f"https://zenodo.org/api/records/{record_id}", params={'access_token': ACCESS_TOKEN})
        
    if r.status_code != 200:
        print(f"Error retrieving record: {r.status_code} {r.text}")
        sys.exit(1)
    
    # Extract download URLs and filenames

    record_data = r.json()
    files = record_data['files']
    download_urls = [f['links']['self'] for f in files]
    filenames = [f['key'] for f in files]
    print(f"Total files to download: {len(download_urls)}")
    
    # Download each file
    for index, (filename, url) in enumerate(zip(filenames, download_urls)):
        file_path = os.path.join(output_folder, filename)

        print(f"Downloading file {index}/{len(download_urls)}: {filename} -> {file_path}")

        with requests.get(url, params={'access_token': ACCESS_TOKEN}, stream=True) as r:
            r.raise_for_status()  # Raise an error for failed requests
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):  # Download in chunks
                    f.write(chunk)

        print(f"Completed: {filename}")
        
    print("All downloads completed successfully!")

if __name__ == "__main__":
    main()