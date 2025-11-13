#!/usr/bin/env python3
"""
Download datasets from Synapse using their API.
Usage: python fetch_from_synapse.py --synapse_id SYNAPSE_ID --output_dir OUTPUT_DIR
"""

import synapseclient 
import synapseutils 
import os
import argparse
import sys
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description='Download datasets from Synapse')
    parser.add_argument("--synapse_id", type=str, required=True, help='Synapse project/folder ID')
    parser.add_argument("--output_dir", type=str, required=True, help='Output directory for downloaded files')
    args = parser.parse_args()
    
    # Check for access token
    access_token = os.getenv("SYNAPSE_ACCESS_TOKEN")
    if not access_token:
        print("Error: SYNAPSE_ACCESS_TOKEN environment variable is not set")
        print("Please set it or create a .env file with: SYNAPSE_ACCESS_TOKEN=your_token")
        print("Get your token from: https://python-docs.synapse.org/en/stable/tutorials/authentication/")
        sys.exit(1)
    
    print(f"Downloading from Synapse ID: {args.synapse_id}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        syn = synapseclient.Synapse() 
        syn.login(authToken=access_token)
        
        print("Successfully logged into Synapse")
        print("Starting download...")
        
        files = synapseutils.syncFromSynapse(syn, args.synapse_id, path=args.output_dir)
        
        print(f"Download completed successfully!")
        print(f"Downloaded {len(files)} files to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()