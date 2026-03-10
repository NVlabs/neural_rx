#!/usr/bin/env python3
import pickle
import sys
import os
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python print_results.py <results_file>")
        sys.exit(1)

    filename = sys.argv[1]
    
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        sys.exit(1)

    print(f"Loading results from: {filename}")
    
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
            
        if len(data) != 3:
            print("Error: Unexpected data format. Expected [ebno_db, BERs, BLERs]")
            sys.exit(1)
            
        ebno_db, BERs, BLERs = data
        
        print("\n" + "="*40)
        print("SNR Points (Eb/No dB)")
        print("="*40)
        print(ebno_db)
        
        print("\n" + "="*40)
        print("Bit Error Rates (BER)")
        print("="*40)
        for key, val in BERs.items():
            # key is (system_name, num_users, mcs_index)
            print(f"System: {key[0]}")
            print(f"Users: {key[1]}, MCS Index: {key[2]}")
            print(f"BER: {val}")
            print("-" * 20)
            
        print("\n" + "="*40)
        print("Block Error Rates (BLER)")
        print("="*40)
        for key, val in BLERs.items():
            print(f"System: {key[0]}")
            print(f"Users: {key[1]}, MCS Index: {key[2]}")
            print(f"BLER: {val}")
            print("-" * 20)

    except Exception as e:
        print(f"Failed to load or parse file: {e}")

if __name__ == "__main__":
    main()
