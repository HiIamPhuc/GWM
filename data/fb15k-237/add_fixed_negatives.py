"""
Quick script to add fixed negatives to FB15k-237 processed data.

Run this after preparing the data to generate fixed negative samples.
"""

import sys
sys.path.insert(0, '../../gwm-rnn/relation-prediction')

from generate_negatives import main

if __name__ == "__main__":
    # Automatically configure for FB15k-237
    import argparse
    
    # Override sys.argv to set default path
    sys.argv = [
        'add_fixed_negatives.py',
        '--data_dir', '../../gwm-rnn/data/fb15k-237/processed/relation-prediction',
        '--num_negatives', '10',
        '--seed', '42'
    ]
    
    print("Generating fixed negatives for FB15k-237...")
    main()
