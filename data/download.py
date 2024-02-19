import gdown
from zipfile import ZipFile
import os
import argparse

def main(args):
    # Download raw data
    if args.download_data and not os.path.exists('data/raw.zip'):
        output = "data/raw.zip"
        id = "1-ThO1XPDK283cHzK-4O3GeY2CTlGWmt1"
        gdown.download(id=id, output=output)

        with ZipFile('data/raw.zip', 'r') as zipObj:
            zipObj.extractall('data/')

    if not os.path.exists('models/saved_models/'):
        os.makedirs('models/saved_models/')
        
    # Download best models
    if args.download_models and not os.path.exists('models/saved_models/best_models.zip'):
        output = "models/saved_models/best_models.zip"
        id = "1NbS5UoG9ju39s21MOlZ2AsA8_wFZwLsA"
        gdown.download(id=id, output=output)

        with ZipFile('models/saved_models/best_models.zip', 'r') as zipObj:
            zipObj.extractall('models/saved_models/')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download data and models for scene recognition.')
    parser.add_argument('--download_data', type=bool, default=False, help='Download raw data.')
    parser.add_argument('--download_models', type=bool, default=False, help='Download best models.')
    args = parser.parse_args()
    main(args)