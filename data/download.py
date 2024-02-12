import gdown
from zipfile import ZipFile
import os


# https://drive.google.com/file/d/1-ThO1XPDK283cHzK-4O3GeY2CTlGWmt1/view?usp=sharing

if not os.path.exists('data/raw.zip'):
    output = "data/raw.zip"
    id = "1-ThO1XPDK283cHzK-4O3GeY2CTlGWmt1"
    gdown.download(id=id, output=output)
    # gdown.download(url, output)

with ZipFile('data/raw.zip', 'r') as zipObj:
    zipObj.extractall('data/')
