import warnings
warnings.filterwarnings('ignore')
import requests
import json
import os
import glob
import time
from fake_useragent import UserAgent

"""
If you encounter a 443 error, you can try to increase the `timeout`. 
Of course, it should be noted that you can first visit `https://civitai.com` to see if the network supports access to the website. 
For API details, you can refer to `https://github.com/civitai/civitai/wiki/REST-API-Reference`.
"""

ua = UserAgent()

if not os.path.exists('data'):
    os.makedirs('data')

res=requests.get("https://civitai.com/api/v1/models?page=1", headers={'user-agent': ua.random}, timeout=20, verify=False)
metadata = json.loads(res.content)['metadata']['totalPages']
print(f'The total page number of meta data is {metadata}')

file_exist = glob.glob("data/*.json")
print(f'Already {len(file_exist)}')
i = 1
while i <= metadata:
    url = f"https://civitai.com/api/v1/models?page={i}"
    if f"data\\{i}.json" in file_exist:
        print(f"exist {url}")
        i += 1
        continue
    try:
        print(f'processing {url}')
        res=requests.get(url, headers={'user-agent': ua.random}, timeout=20, verify=False)
        res.encoding="utf-8"
        
        if "window._cf_translation" in str(res.content) or '<!DOCTYPE html>' in str(res.content):
            raise KeyError
            
        with open(f"data/meta-{i}.json", "wb") as f:
            f.write(res.content)
            
        print(res.status_code, {url})
        i += 1
    except Exception as e:
        time.sleep(3)
        continue


