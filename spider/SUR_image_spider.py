import json
import glob
import time
import os
import requests
import argparse
import warnings
warnings.filterwarnings('ignore')
from fake_useragent import UserAgent
ua = UserAgent(verify_ssl=False)

if not os.path.exists('image'):
    os.makedirs('image')

with open(f"data/data.json", "r") as f:
    data = json.load(f)

file_exist = glob.glob("image/*.png")
i = 0
while i <= len(data["items"]):
    image = data["items"][i]
    try:
        url_list = image["url"].split("/")
        image_name = url_list[-3]
        data["items"][i]["image_name"] = image_name
        if f"image\\{image_name}.png" in file_exist:
            i += 1
            continue
        res=requests.get(image["url"], headers={'user-agent': ua.random}, timeout=20, verify=False)
        
        if res.status_code == 403:
            print(res.status_code, image["url"])
            time.sleep(3)
            continue
        
        with open(f"image/{image_name}.png", "wb") as f:
            f.write(res.content)
        print(res.status_code, image["url"])
        i += 1
    except Exception as e:
        print("error", image["url"])
        print(e)
        time.sleep(3)
        continue
