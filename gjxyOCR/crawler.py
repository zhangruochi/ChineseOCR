from icrawler.builtin import BaiduImageCrawler
from pathlib import Path


root = Path("original_captchas")

download_root = Path("download")


all_phrases = []
for p in root.glob("*.png"):
    all_phrases.append(p.name.split(".")[0])

all_phrases = list(set(all_phrases))

print(all_phrases)


for name in all_phrases:
    p = download_root / name
    if not p.exists():
        p.mkdir()



for name in all_phrases:
    baidu_crawler = BaiduImageCrawler(storage={'root_dir': str(download_root/name)})
    baidu_crawler.crawl(keyword=name, offset=0, max_num=200,
                        min_size=(200,200), max_size=None)



