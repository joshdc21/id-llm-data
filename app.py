import requests as req
from bs4 import BeautifulSoup as bs
import csv
from tqdm import tqdm
from datetime import timezone, timedelta, datetime
import json
from urllib.parse import urlparse
# pyrefly: ignore [missing-import]
from langdetect import detect
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasketch import MinHash, MinHashLSH

model_name = "cahya/gpt2-small-indonesian-522M"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Header so you don't get blocked by the server
hades = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                 'AppleWebKit/537.36 (KHTML, like Gecko) '
                 'Chrome/127.0.0.0 Safari/537.36'
}

# awal 2026 - sekarang
WIB = timezone(timedelta(hours=7))
start_date = datetime(2026, 1, 1, tzinfo=WIB)
end_date = datetime.now(WIB)

# Daftar URL dan Kategori yang ingin di scrape
laman_detik = [
    ("olahraga", "https://sport.detik.com/indeks?page={page}"),
    ("ekonomi", "https://finance.detik.com/indeks?page={page}"),
    ("internasional", "https://news.detik.com/internasional/indeks?page={page}"),
    ("nasional", "https://news.detik.com/indeks?page={page}"),
    ("teknologi", "https://inet.detik.com/indeks?page={page}")
]

laman_cnn = [
    ("nasional", "https://www.cnnindonesia.com/nasional/indeks/3?page={page}"),
    ("internasional", "https://www.cnnindonesia.com/internasional/indeks/6?page={page}"),
    ("ekonomi", "https://www.cnnindonesia.com/ekonomi/indeks/5?page={page}"),
    ("teknologi", "https://www.cnnindonesia.com/teknologi/indeks/8?page={page}"),
    ("olahraga", "https://www.cnnindonesia.com/olahraga/indeks/7?page={page}")
]
laman_tribun = [
    ("nasional", "https://www.tribunnews.com/index-news/nasional?page={page}"),
    ("internasional", "https://www.tribunnews.com/index-news/internasional?page={page}"),
    ("ekonomi", "https://www.tribunnews.com/index-news/bisnis?page={page}"),
    ("teknologi", "https://www.tribunnews.com/index-news/techno?page={page}"),
    ("olahraga", "https://www.tribunnews.com/index-news/sport?page={page}")    
]

# map bulan dan hari ke english
month = {
    "Jan": "Jan", "Feb": "Feb", "Mar": "Mar", "Apr": "Apr",
    "Mei": "May", "Jun": "Jun", "Jul": "Jul", "Agu": "Aug", "Agt": "Aug",
    "Sep": "Sep", "Okt": "Oct", "Nov": "Nov", "Des": "Dec",
    "Januari": "Jan", "Februari": "Feb", "Maret": "Mar", "April": "Apr",
    "Agustus": "Aug", "September": "Sep", "Oktober": "Oct", "November": "Nov", "Desember": "Dec"
}

day = {
    "Senin": "Mon", "Selasa": "Tue", "Rabu": "Wed", "Kamis": "Thu",
    "Jumat": "Fri", "Sabtu": "Sat", "Minggu": "Sun"
}

# Inisialisasi MinHash LSH
num_perm = 128
mh_lsh = MinHashLSH(threshold=0.7, num_perm=num_perm)

def get_min_hash(text, num_perm=num_perm):
    m = MinHash(num_perm=num_perm)
    words = text.lower().split()
    n = 5
    for i in range(len(words) - n + 1):
        kgram = " ".join(words[i:i+n])
        m.update(kgram.encode('utf-8'))
    return m 

def parse_date(date_str):
    if not date_str:
        return None

    # Hapus | dan •
    if "|" in date_str:
        date_str = date_str.split("|")[-1].strip()
    if "•" in date_str:
        date_str = date_str.split("•")[-1].strip()

    date_str = date_str.replace("WIB", "").strip()

    # Case 1: "x menit yang lalu", "x jam yang lalu", "x hari yang lalu"
    if "lalu" in date_str:
        parts = date_str.split()
        try:
            # Index pertama angka(Index Angka 7 di 7 menit yang lalu)
            idx = next(i for i, p in enumerate(parts) if p.isdigit())
            angka = int(parts[idx])
            unit = parts[idx + 1].lower()

            now = datetime.now(WIB)

            if "menit" in unit:
                return now - timedelta(minutes=angka)
            elif "jam" in unit:
                return now - timedelta(hours=angka)
            elif "hari" in unit:
                return now - timedelta(days=angka)
            else:
                return now
        except (ValueError, IndexError, StopIteration):
            pass

    # Case 2: "Hari, x bulan tahun 00:00"
    for indo, eng in month.items():
        date_str = date_str.replace(indo, eng)
    for indo, eng in day.items():
        date_str = date_str.replace(indo, eng)

    for fmt in ["%a, %d %b %Y %H:%M", "%d %b %Y %H:%M", "%a, %d %b %Y %H:%M:%S", "%d %b %Y %H:%M:%S"]:
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=WIB)
        except ValueError:
            continue

    return None    

def symbol_to_word_ratio(text):
    words = re.findall(r"\b\w+\b", text)
    symbols = re.findall(r'[@#$%^&*~|\\{}\[\]]', text)

    total = len(symbols) / max(len(words), 1)

    if total > 0.2:
        return False
    
    return True

def digit_to_word_ratio(text):
    words = re.findall(r"\b\w+\b", text)
    digits = re.findall(r"\b\d+\b", text)

    total = len(digits) / max(len(words), 1)

    if total > 0.3:
        return False
    
    return True

def repetitive_text_ratio(text, n=8):
    words = text.split()
    chunks = [ " ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    
    if not chunks:
        return True    

    unique_chunks = set(chunks)
    repeated_chunks = len(chunks) - len(unique_chunks)
    
    if repeated_chunks / len(chunks) > 0.3:
        return False
    
    return True

def all_caps_ratio(text):
    words = text.split()
    caps = sum(1 for w in words if w.isupper() and len(w) > 1)
    ratio = caps / len(words)

    if ratio > 0.5:
        return False

    return True

def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            labels=inputs["input_ids"]
        )

    loss = outputs.loss
    perplexity = torch.exp(loss)

    return perplexity.item()

def scrape_article(laman, max_page):
    total_articles = 0
    # Buat JSONL buat news site
    with open("articles.jsonl", "w", encoding="utf-8") as f:

        # iterate semua laman yang ada di daftar laman
        for category, url_template in laman:
            print(f"\n--- Scraping Kategori: {category} ---")
            
            # substitute {page} dengan angka mulai dari 1 sampai max_page
            for page in range(1, max_page + 1):
                url = url_template.format(page=page)
                print(f"Scraping page {page} | url {url}")

                # Mengdownload index html menggunakan header user agent
                res = req.get(url, headers=hades).text
                sop = bs(res, 'lxml')

                """
                Locate article card elements
                - CNN/Detik pake <article>
                - Tribun pake <ul class = "lsi"> atau <ul id = "latestul">
                """
                lists = sop.find_all("article") or sop.select("ul.lsi > li, #latestul > li") or sop.select("a.article_inview, a[dtr-act='artikel']")

                if not lists:
                    tqdm.write(f"No more articles for {category} on page {page}")
                    break

                # Loading bar untuk setiap page
                for x in tqdm(lists, desc=f"[{category}] Halaman {page}", unit="artikel"):
                    try:
                        # Headline
                        h3_tag = x.find('h3', class_='media__title') or \
                                 x.find('h2', class_='title') or x.find('h3') or \
                                 x.find('h2') or x.find('h4')
                        headline = h3_tag.get_text(strip=True) if h3_tag else ""

                        # URL
                        a_tag = (x if x.name == "a" else None) or \
                                (h3_tag.find("a") if h3_tag else None) or \
                                 x.find("a")
                        if not a_tag or "href" not in a_tag.attrs:
                            continue
                        article_url = a_tag["href"].strip()

                        # Source website
                        source = urlparse(article_url).netloc

                        if not headline:
                            headline = a_tag.get_text(strip=True)

                        # Date string
                        date_tag = (
                            x.find('div', class_='media__date') 
                            or x.find('span', class_='date') 
                            or x.find('span', class_='text-cnn_black_light3')
                            or x.find('time')
                            or x.find('span', class_='grey')
                        )
                        date_str = date_tag.get_text(strip=True) if date_tag else ""

                        # Parse Date(From "Kamis, 10 Sep 2026 19:00" ke DateTime)
                        parsed_dt = parse_date(date_str)
                        if not parsed_dt:
                            tqdm.write(f"Fail to parse date: '{date_str}'")
                            continue

                        if not (start_date <= parsed_dt <= end_date):
                            continue

                        # Get article content
                        res_ = req.get(article_url, headers=hades).text
                        sop_ = bs(res_, 'lxml')

                        # Container utama content article
                        content_div = (
                            sop_.find("div", class_="detail__body-text itp_bodycontent")
                            or sop_.find("div", class_="detail__body-text")
                            or sop_.find("div", class_="detail-text")
                            or sop_.find("div", class_="itp_bodycontent")
                            or sop_.find("div", class_="detail__body")
                            or sop_.find("div", class_="txt-article")
                            or sop_.find("div", class_="side-article")
                        )   

                        """
                        Ambil semua paragraph, terus di combine
                        remove clutters kayak advertisement atau scroll ads
                        """
                        clutters = [
                            "ADVERTISEMENT",
                            "SCROLL TO RESUME CONTENT",
                            "SCROLL TO CONTINUE WITH CONTENT",
                            "Baca juga:"
                        ]

                        if content_div:
                            paragraphs = [p.get_text(strip=True) for p in content_div.find_all("p")]
                            content = ' '.join(paragraphs)
                            content_ = content
                            for c in clutters:
                                content_ = content_.replace(c, "")
                            # Remove [Gambas:...] with regex
                            content_ = re.sub(r"\[Gambas.*?\]", "", content_).strip()
                            content_ = re.sub(r"Diskusi\s+beasiswa.*", "", content_, flags=re.IGNORECASE)
                            content_ = content_.strip()
                        else:
                            content_ = ''

                        # Skip artikel yang hanya 50 words or less
                        if not content_ or len(content_.split()) < 50:
                            continue
                        
                        # Skip corrupted text
                        if not symbol_to_word_ratio(content_):
                            tqdm.write(f"Filtered out article with too many symbols: {headline[:40]}")
                            continue
                        
                        # Skip stats table/raw stock table
                        if not digit_to_word_ratio(content_):
                            tqdm.write(f"Filtered out article with too many digits: {headline[:40]}")
                            continue
                        
                        # Skip duplicates
                        if not repetitive_text_ratio(content_):
                            tqdm.write(f"Filtered out article with repetitive text: {headline[:40]}")
                            continue
                        
                        # Skip all caps
                        if not all_caps_ratio(content_):
                            tqdm.write(f"Filtered out article with all caps: {headline[:40]}")
                            continue
                        
                        # Skip artikel yang bukan bahasa indonesia
                        try:
                            if detect(content_) != "id":
                                tqdm.write(f"Filtered out non-Indonesian article: {headline[:40]}")
                                continue
                        except Exception:
                            continue

                        """
                        1. Create minhash for the content
                        2. Search LSH Index for similar articles
                        3. If similar skip the article
                        4. Keep the article for next checking if not similar
                        """
                        article_hash = get_min_hash(content_)
                        nearest = mh_lsh.query(article_hash)

                        if nearest:
                            tqdm.write(f"Filtered out duplicate article: {headline[:40]}")
                            continue
                        
                        mh_lsh.insert(article_url, article_hash)

                        # perplexity based scoring
                        perplexity = calculate_perplexity(content_)
                        if perplexity > 500:
                            tqdm.write(f"Filtered out low-quality article (perplexity > 500): {headline[:40]}")
                            continue

                        # Penanda kalau success ngeproses artikelnya
                        tqdm.write(f"Success [{category}]: {headline[:50]}")
                        total_articles += 1

                        # Simpan di csv dengan Kategori
                        f.write(json.dumps({
                            "Link": article_url,
                            "Source":source,
                            "Kategori": category,
                            "Title": headline,
                            "Tanggal": parsed_dt.strftime("%Y-%m-%d %H:%M"),
                            "Isi Berita": content_
                        }, ensure_ascii=False) + "\n")

                    except Exception as e:
                        tqdm.write(f"Error scraping {x.find('a')['href'] if x.find('a') else 'article'} | {str(e)}")

    print(f"\nFinished scraping! Saved total {total_articles} articles to articles.jsonl")

def main():
    scrape_article(laman_detik + laman_cnn + laman_tribun, 5)

if __name__ == "__main__":
    main()