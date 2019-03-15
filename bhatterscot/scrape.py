import os
import re

import requests
from bs4 import BeautifulSoup

from bhatterscot.config import BASE_DIR, CORPUS_NAME

HTML_PARSER = 'html.parser'


def scrape_url(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, HTML_PARSER)
    links = [l.get('href') for l in soup.find_all(name='a', attrs={'class': 'category-page__member-link'})]
    links = [l for l in links if
             type(l) == str and l.startswith('/wiki/') and ':' not in l and 'sitemap' not in l.lower()]
    for link in links:
        game_title = ''.join([c for c in link.split('wiki/')[-1] if c.isalnum() or c in ['_', '-']])
        print(f'Scraping "{game_title}"...', end='')
        r = requests.get(f'https://transcripts.fandom.com{link}')
        soup = BeautifulSoup(r.content, HTML_PARSER)
        anchors = soup.findAll('a')
        for anchor in anchors:
            anchor.extract()
        lines = [line for line in soup.get_text().split('\n')
                 if ':' in line[:20] and line[0].isupper() and len(line.split(':')[1]) > 1
                 and 'content' not in line.split(':')[0].lower()]
        lines = [re.sub(r'([(\[<]).*?([)\]>])', '\g<1>\g<2>', line) for line in lines]
        lines = [re.sub('[\[\]()<>]', '', line) for line in lines]
        lines = [line.split(':', 1)[1].strip().lower() for line in lines]
        if len(lines) < 10:
            print('too few lines.')
            continue
        filepath = os.path.join(BASE_DIR, CORPUS_NAME, f'{game_title}.txt')
        os.makedirs(os.path.join(BASE_DIR, CORPUS_NAME), exist_ok=True)
        with open(filepath, 'w') as f:
            f.writelines([line + '\n' for line in lines])
        print('DONE!')


def scrape_video_game_transcripts():
    scrape_url('https://transcripts.fandom.com/wiki/Category:Video_Games')
    scrape_url('https://transcripts.fandom.com/wiki/Category:Video_Games?from=Robot+Carnival')


if __name__ == '__main__':
    scrape_video_game_transcripts()
