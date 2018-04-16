from datetime import datetime
import os, os.path
import json
from bs4 import BeautifulSoup
import youtube_dl
from pycaption import DFXPReader
from concurrent.futures import ThreadPoolExecutor

THREAD_COUNT = 32

"""
Grab the captions from every video in a YouTube watch-history.html from Google Takeout

Outputs id_to_filename.json which lists the ids and file prefixes (<VIDEO_ID>:<VIDEO_TITLE>) 
for all of the videos. (files are in "<VIDEO_TITLE>-<VIDEO_ID>.txt" format)

Usage: 
    0. `sudo pip install -r requirements.txt`

    1. move your YouTube folder from Takeout into ../ from this script

    2. `python3 extractor.py` # this takes a LONG time

    3. 
    ```
    rm -rf *.ttml  # optional 

    mkdir data
    mv *.txt data/
    mv *.jpg data/

    4.
    ```
    cd ..
    python3 tf_idf.py
    ```

    5. Open Jupyter notebook to Plotting.ipynb or Plotting-copy
"""
# TODO: give option for checkpoints
# TODO: save also the publisher of the video

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        func_out = func(*args, **kwargs)
        end_time = datetime.now()
        print('This took: {}'.format(end_time - start_time))
        return func_out
    return wrapper

def partition(data, pcount):
    plen = int(len(data) / pcount)
    result = []
    current = 0
    for i in range(pcount-1):
        result.append(data[current:current+plen])
        current += plen
    result.append(data[current:])
    return result

@timeit
def download_captions():
    # GRAB LINKS
    watch_history_path = '../YouTube/history/watch-history.html'
    soup = BeautifulSoup(open(watch_history_path, encoding='utf8'), 'html.parser')
    links = [link.attrs['href'] for link in soup.find_all('a')] # grab links
    tmp_links = []
    for link in links:
        if not '.com/channel/' in link:
            tmp_links.append(link)
    links = list(set(tmp_links)) # remove dupes
    print("Found {} unique video links in watch-history.html".format(len(links))) 
    # GET IDS AND TITLES OF ALL VIDEOS
    class Logger(object):
        def __init__(self):
            self.logs = []
        def debug(self, msg):
            if not msg.startswith('['):
                self.logs.append(msg)
        def warning(self, msg):
            print("WARNING: "+msg)
        def error(self, msg):
            print(msg)

    # MAIN PARAMS
    def progress_hook(d):
        part = d['fragment_index']
        tot = d['fragment_count']
        print("{}/{} captions downloaded".format(part, tot))
        if d['status'] == 'finished':
            print('Done downloading, now converting ...')

    logger = Logger()
    ydl_opts = {
        'skip_download': True,
        'ignoreerrors': True,
        'forceid': True,
        'forcefilename': True,
        'writeautomaticsub': True,
        'writethumbnail': True,
        'subtitlesformat': 'ttml',
        'logger': logger,
        'progress_hooks': [progress_hook],
    }
    def download_links(links):
        # DOWNLOAD CAPTIONS AND THUMBNAILS
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([links])
    with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
        executor.map(download_links, links)
    # threads = [Thread(target=download_links, args=(new_links,)) for new_links in partition(links, THREAD_COUNT)]
    # for t in threads:
    #     t.daemon = True
    #     t.start()
    # for t in threads:
    #     t.join()
    

def find_valid_files():
    pass

@timeit
def parse_captions(path='./'):
    # GRAB ALL VALID CAPTION FILES AND THEIR FILENAMES
    filenames = []
    ids = []
    for file in os.listdir(path):
        if file.endswith(".ttml"):
            filenames.append(file[:-8])
            ids.append(file[-19:-8])
    id_to_filename = {ids[i] : filenames[i] for i in range(len(filenames))}

    # PARSE CAPTIONS AND PUT THEM IN NICE FORMAT
    cap_reader = DFXPReader()

    good_id_to_filename = {} # only save ones that have captions
    n = 0
    N = len(id_to_filename)
    for id in id_to_filename:
        n+=1
        ttml_file = path+id_to_filename[id]+'.en.ttml'
        text_file = path+id_to_filename[id]+'.txt'
        image_file = path+id_to_filename[id]+'.jpg'
        # check if file exists. if not, this video has no autocaptions
        if os.path.isfile(ttml_file) and os.path.isfile(image_file):
            good_id_to_filename[id] = id_to_filename[id]
            with open(ttml_file, 'r', encoding='utf8') as f:
                ttml_txt = f.read()
                caption_set = cap_reader.read(ttml_txt)
                captions = caption_set.get_captions('en-US')
                caption_text = ' '.join([caption.get_text() if caption is not None else '' \
                                                            for caption in captions])
            with open(text_file, 'w', encoding='utf8') as f:
                f.write(caption_text)
            if n % 100 == 0:
                print("{}/{} captions grabbed".format(n, N))

    json.dump(good_id_to_filename, open('id_to_filename.json', 'w'), separators=(', \n', ': '))
    print("Saved captions and thumbnails for {} videos".format(len(good_id_to_filename)))
    print("Done.")

if __name__ == '__main__':
    download_captions()
    parse_captions(path='./')
