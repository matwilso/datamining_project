import sqlite3

DB_NAME = 'captions.db'

def js(s1, s2):
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union

MICRO_IN_MINUTE = 60 * 1000 * 1000

def bag_of_words(string):
    return set(string.split())

class Caption:
    def __init__(self, caption, video, start, stop):
        self.video = video
        self.caption = caption
        self.start = start
        self.stop = stop
        self.bag = bag_of_words(caption)

def yt_time(time):
    seconds = time // 1000 // 1000
    minutes = seconds // 60
    seconds -= minutes * 60
    hours = minutes // 60
    minutes -= hours * 60
    result = ''
    if hours > 0:
        result += '{}h'.format(hours)
    if minutes > 0:
        result += '{}m'.format(minutes)
    if seconds > 0:
        result += '{}s'.format(seconds)
    return result

def yt_link(video_id, start):
    return f'https://www.youtube.com/watch?v={video_id}&t={yt_time(start)}'

captions: list = []
with sqlite3.connect(DB_NAME) as conn:
    # Don't care that much about transactions since not updating while running this.
    curs = conn.cursor()
    videos = [row[0] for row in curs.execute('SELECT rowid FROM video')]
    for video in videos:
        start = 0
        end = MICRO_IN_MINUTE
        while True:
            curs.execute('''
                SELECT caption 
                FROM transcription 
                WHERE video = ? AND start >= ? AND stop < ?
                ORDER BY start ASC
            ''', (video, start, end,))
            result = curs.fetchall()
            if len(result) == 0:
                break
            caption = ''.join(row[0] for row in result)
            captions.append(Caption(caption, video, start, end))
            start = end
            end += MICRO_IN_MINUTE

print(f'Searching over {len(captions)} caption documents.')
while True:
    print('Input a search query:')
    query = input()
    query_bag = bag_of_words(query)
    top5 = sorted(captions, key=lambda caption: -js(caption.bag, query_bag))[:5]
    with sqlite3.connect(DB_NAME) as conn:
        curs = conn.cursor()
        for top in top5:
            curs.execute('SELECT youtube_id FROM video WHERE rowid=?', (top.video,))
            yt_id = curs.fetchone()[0]
            print(yt_link(yt_id, top.start))


