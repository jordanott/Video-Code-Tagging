import sys
sys.path.append('Model/')
from model import cnn

def to_seconds(t):
    count = 0
    total = 0
    for i in reversed(t.split(':')):
        total += 60**count * int(i)
        count += 1
    return total

def get_data():
    DATA_PATH = 'DataCollection/'
    links = open(DATA_PATH+'links.txt')
    data = {}
    for line in links:
        link,name,code_time,nocode_time = line.split(' | ')
        link,name,code_time,nocode_time = link.strip(),name.strip(),code_time.strip(),nocode_time.strip()
        data[name] = [code_time.split(','),nocode_time.split(',')]
    return data

data = get_data()
print data
