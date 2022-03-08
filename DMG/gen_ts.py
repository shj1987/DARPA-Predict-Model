import sys
import json
from collections import defaultdict
import pandas as pd
import numpy as np
from datetime import datetime as dt
import calendar

count = 0
series = {}
frame_name = []
with open('WeSTClass/news_manual/classes.txt') as IN:
    for line in IN:
        frame_name.append(line.strip().split(':')[1])

# Dump all the time series
def dump_series(news_event, fn):
    news_time = dict()
    for n in frame_name:
        tmp = dict()
        for d in all_dates:
            stamp = str(int(calendar.timegm(dt.strptime(d, "%Y-%m-%d").timetuple()))) + '000'
            tmp[stamp] = news_event[n][d]
        news_time[n] = tmp
#     json.dump({i:json.dumps({'EventCount': news_time[i]}) for i in news_time}, open(fn, 'w'))
    json.dump({i:json.dumps(news_time[i]) for i in news_time}, open(fn, 'w'))

    
series_start_date = sys.argv[2]
series_end_date = sys.argv[3]

all_dates = [i.strftime("%Y-%m-%d") for i in pd.date_range(start=series_start_date,end=series_end_date).to_pydatetime().tolist()]



for target_method in [
            'Leidos',
            #'hybrid3',
            'Westclass',
            #'Westclass_top1000',
            # 'Leidos+retrieval_ft1',
            #'retrieval',
            'Leidos+retrieval_ft2'
            ]:
    series[target_method] = {frame: defaultdict(float) for frame in frame_name}
url2result = set()
with open('v1_append/merged_hybrid.jsonl') as fin:
    for line1 in fin:
        js = json.loads(line1)
        #valid_url.append(js['url'])
        y = dt.strptime(js['date'], "%Y-%m-%d")
        if  y >= dt.strptime(series_start_date, "%Y-%m-%d") and y<= dt.strptime(series_end_date, "%Y-%m-%d"):
        #print(line1)
        #break
            for target_method in ['Leidos','Westclass','Leidos+retrieval_ft2']:
                if f'{target_method}_prob' in js:
                    #js['west_prob'][frame_name[idx]] = p
                    for idx, f in enumerate(frame_name):
                        #if js['roberta_prob'][f] > 0.1:
                        #if frame_name[idx] == 'mistreatment':
                        #    print(y, js['phrased_title'])
                        #roberta_data[(y-a).days][f] += js['roberta_prob'][f]
                       # print(js['roberta_prob'][f])
                        series[target_method][f][js['date']] += js[f'{target_method}_prob'][f]
                        #print(west_series[f][y])
        url2result.add(js['url'])
for k in series:
    dump_series(series[k], f'v1_append/{k}_time_series_to_{series_end_date}.json')
        
url2event = defaultdict(list)
with open(sys.argv[1]) as IN:
    for line in IN:
        tmp = json.loads(line)
        if tmp['sourceurl'] in url2result:
            url2event[tmp['sourceurl']].append(tmp['EventCode'])
print(count)

IN=open(sys.argv[4],'r')
tmp = json.load(IN)

eventCodeMap = {}
for k in tmp.keys():
    eventCodeMap[k] = len(eventCodeMap)
print(eventCodeMap)

import pandas as pd
import calendar
from datetime import datetime
# Calculate the correlation matrix between gdelt event code and news frame
corr_mat = dict()
for target_method in [
            'Leidos',
            #'hybrid3',
            'Westclass',
            #'Westclass_top1000',
            # 'Leidos+retrieval_ft1',
            #'retrieval',
            'Leidos+retrieval_ft2'
            ]: # , 'lotclass', 'tuned_bert'
    selected_dates = [i.strftime("%Y-%m-%d") for i in pd.date_range(start=series_start_date,end=series_end_date).to_pydatetime().tolist()]
    selected_stamp = dict()
    date2stamp = dict()
    for d in selected_dates:
        stamp = str(int(calendar.timegm(datetime.strptime(d, "%Y-%m-%d").timetuple()))) + '000'
        selected_stamp[stamp] = d
        date2stamp[d] = stamp
    '''
    gdelt = defaultdict(dict)

    for ec, series in json.load(open('../data/timeseries/cp6_gdelt_timeseries.json')).items():
        series = json.loads(series)
        for i in selected_dates:
            gdelt[ec][i]  = series[date2stamp[i]]
    '''
    # west classification
    import numpy as np
    eventcode_doc = defaultdict(int)
    eventcode_nar_score = defaultdict(lambda:defaultdict(float))
    for fn in ['./v1_append/merged_hybrid.jsonl']:
        with open(f'{fn}') as fin:
            for line in fin:
                js = json.loads(line)
                if js['url'] in url2event and f'{target_method}_prob' in js: # js['relevance_prediction'] and 
                    for ec in url2event[js['url']]:
                        eventcode_doc[ec] += 1
                        for nar in frame_name:
                            if nar in js[f'{target_method}_prob']:
                                score = js[f'{target_method}_prob'][nar]
                                eventcode_nar_score[ec][nar] += score
    twitterGdeltMat_norm = {i:{j:0.0 for j in eventCodeMap} for i in frame_name}
    twitterGdeltMat_raw = {i:{j:0.0 for j in eventCodeMap} for i in frame_name}
    for n2 in eventCodeMap:
        for n1 in frame_name:
            twitterGdeltMat_norm[n1][n2] = eventcode_nar_score[n2][n1] / eventcode_doc[n2] if eventcode_doc[n2] > 0 else 0.0
            twitterGdeltMat_raw[n1][n2] = eventcode_nar_score[n2][n1]
    json.dump(twitterGdeltMat_norm, open(f'./v1_append/news_gdelt_{target_method}_corr_to_{series_end_date}.json', 'w'))
    corr_mat[target_method] = twitterGdeltMat_norm