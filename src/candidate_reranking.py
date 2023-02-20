import json
import sys


q_number= 0
qt_number = 0
qe_number = 0 

total = 0
dir = "../"+sys.argv[1]+"/scored_"+sys.argv[2]+".json"
dir_w = "../"+sys.argv[1]+"/pruned_"+sys.argv[2]+".json"

file = open(dir,'r')
file_w = open(dir_w,'w')
sum = 0
intersection = 0
total_number = 0
for line in file:
    jline = json.loads(line)
    total += 1
    answer_tag = ""
    triples = jline['triples']
    lists = []
    score_lists = []
    for triple in triples:
        if triple['answer'] and answer_tag== "":
            if triple['tag'] !="answer" and triple['tag'] !="answer_entity":
                answer_tag = triple['tag']
        lists.append((triple,triple['prune_score']))
        score_lists.append((triple,triple['prune_score']))
    
    lists.sort(key=lambda s:s[1], reverse=True)
    lists = lists[:5]
    score_lists.sort(key=lambda s:s[1], reverse=True)
    score_lists = score_lists[:5]
    keep_candidates = []
    for l in lists:
        keep_candidates.append(l[0])
    if lists[0][0]['object']==score_lists[0][0]['object'] and lists[0][0]['answer']:
        intersection +=1

    
    new_candidates = []
    has_answer = False
    for triple in keep_candidates:
        inserted=False
        objectt = triple['object'].lower()
        p_score = triple['prune_score']

        for t in new_candidates:
            if objectt ==t['object'].lower():
                inserted = True
                if p_score>t['prune_score']:
                    t=triple

        if inserted ==False: 
            new_candidates.append(triple)
        if triple['answer']:
            has_answer = True
            
    if has_answer :
        sum += len(new_candidates)
        total_number +=1
        jline['triples'] = new_candidates
        file_w.write(json.dumps(jline)+"\n")

        
    
    if answer_tag =="question":
        q_number += 1
    elif answer_tag == "question_tokens":
        qt_number +=1
    elif answer_tag == "question_entities":
        qe_number +=1

file.close()
