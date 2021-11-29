# This file contains the evaluation functions

import re
import numpy as np
import editdistance
import jieba

sentiment_word_list = ['positive', 'negative', 'neutral']
# senttag2word = {'正': 'positive', '其他': 'negative', '负': 'neutral'}
# sentiment_word_list_cn = ['正', '负','']
aspect_cate_list_cn1=['value', 'is_good', 'feeling1', 'breakfast', 'traffic', 'environment', 'is_has', 'is_convenient', 'facility_overview', 'afternoon_tea', 'parking', 'bedding', 'hotel_receptionist', 'children1', 'boss', 'come_again', 'subway_station', 'decorate', 'airport', 'restaurant', 'air_conditioner', 'facility_air_conditioning', '', 'beach', 'insect', 'balcony', 'air1', 'seafood', 'security', 'scenic', 'landlord', 'toilet_room', 'bathroom', 'toiletries_overview', 'slippers', 'elevator', 'bathtub', 'carpet', 'shower', 'corridor', 'parent_child_room', 'business_zone', 'lighting', 'farm_vegetables', 'projector', 'gymnasium', 'window', 'pedestrian_zone', 'dinner', 'style', 'heating', 'buffet', 'sofa', 'facility_electric', 'lunch', 'rice_noodles', 'swimming_pool', 'duty_free_store', 'railway_station', 'lamp', 'cash_pledge']
aspect_cate_list_cn=['is_good', 'feeling1', 'hygiene', 'environment', 'is_has', 'come_again', 'hot_spring', 'price', 'recommend1', 'traffic', 'is_convenient', 'facility_overview', 'breakfast', 'balcony', 'landlord', 'business1', 'value', 'air_conditioner', 'facility_air_conditioning', 'service_pickup', 'corridor', 'parking', 'elevator', 'boss', 'airport', 'guest_room', 'hotel_receptionist', 'scenic', 'attitude', 'bedding', 'hot_water', 'shower', 'toiletries_overview', 'toilet_room', 'bathroom', 'swimming_pool', 'children1', 'decorate', 'bed1', 'lighting', 'snack', 'buffet', 'center', 'type', 'boundless_pool', 'facility_washing_service', 'security', 'subway_station', 'dinner', 'skiing', 'toilet', 'gymnasium', 'insect', 'door', 'air1', 'lakescape', 'riverscape', 'style', 'afternoon_tea', 'television', 'traditional_garden', 'cash_pledge', 'pedestrian_zone', 'activity', 'sofa', 'bath_towel', 'family', 'seascape', 'projector', 'heating', 'slippers', 'restaurant', 'coffee', 'business_zone', 'bathtub', 'railway_station', 'window', 'wellness', 'seafood', 'children_playground', 'curtain', 'couple', 'beach', 'service_wake_up_call', 'sound', 'carpet', 'smell', 'lunch', 'floor', 'towel1', 'service_taxi_booking', 'sunset', 'lamp', 'exam', 'rice_noodles', 'chinese_food', 'parent_child_room', 'audio', 'facility_electric', 'service_room', 'food_format', 'computer_game', 'amusement_park', 'computer1', 'facility_kitchen', 'farm_vegetables', 'service_luggage', 'duty_free_store', 'european_food']
aspect_cate_list = ['location general',
 'food prices',
 'food quality',
 'ambience general',
 'service general',
 'restaurant prices',
 'drinks prices',
 'restaurant miscellaneous',
 'drinks quality',
 'drinks style_options',
 'restaurant general',
 'food style_options']


def extract_spans_extraction(task, seq):
    extractions = []
    if task == 'uabsa' and seq.lower() == 'none':
        return []
    else:
        if task in ['uabsa', 'aope']:
            all_pt = seq.split('; ')
            for pt in all_pt:
                pt = pt[1:-1]
                try:
                    a, b = pt.split(', ')
                except ValueError:
                    a, b = '', ''
                extractions.append((a, b))
        elif task in ['tasd', 'aste']:
            all_pt = seq.split('; ')
            for pt in all_pt:
                pt = pt[1:-1]
                try:
                    a, b, c = pt.split(', ')
                except ValueError:
                    a, b, c = '', '', ''
                extractions.append((a, b, c))      
        elif task in ['tasd-cn']:
            all_pt = seq.split('; ')
            for pt in all_pt:
                pt = pt[1:-1]
                try:
                    a,b,c,d= pt.split(', ')

#                     a=pt.split(', ')
#                     if len(a)==8:
#                         aa,b,c,d,e,f=[a[0],a[1],a[2],a[3],a[4]+', '+a[5],a[6]+', '+a[7]]
#                     else:
#                         a, b, c, d ,e , f='', '', '', '','',''
                        
                except ValueError:
                    a, b, c, d ,e , f= '', '', '', '','',''
                    break
                extractions.append((b, d)) # 只测俩
#                 extractions.append((b, d,e)) # 测仨
#                 extractions.append((a, b, c, d)) #全测
        elif task in ['tasd-cn2','tasd-cn2-xtc']:
            all_pt = seq.split('; ')
            for pt in all_pt:
                pt = pt[1:-1]
                try:
                    a, b = pt.split(', ')
                except ValueError:
                    a, b = '', ''
                extractions.append((a, b)) 
        return extractions


def extract_pairs(seq):
    aps = re.findall('\[.*?\]', seq)
    aps = [ap[1:-1] for ap in aps]
    pairs = []
    for ap in aps:
        # the original sentence might have 
        try:
            at, ots = ap.split('|')
        except ValueError:
            at, ots  = '', ''
        
        if ',' in ots:     # multiple ots 
            for ot in ots.split(', '):
                pairs.append((at, ot))
        else:
            pairs.append((at, ots))    
    return pairs        


def extract_triplets(seq):
    aps = re.findall('\[.*?\]', seq)
    aps = [ap[1:-1] for ap in aps]
    triplets = []
    for ap in aps:
        try:
            a, b, c = ap.split('|')
        except ValueError:
            a, b, c = '', '', ''
        
        # for ASTE
        if b in sentiment_word_list:
            if ',' in c:
                for op in c.split(', '):
                    triplets.append((a, b, op))
            else:
                triplets.append((a, b, c))
        # for TASD
        else:
            if ',' in b:
                for ac in b.split(', '):
                    triplets.append((a, ac, c))
            else:
                triplets.append((a, b, c))

    return triplets

def extract_spans_annotation(task, seq):
    if task in ['aste', 'tasd','tasd-cn']:
        extracted_spans = extract_triplets(seq)
    elif task in ['aope', 'uabsa']:
        extracted_spans = extract_pairs(seq)

    return extracted_spans

def recover_terms_with_editdistance(original_term, sent):
    words = original_term.split(' ')
    new_words = []
#     print('opinion:',words,'sent:',sent)
    for word in words:
        edit_dis = []
        for token in sent:
            edit_dis.append(editdistance.eval(word, token))
        smallest_idx = edit_dis.index(min(edit_dis))
        new_words.append(sent[smallest_idx])
    new_term = ' '.join(new_words)
    return new_term


def fix_preds_uabsa(all_pairs, sents):

    all_new_pairs = []
    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                # AT not in the original sentence
                if pair[0] not in  ' '.join(sents[i]):
                    # print('Issue')
                    new_at = recover_terms_with_editdistance(pair[0], sents[i])
                else:
                    new_at = pair[0]

                if pair[1] not in sentiment_word_list:
                    new_sentiment = recover_terms_with_editdistance(pair[1], sentiment_word_list)
                else:
                    new_sentiment = pair[1]

                new_pairs.append((new_at, new_sentiment))
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)

    return all_new_pairs


def fix_preds_aope(all_pairs, sents):

    all_new_pairs = []

    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                #print(pair)
                # AT not in the original sentence
                if pair[0] not in  ' '.join(sents[i]):
                    # print('Issue')
                    new_at = recover_terms_with_editdistance(pair[0], sents[i])
                else:
                    new_at = pair[0]

                # OT not in the original sentence
                ots = pair[1].split(', ')
                new_ot_list = []
                for ot in ots:
                    if ot not in ' '.join(sents[i]):
                        # print('Issue')
                        new_ot_list.append(recover_terms_with_editdistance(ot, sents[i]))
                    else:
                        new_ot_list.append(ot)
                new_ot = ', '.join(new_ot_list)

                new_pairs.append((new_at, new_ot))
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)

    return all_new_pairs


# for ASTE
def fix_preds_aste(all_pairs, sents):

    all_new_pairs = []

    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                #two formats have different orders
                p0, p1, p2 = pair
                # for annotation-type
                if p1 in sentiment_word_list:
                    at, ott, ac = p0, p2, p1
                    io_format = 'annotation'
                # for extraction type
                elif p2 in sentiment_word_list:
                    at, ott, ac = p0, p1, p2
                    io_format = 'extraction'

                #print(pair)
                # AT not in the original sentence
                if at not in  ' '.join(sents[i]):
                    # print('Issue')
                    new_at = recover_terms_with_editdistance(at, sents[i])
                else:
                    new_at = at
                
                if ac not in sentiment_word_list:
                    new_sentiment = recover_terms_with_editdistance(ac, sentiment_word_list)
                else:
                    new_sentiment = ac
                
                # OT not in the original sentence
                ots = ott.split(', ')
                new_ot_list = []
                for ot in ots:
                    if ot not in ' '.join(sents[i]):
                        # print('Issue')
                        new_ot_list.append(recover_terms_with_editdistance(ot, sents[i]))
                    else:
                        new_ot_list.append(ot)
                new_ot = ', '.join(new_ot_list)
                if io_format == 'extraction':
                    new_pairs.append((new_at, new_ot, new_sentiment))
                else:
                    new_pairs.append((new_at, new_sentiment, new_ot))
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)
    
    return all_new_pairs


def fix_preds_tasd(all_pairs, sents):

    all_new_pairs = []

    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                #print(pair)
                # AT not in the original sentence
                sents_and_null = ' '.join(sents[i]) + 'NULL'
                if pair[0] not in  sents_and_null:
                    # print('Issue')
                    new_at = recover_terms_with_editdistance(pair[0], sents[i])
                else:
                    new_at = pair[0]
                
                # AC not in the list
                acs = pair[1].split(', ')
                new_ac_list = []
                for ac in acs:
                    if ac not in aspect_cate_list:
                        new_ac_list.append(recover_terms_with_editdistance(ac, aspect_cate_list))
                    else:
                        new_ac_list.append(ac)
                new_ac = ', '.join(new_ac_list)
                
                if pair[2] not in sentiment_word_list:
                    new_sentiment = recover_terms_with_editdistance(pair[2], sentiment_word_list)
                else:
                    new_sentiment = pair[2]
            
                new_pairs.append((new_at, new_ac, new_sentiment))
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)
    
    return all_new_pairs

def fix_preds_tasd_cn(all_pairs, sents):
    
    all_new_pairs = []
    if len(all_pairs[0][0])==4:
        for i, pairs in enumerate(all_pairs):
            sents[i]=[''.join(sents[i])]
            new_pairs = []
            if pairs == []:
                all_new_pairs.append(pairs)
            else:
                for pair in pairs:
                    # print(pair)
                    # AT not in the original sentence
                    sents_and_null = ' '.join(sents[i]) + 'NULL'
                    if pair[0] not in  sents_and_null:
                        # print(pair[0])
                        # print(sents[i])
                        # print('Issue')
                        seg_list=list(jieba.cut(sents[i][0],cut_all=True))
                        new_at = recover_terms_with_editdistance(pair[0], seg_list)
                    else:
                        new_at = pair[0]
                
                    # AC not in the list
                    acs = pair[1].split(', ')
                    new_ac_list = []
                    for ac in acs:
                        if ac not in aspect_cate_list_cn:
                            new_ac_list.append(recover_terms_with_editdistance(ac, aspect_cate_list_cn))
                        else:
                            new_ac_list.append(ac)
                    new_ac = ', '.join(new_ac_list)
                
                    ots = pair[2].split(', ')
                    new_ot_list = []
                    for ot in ots:
                        if ot not in ' '.join(sents[i]):
#                         print('Issue')
                            # print(sents[i])
                            seg_list=list(jieba.cut(sents[i][0],cut_all=True))
                            new_ot_list.append(recover_terms_with_editdistance(ot, seg_list))
                        else:
                            new_ot_list.append(ot)
                    new_ot = ', '.join(new_ot_list)
                
                    if pair[3] not in sentiment_word_list_cn:
                        new_sentiment = recover_terms_with_editdistance(pair[3], sentiment_word_list_cn)
                    else:
                        new_sentiment = pair[3]
            
                    new_pairs.append((new_at, new_ac,new_ot, new_sentiment))#全测
                    # print(pair, '>>>>>', word_and_sentiment)
                    # print(all_target_pairs[i])
                all_new_pairs.append(new_pairs)
    elif len(all_pairs[0][0])==2:
        for i, pairs in enumerate(all_pairs):
            sents[i]=[''.join(sents[i])]
            new_pairs = []
            if pairs == []:
                all_new_pairs.append(pairs)
            else:
                for pair in pairs:
                    # AC not in the list
                    acs = pair[0].split(', ')
                    new_ac_list = []
                    for ac in acs:
                        if ac not in aspect_cate_list_cn:
                            new_ac_list.append(recover_terms_with_editdistance(ac, aspect_cate_list_cn))
                        else:
                            new_ac_list.append(ac)
                    new_ac = ', '.join(new_ac_list)

                    if pair[1] not in sentiment_word_list_cn:
                        new_sentiment = recover_terms_with_editdistance(pair[1], sentiment_word_list_cn)
                    else:
                        new_sentiment = pair[1]
            
                    new_pairs.append((new_ac, new_sentiment))
                all_new_pairs.append(new_pairs)
    
    return all_new_pairs

def fix_preds_tasd_cn2(all_pairs, sents):
    
    all_new_pairs = []

    for i, pairs in enumerate(all_pairs):
        sents[i]=[''.join(sents[i])]
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                # AC not in the list
                acs = pair[0].split(', ')
                new_ac_list = []
                for ac in acs:
                    if ac not in aspect_cate_list_cn:
                        new_ac_list.append(recover_terms_with_editdistance(ac, aspect_cate_list_cn))
                    else:
                        new_ac_list.append(ac)
                new_ac = ', '.join(new_ac_list)

                if pair[1] not in sentiment_word_list_cn:
                    new_sentiment = recover_terms_with_editdistance(pair[1], sentiment_word_list_cn)
                else:
                    new_sentiment = pair[1]
            
                new_pairs.append((new_ac, new_sentiment))
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)
    
    return all_new_pairs

def fix_pred_with_editdistance(all_predictions, sents, task):
    if task == 'uabsa':
        fixed_preds = fix_preds_uabsa(all_predictions, sents)
    elif task == 'aope':
        fixed_preds = fix_preds_aope(all_predictions, sents) 
    elif task == 'aste': 
        fixed_preds = fix_preds_aste(all_predictions, sents) 
    elif task == 'tasd':
        fixed_preds = fix_preds_tasd(all_predictions, sents) 
    elif task == 'tasd-cn':
        fixed_preds = fix_preds_tasd_cn(all_predictions, sents) 
    elif task == 'tasd-cn2' or task == 'tasd-cn2-xtc':
        fixed_preds = fix_preds_tasd_cn2(all_predictions, sents) 
    else:
        print("*** Unimplemented Error ***")
        fixed_preds = all_predictions

    return fixed_preds


def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold pairs/triplets
    The input needs to be already processed
    """
    # number of true postive, gold standard, predicted aspect terms
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        gold_pt[i]=list(set(gold_pt[i]))
        pred_pt[i]=list(set(pred_pt[i]))
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores

def idx_match(idx1,idx2):
    if idx1[0]>idx2[0]:
        idx1,idx2=idx2,idx1
    if idx1[0]<=idx2[0] and idx1[1]-idx2[0]>=0:
        return True
    else:
        return False
def compute_f1_scores_idx(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold pairs/triplets
    The input needs to be already processed
    """
    # number of true postive, gold standard, predicted aspect terms
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        gold_pt[i]=list(set(gold_pt[i]))
        pred_pt[i]=list(set(pred_pt[i]))
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            for j in gold_pt[i]:
                if t[1]==j[1] and t[0]==j[0] and idx_match(t[2],j[2]):
                    n_tp += 1

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores

def compute_scores(pred_seqs, gold_seqs, sents, io_format, task, hasidx=False):
    """
    compute metrics for multiple tasks
    """
    assert len(pred_seqs) == len(gold_seqs) 
    num_samples = len(gold_seqs)

    all_labels, all_predictions = [], []

    for i in range(num_samples):
        if io_format == 'annotation':
            gold_list = extract_spans_annotation(task, gold_seqs[i])
            pred_list = extract_spans_annotation(task, pred_seqs[i])
        elif io_format == 'extraction':
            gold_list = extract_spans_extraction(task, gold_seqs[i])
            pred_list = extract_spans_extraction(task, pred_seqs[i])
#             print(gold_list)
        all_labels.append(gold_list)
        all_predictions.append(pred_list)
    print("\nResults of raw output")
    labels=np.array(all_labels)
    np.save('./labels.npy',labels)
    predictions=np.array(all_predictions)
    np.save('./predictions_nofix.npy',predictions)
    if hasidx:
        raw_scores = compute_f1_scores_idx(all_predictions, all_labels)
    else:
        raw_scores = compute_f1_scores(all_predictions, all_labels)
    print(raw_scores)
    try:
        all_predictions_fixed = fix_pred_with_editdistance(all_predictions, sents, task)
        np.save('./predictions_fix.npy',all_predictions_fixed)
        print("\nResults of fixed output")
        fixed_scores = compute_f1_scores(all_predictions_fixed, all_labels)
        print(fixed_scores)
    except:
        all_predictions_fixed=[]
        fixed_scores=''
    return raw_scores, fixed_scores, all_labels, all_predictions, all_predictions_fixed