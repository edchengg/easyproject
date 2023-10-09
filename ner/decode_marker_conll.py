import pickle
import os
import copy
from process_conll import InputExample, save_pickle
from difflib import SequenceMatcher
import time
from xml.sax.saxutils import unescape
import jieba
from sacremoses import MosesTokenizer
import tqdm
import json

# A tokenizer for tokenizing Chinese text
# Author: Chao Jinag at June 20, 2021
import re
import itertools
from nltk import word_tokenize

FULL2HALF = dict((i + 0xFEE0, i) for i in range(0x30, 0x40))


# FULL2HALF = dict((i + 0xFEE0, i) for i in range(0x21, 0x7F))
# FULL2HALF[0x3000] = 0x20
# FULL2HALF = {'１': '1', '２': '2', '３': '3', '４': '4', '５': '5', '６': '6', '７': '7', '８': '8', '９': '9', '０': '0'}
def halfen(s):
    '''
    Convert full-width characters to ASCII counterpart.
    halfen('１２３４５６７８９０') == '1234567890'
    '''
    return str(s).translate(FULL2HALF)


# HALF2FULL = dict((i, i + 0xFEE0) for i in range(0x21, 0x7F))
# HALF2FULL[0x20] = 0x3000
# def fullen(s):
#     '''
#     Convert all ASCII characters to the full-width counterpart.
#     '''
#     return str(s).translate(HALF2FULL)
def judge_if_Chinese_char(sent):
    '''
    Judge each char in a sent if Chinese
    Input:
        a sentence. "由于这种情绪普遍存在，投资者自然会寻求“防御性”投资。"
    Output:
        a 0, 1 list, 1 means is Chinese Char, 0 means not.
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0]
    '''

    return [1 if re.match(r'[\u4e00-\u9fff]+', i) != None else 0 for i in sent]

def judge_if_Chinese_and_Japanese_char(sent):
    '''
    Judge each char in a sent if Chinese or Japanese
    Input:
        a sentence. "由于这种情绪普遍存在，投资者自然会寻求“防御性”投资。"
    Output:
        a 0, 1 list, 1 means is Chinese Char, 0 means not.
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0]
    '''
    'U+0E00..U+0E7F'
    return [1 if re.match(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f\u0E00-\u0E7Fa]+', i) != None else 0 for i in sent]

def merge_adjacent_identical_numbers(number_list):
    '''
    Merge adjacent identical elements in to a sublist.
    Input:
        a number list.
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0]
    Output:
        grouped list: [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0], [1, 1, 1, 1, 1, 1, 1, 1], [0], [1, 1, 1], [0], [1, 1], [0]]
        a list of 3-element tuples, (start_idx, end_idx, duplicated_element).
        [[0, 10, 1], [10, 11, 0], [11, 19, 1], [19, 20, 0], [20, 23, 1], [23, 24, 0], [24, 26, 1], [26, 27, 0]]

    '''
    U = []
    key_func = lambda x: x
    for key, group in itertools.groupby(number_list, key_func):
        U.append(list(group))

    begin_end_idx_list = []
    count = 0
    for i in U:
        start_idx = count
        for j in i:
            count += 1
        end_idx = count
        begin_end_idx_list.append([start_idx, end_idx, i[0]])

    return U, begin_end_idx_list



def Chinese_and_Japanese_tokenizer(sent):
    '''
    Yang Chen - edit based on Chao Jiang's Chinese tokenizer - including Japanese

    :param sent:
    :return:
    '''
    # sent = "由于这种情绪普遍存在，投资者自然会寻求“防御性”投资。"
    output = []

    # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0]
    Chinese_or_Eng_list = judge_if_Chinese_and_Japanese_char(sent)

    # group = [[0, 10, 1], [10, 11, 0], [11, 19, 1], [19, 20, 0], [20, 23, 1], [23, 24, 0], [24, 26, 1], [26, 27, 0]]
    _, group = merge_adjacent_identical_numbers(Chinese_or_Eng_list)

    for (start_idx, end_idx, Chinese_or_not) in group:
        # if English, use word_tokenize
        if Chinese_or_not == 0:
            char_list = sent[start_idx: end_idx]
            tokenized_list = word_tokenize(char_list)
            change_dict = {'``': '"', "''": '"', }
            # need to convert full-width number in CoNLL data to regular half-width number
            tokenized_list = [halfen(i) if i not in change_dict else halfen(change_dict[i]) for i in tokenized_list]
            output.extend(tokenized_list)
        # if Chinese, do char-tokenization
        elif Chinese_or_not == 1:
            # need to convert full-width number in CoNLL data to regular half-width number
            output.extend([halfen(c) for c in sent[start_idx: end_idx]])

    return output, " ".join(output)

def Chinese_tokenizer(sent):
    # sent = "由于这种情绪普遍存在，投资者自然会寻求“防御性”投资。"
    output = []

    # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0]
    Chinese_or_Eng_list = judge_if_Chinese_char(sent)

    # group = [[0, 10, 1], [10, 11, 0], [11, 19, 1], [19, 20, 0], [20, 23, 1], [23, 24, 0], [24, 26, 1], [26, 27, 0]]
    _, group = merge_adjacent_identical_numbers(Chinese_or_Eng_list)

    for (start_idx, end_idx, Chinese_or_not) in group:
        # if English, use word_tokenize
        if Chinese_or_not == 0:
            char_list = sent[start_idx: end_idx]
            tokenized_list = word_tokenize(char_list)
            change_dict = {'``': '"', "''": '"', }
            # need to convert full-width number in CoNLL data to regular half-width number
            tokenized_list = [halfen(i) if i not in change_dict else halfen(change_dict[i]) for i in tokenized_list]
            output.extend(tokenized_list)
        # if Chinese, do char-tokenization
        elif Chinese_or_not == 1:
            # need to convert full-width number in CoNLL data to regular half-width number
            output.extend([halfen(c) for c in sent[start_idx: end_idx]])

    return output, " ".join(output)

rule1 = {'[[': '[', ']]': ']', '[ [': '[', '] ]': ']'}
rule2 = {'《': '[', '》': ']'}
rule3 = {'【': '[', '】': ']'}
#rule4 = {'。 ]': ''}
marker_tags = ['[', ']']
post_processing_rules = [rule1, rule2, rule3]
start_marker_tags = ['[']
marker_map = {'[': ']'}

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def load_pickle(file_name):
    # load saved results from model
    assert os.path.isfile(file_name), f"file does not exist: {file_name}"
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    print(f"Loaded data from {file_name}")
    return data

html_tags = ['<LOC>', '</LOC>', '<ORG>', '</ORG>', '<PER>', '</PER>', '<MISC>', '</MISC>']
start_html_tags = ['<LOC>', '<ORG>', '<PER>', '<MISC>']
html_map = {'<LOC>': '</LOC>', '<ORG>': '</ORG>',  '<PER>': '</PER>', '<MISC>': '</MISC>'}

def decode_label_span(label):
    label_tags = label
    span_labels = []
    last = 'O'
    start = -1
    for i, tag in enumerate(label_tags):
        pos, _ = (None, 'O') if tag == 'O' else tag.split('-')
        if (pos == 'B' or tag == 'O') and last != 'O':  # end of span
            span_labels.append((start, i, last.split('-')[1]))
        if pos == 'B' or last == 'O':  # start of span or move on
            start = i
        last = tag
    #print(label_tags)
    if label_tags[-1] != 'O':
        span_labels.append((start, len(label_tags), label_tags[-1].split('-')[1]))

    return span_labels

def label_number_calculation(label_span):
    #LOC, ORG, PER
    num_loc = 0
    num_per = 0
    num_org = 0
    for span in label_span:
        if span[-1] == 'LOC':
            num_loc += 1
        elif span[-1] == 'ORG':
            num_org += 1
        elif span[-1] == 'PER':
            num_per += 1
    return [num_loc, num_org, num_per]

def xml_decode(example, lang):
    trans_sent = example.xml_trans[lang]
    org_encode_sent = example.xml_encode_sent
    trans_sent = unescape(trans_sent, {"&apos;": "'", "&quot;": '"', "&#39;": "'"})
    LABEL = ['<LOC>', '</LOC>', '<ORG>', '</ORG>', '<PER>', '</PER>', '<MISC>', '</MISC>']
    for tag in LABEL:
        if tag in trans_sent:
            trans_sent = trans_sent.replace(tag, ' {} '.format(tag))

    unequal = False
    for tag in LABEL:
        if trans_sent.count(tag) != org_encode_sent.count(tag):
            # unmatched
            unequal = True


    sentence = trans_sent.split()

    new_sentence = []
    labels = []
    label_start = False
    for c in sentence:
        if c in LABEL:
            lab = c.strip('<>')
            if '/' not in c:
                label_start = True
                label_continue = False
            elif '/' in c:
                label_start = False
        else:
            # tokenize c
            if lang == 'zh':
                #use jieba
                tmp = list(jieba.cut(c))
                tokens = [i for i in tmp if i.strip() != ""]
            else:
                #use mose
                tokens = mt.tokenize(c, escape=False)
                for idx, t in enumerate(tokens):
                    t = unescape(t, {"&apos;": "'", "&quot;": '"', "&#39;": "'"})
                    tokens[idx] = t

            for idx, cc in enumerate(tokens):
                if label_start == True:
                    if idx == 0 and label_continue == False:
                        labels.append('B-' + lab)
                        label_continue = True
                    else:
                        labels.append('I-' + lab)
                else:
                    labels.append('O')
                new_sentence.append(cc)

    src_label_span = example.span_labels
    src_label_number = label_number_calculation(src_label_span)
    if labels == []:
        return None, None, True, [src_label_number, [0, 0, 0, 0]]

    # final validation
    tgt_label_span = decode_label_span(labels)

    if (unequal == False) and (len(tgt_label_span) != len(src_label_span)):
        #print(trans_sent)
        unequal = True

    # calculate projection rate
    tgt_label_number = label_number_calculation(tgt_label_span)
    projection_stat = [src_label_number, tgt_label_number]
    # projection_rate =  [[],[]]
    if src_label_number != tgt_label_number:
        unequal = True
    return new_sentence, labels, unequal, projection_stat


def write_into_output(output, sentence, labels):
    for word, lab in zip(sentence, labels):
        output.write('{}\t{}\n'.format(word, lab))

    output.write('\n')


def calculate_proj_rate(src_label_stats, tgt_label_stats):
    total_src_label = sum(sum(p) for p in src_label_stats)
    total_tgt_label = sum(sum(p) for p in tgt_label_stats)
    proj_rate = total_tgt_label / total_src_label

    granular_proj_rate = []
    granular_src_idx_label = []
    granular_tgt_idx_label = []
    granular_proj_rate = []
    for idx in range(len(src_label_stats[0])):
        total_src_idx_label = sum(p[idx] for p in src_label_stats)
        total_tgt_idx_label = sum(p[idx] for p in tgt_label_stats)

        granular_src_idx_label.append(total_src_idx_label)
        granular_tgt_idx_label.append(total_tgt_idx_label)

        proj_rate_idx = total_tgt_idx_label / total_src_idx_label
        granular_proj_rate.append(proj_rate_idx)

    summary = {'proj_rate': proj_rate,
               'src_label_num': total_src_label,
               'tgt_label_num': total_tgt_label,
               'loc': granular_proj_rate[0],
               'src_loc_num': granular_src_idx_label[0],
               'tgt_loc_num': granular_tgt_idx_label[0],
               'org': granular_proj_rate[1],
               'src_org_num': granular_src_idx_label[1],
               'tgt_org_num': granular_tgt_idx_label[1],
               'per': granular_proj_rate[2],
               'src_per_num': granular_src_idx_label[2],
               'tgt_per_num': granular_tgt_idx_label[2],
               }

    return summary


def extract_XML_entity(sentence, html_tags):
    entity_list = []
    entity_tag_list = []

    for tag in html_tags:
        sentence = sentence.replace(tag, ' {} '.format(tag))

    toks = sentence.split()
    end_tag = None
    #print(toks)
    for idx, tok in enumerate(toks):
        if tok in start_html_tags:
            start = idx
            end_tag = html_map[tok]
        elif tok == end_tag:
            entity = toks[start+1: idx]
            entity_list.append(entity)
            entity_tag_list.append(end_tag.strip('</>'))

            end_tag = None
        else:
            pass

    return entity_list, entity_tag_list

def marker_decode(example, lang, tag_list):
    trans_sent = example.marker_trans[lang]
    trans_sent = post_process(trans_sent, post_processing_rules)
    trans_sent = unescape(trans_sent, {"&apos;": "'", "&quot;": '"', "&#39;": "'"})
    org_encode_sent = example.xml_encode_sent

    for tag in marker_tags:
        if tag in trans_sent:
            trans_sent = trans_sent.replace(tag, ' {} '.format(tag))

    sentence = trans_sent.split()

    new_sentence = []
    labels = []
    label_start = False
    lab_idx = 0

    for c in sentence:
        if c in marker_tags:
            lab = tag_list[lab_idx]
            if c == '[':
                label_start = True
                label_continue = False
            elif c == ']':
                label_start = False
                lab_idx += 1
        else:

            # tokenize c
            if lang == 'zh':
                # use jieba
                tmp = list(jieba.cut(c))
                tokens = [i for i in tmp if i.strip() != ""]
            elif lang == 'ja' or lang == 'th':
                tokens = Chinese_and_Japanese_tokenizer(c)[0]
            else:
                # use mose
                tokens = mt.tokenize(c, escape=False)
                for idx, t in enumerate(tokens):
                    t = unescape(t, {"&apos;": "'", "&quot;": '"', "&#39;": "'"})
                    tokens[idx] = t

            for idx, cc in enumerate(tokens):
                if label_start == True:
                    if idx == 0 and label_continue == False:
                        labels.append('B-' + lab)
                        label_continue = True
                    else:
                        labels.append('I-' + lab)
                else:
                    labels.append('O')
                new_sentence.append(cc)

    src_label_span = example.span_labels
    src_label_number = label_number_calculation(src_label_span)
    if labels == []:
        return None, None, True, [src_label_number, [0, 0, 0, 0]]

    # final validation
    tgt_label_span = decode_label_span(labels)

    if (len(tgt_label_span) != len(src_label_span)):
        # print(trans_sent)
        unequal = True
    else:
        unequal = False
    # calculate projection rate
    tgt_label_number = label_number_calculation(tgt_label_span)
    projection_stat = [src_label_number, tgt_label_number]
    # projection_rate =  [[],[]]
    if src_label_number != tgt_label_number:
        unequal = True

    return new_sentence, labels, unequal, projection_stat


def post_process(line, post_processing_rules):

    for rule in post_processing_rules:
        for key, val in rule.items():
            line = line.replace(key, val)

    return line


def extract_marker_entity(sentence, marker_tags):
    entity_list = []
    for tag in marker_tags:
        sentence = sentence.replace(tag, ' {} '.format(tag))
    toks = sentence.split()
    start_marker_tag = '['
    end_tag = ']'
    start = None
    for idx, tok in enumerate(toks):
        if tok == start_marker_tag:
            start = idx
        elif tok == end_tag and start != None:

            entity = toks[start + 1: idx]

            entity_list.append(entity)
            start = None
        else:
            pass
    return entity_list

def check_marker_correct(marker, en_entity_list):
    boolean = True
    for tag in marker_tags:
        if marker.count(tag) != len(en_entity_list):
            boolean = False
            break
    return boolean

def write_into_output(output, sentence, labels):
    for word, lab in zip(sentence, labels):
        output.write('{} {}\n'.format(word, lab))

    output.write('\n')

def F1_score(tags,predicted):
    tags = set(tags)
    predicted = set(predicted)

    tp = len(tags & predicted)
    fp = len(predicted) - tp
    fn = len(tags) - tp

    if tp>0:
        precision=float(tp)/(tp+fp)
        recall=float(tp)/(tp+fn)
        f1 = 2*((precision*recall)/(precision+recall))
        return precision, recall, f1
    else:
        return 0, 0, 0

def convert_stat_tags(gold):
    tag = []
    # convert number of entity in a sentence into tuples
    # sent_id,entity_type,entity_id
    for idx, sent in enumerate(gold):
        for idxx, t in enumerate(sent):
              for n in range(t):
                    tag.append('{}-{}-{}'.format(idx, idxx, n))
    return tag

def calculate_proj_f1(src_label_stats, tgt_label_stats):
    gold = convert_stat_tags(src_label_stats)
    proj = convert_stat_tags(tgt_label_stats)

    proj_precision, proj_recall, proj_f1 = F1_score(gold, proj)
    summary = {'proj_f1': proj_f1,
               'proj_precision': proj_precision,
               'proj_recall': proj_recall,
               'src_label_stats': src_label_stats,
               'tgt_label_stats': tgt_label_stats
               }

    return summary

if __name__ == "__main__":
    
    mt_system = "nllb_3Bft"
    if not os.path.exists("output_{}_conll".format(mt_system)):
        os.mkdir("output_{}_conll".format(mt_system))

    for src_lang in ["en"]:
        en_examples = load_pickle("conll_nllb_3B_ft.pkl")
        for THRESHOLD in [0.5]:
            for tgt_lang in ["bam_Latn", "ewe_Latn", "fon_Latn", "hau_Latn", "ibo_Latn", "kin_Latn", "lug_Latn", "luo_Latn", 
    "mos_Latn", "nya_Latn", "sna_Latn", "swh_Latn", "tsn_Latn", "twi_Latn", "wol_Latn", "xho_Latn", "yor_Latn", "zul_Latn"]:
                tgt_lang = tgt_lang.split("_")[0]
                wrong_number = 0
                total_number = 0
                num_xml = 0
                num_marker = 0
                num_en_match = 0
                num_xml_match = 0
                num_trans_match = 0
                out = open('output_{}_conll/{}-{}-{}-marker.txt'.format(mt_system, mt_system, src_lang, tgt_lang), 'w')

                mt = MosesTokenizer(lang=tgt_lang)

                src_label_stats = []
                tgt_label_stats = []
                num_correct = 0
                num_multi_span = 0
                for idx in tqdm.tqdm(range(len(en_examples))):
                    xml = en_examples[idx].xml_trans[tgt_lang]
                    marker_pre = en_examples[idx].marker_trans[tgt_lang]
                    marker = post_process(marker_pre, post_processing_rules)
                    trans_entity_list = en_examples[idx].ent_trans[tgt_lang]
                    trans_entity_list = [i.lower() for i in trans_entity_list]
                    en_entity_list = en_examples[idx].entity_list
                    tag_list = en_examples[idx].tag_list
                    xml_entity, xml_ent_tags = extract_XML_entity(xml, html_tags)
                    marker_entity = extract_marker_entity(marker, marker_tags)
                    correct_marker_num = check_marker_correct(marker, en_entity_list)
                    use_xml = False
                    en_entity_list = [' '.join(tmp).lower() for tmp in en_entity_list]
                    xml_entity = [' '.join(tmp).lower() for tmp in xml_entity]
                    marker_entity = [' '.join(tmp).lower() for tmp in marker_entity]
                    # multi span 31% 4363
                    # check number entity is correct
                    wrong_flag = False
                    if correct_marker_num and len(marker_entity) == len(en_entity_list):
                        num_correct += 1               
                        # check number of entity == 1? if == 1: no need to align
                        if len(en_entity_list) == 1:
                            marker_entities_tag = tag_list
                        else:
                            if len(set(tag_list)) == 1:
                                # check all entity tag is same type? if same type: no need to align
                                marker_entities_tag = tag_list
                            else:
                                # for each ent in marker: find it in trans,en entity list: hard string match
                                marker_entities_tag = []
                                for ent in marker_entity:
                                    # check if its in xml_entity
                                    if ent in trans_entity_list:
                                        iidx = trans_entity_list.index(ent)
                                        marker_entities_tag.append(tag_list[iidx])
                                        num_trans_match += 1
                                    elif ent in en_entity_list:
                                        iidx = en_entity_list.index(ent)
                                        marker_entities_tag.append(tag_list[iidx])
                                        num_en_match += 1
                                    else:
                                        found = False
                                        # if not found:
                                        # use string similarity to find it
                                        if not found:
                                            for iddx, ent_ind in enumerate(trans_entity_list):
                                                if similar(ent_ind, ent) > THRESHOLD:
                                                    marker_entities_tag.append(tag_list[iddx])
                                                    found = True
                                                    break

                                        if not found:
                                            for iddx, ent_en in enumerate(en_entity_list):
                                                if similar(ent_en, ent) > THRESHOLD:
                                                    marker_entities_tag.append(tag_list[iddx])
                                                    found = True
                                                    break

                                        if not found:
                                            # if still not found, add 'X'
                                            marker_entities_tag.append('X')

                                if 'X' in marker_entities_tag:
                                    # check number of X
                                    if marker_entities_tag.count('X') == 1:
                                        # if only has one 'X' --> we can find out the tag
                                        # else: failed to align --> replace with XML
                                        # find the rest tag
                                        iddx = marker_entities_tag.index('X')
                                        copy_trans_entities_tag = copy.deepcopy(tag_list)
                                        try:
                                            for x in marker_entities_tag:
                                                if x != 'X':
                                                    copy_trans_entities_tag.remove(x)

                                            marker_entities_tag[iddx] = copy_trans_entities_tag[0]
                                        except:
                                            wrong_flag = True
                                    else:
                                        wrong_flag = True
                    else:
                        wrong_flag = True

                    if wrong_flag == True:
                        continue

                    if not sorted(marker_entities_tag) == sorted(tag_list):
                        wrong_number += 1
                    else:
                        new_sentence, labels, unequal, projection_stats = marker_decode(en_examples[idx], tgt_lang, marker_entities_tag)

                        if not unequal:
                            src_label_stats.append(projection_stats[0])
                            tgt_label_stats.append(projection_stats[1])
                            total_number += 1
                            num_marker += 1
                            write_into_output(out, new_sentence, labels)

                summary = calculate_proj_f1(src_label_stats, tgt_label_stats)

                stats = {'src_lang': src_lang,
                        'tgt_lang': tgt_lang,
                         'num_xml_sentence': num_xml,
                         'num_marker_sentence': num_marker,
                         'num_total_sentence': total_number,
                         }
                #print(stats)
                print(num_correct)
                final_stats = {**summary, **stats}
                
                final_stats["sentence_percentage"] = stats["num_total_sentence"]/14041
                print(stats["num_total_sentence"]/14041)
                with open("output_{}_conll/{}-{}-{}-marker-stats.json".format(mt_system, mt_system, src_lang, tgt_lang), "w") as f:
                    json.dump(final_stats, f)