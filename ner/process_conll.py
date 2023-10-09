import os
import logging
import copy
import pickle
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, langs=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          words: list. The words of the sequence.
          labels: (Optional) list. The labels for each word of the sequence. This should be
          specified for train and dev examples, but not for test examples.
        """
        tag2mark_xml = {
            'ORG': {'s': '<ORG>', 'e': '</ORG>'},
            'PER': {'s': '<PER>', 'e': '</PER>'},
            'LOC': {'s': '<LOC>', 'e': '</LOC>'},
            'MISC': {'s': '<MISC>', 'e': '</MISC>'},
        }

        tag2mark_marker = {
            'ORG': {'s': '[', 'e': ']'},
            'PER': {'s': '[', 'e': ']'},
            'LOC': {'s': '[', 'e': ']'},
            'MISC': {'s': '[', 'e': ']'}
        }

        self.guid = guid
        self.words = words
        self.labels = labels
        self.langs = langs
        self.entity_list, self.tag_list, self.span_labels = self.extract_entity(self.words, self.labels)
        self.xml_encode_sent = self.encode(self.words, self.labels, tag2mark_xml)
        self.marker_encode_sent = self.encode(self.words, self.labels, tag2mark_marker)
        self.org_trans = {}
        self.xml_trans = {}
        self.marker_trans = {}
        self.ent_trans = {}


    def decode_label_span(self, label):
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
        if label_tags[-1] != 'O':
            span_labels.append((start, len(label_tags), label_tags[-1].split('-')[1]))

        return span_labels

    def extract_entity(self, word, label):

        span_labels = self.decode_label_span(label)
        entity_list = []
        tag_list = []
        for span in span_labels:
            s, e, tag = span
            entity = word[s: e]
            entity_list.append(entity)
            tag_list.append(tag)

        return entity_list, tag_list, span_labels

    def encode(self, tokens, label, tag2mark):
        copy_tokens = copy.deepcopy(tokens)
        prev_start_lab = 'O'
        for idx, lab in enumerate(label):
            if lab != 'O' and (prev_start_lab == 'I' or prev_start_lab == 'B'):
                start_lab = lab.split('-')[0]
                tag = lab.split('-')[1]

                # check if its the same label
                if start_lab == 'B':
                    # new label span

                    token = copy_tokens[idx]
                    token_new = '{} {} {}'.format(tag2mark[prev_tag]['e'], tag2mark[tag]['s'], token)
                    copy_tokens[idx] = token_new
                    prev_start_lab = start_lab
                    prev_tag = tag
                else:
                    prev_start_lab = start_lab
                    prev_tag = tag
            elif lab != 'O':
                start_lab = lab.split('-')[0]
                tag = lab.split('-')[1]

                if start_lab == 'B':
                    token = copy_tokens[idx]
                    # modify token
                    token_new = '{} {}'.format(tag2mark[tag]['s'], token)
                    copy_tokens[idx] = token_new

                prev_start_lab = start_lab
                prev_tag = tag

            elif lab == 'O' and (prev_start_lab == 'I' or prev_start_lab == 'B'):
                token = copy_tokens[idx - 1]
                # modify token
                token_new = '{} {}'.format(token, tag2mark[prev_tag]['e'])
                copy_tokens[idx - 1] = token_new

                prev_start_lab = 'O'
                prev_tag = 'O'
            else:
                prev_start_lab = 'O'
                prev_tag = 'O'

        # last
        if label[-1] != 'O':
            start_lab = label[-1].split('-')[0]
            tag = label[-1].split('-')[1]

            if start_lab == 'B' or start_lab == 'I':
                new_token = copy_tokens[-1] + ' {}'.format(tag2mark[tag]['e'])
                copy_tokens[-1] = new_token

        encoded_sentence = ' '.join(copy_tokens)
        return encoded_sentence

    def add_org_translation(self, lang, sent):
        self.org_trans[lang] = sent
    def add_xml_translation(self, lang, sent):
        self.xml_trans[lang] = sent
    def add_marker_translation(self, lang, sent):
        self.marker_trans[lang] = sent
    def add_ent_translation(self, lang, ent_list):
        self.ent_trans[lang] = ent_list



    def __str__(self):
        str = "words:{}\nlabels:{}\nentity_list:{}\ntag_list:{}\nspan:{}\nxml:{}\nmarker:{}".format(
            self.words, self.labels, self.entity_list, self.tag_list, self.span_labels,
            self.xml_encode_sent, self.marker_encode_sent
        )
        return str


#1. load English data
def read_examples_from_file(file_path, lang, lang2id=None):
    if not os.path.exists(file_path):
        logger.info("[Warming] file {} not exists".format(file_path))
        return []
    guid_index = 1
    examples = []
    subword_len_counter = 0
    if lang2id:
        lang_id = lang2id.get(lang, lang2id['en'])
    else:
        lang_id = 0
    logger.info("lang_id={}, lang={}, lang2id={}".format(lang_id, lang, lang2id))
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        langs = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if word:
                    examples.append(InputExample(guid="{}-{}".format(lang, guid_index),
                                                 words=words,
                                                 labels=labels,
                                                 langs=langs))
                    guid_index += 1
                    words = []
                    labels = []
                    langs = []
                    subword_len_counter = 0
                else:
                    print(f'guid_index', guid_index, words, langs, labels, subword_len_counter)
            else:
                splits = line.strip().split()
                word = splits[0]

                words.append(splits[0])
                langs.append(lang_id)
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(guid="%s-%d".format(lang, guid_index),
                                         words=words,
                                         labels=labels,
                                         langs=langs))
    return examples





def save_pickle(file_name, data):
    # save results from model
    if os.path.isfile(file_name):
        print("WARNING! overwriting existing saved files")
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
    print(f"Saved to {file_name}")
    return data

if __name__ == "__main__":
    pass