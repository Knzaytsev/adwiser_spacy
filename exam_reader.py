import pandas as pd
from os import listdir
from os.path import isfile, join, basename
import re
from numpy import empty, mat
import spacy
from spacy.tokens import Token
from annotator import output_maker
from spacy.matcher import DependencyMatcher
from spacy.matcher import Matcher
from spellchecker import SpellChecker
from spacy import displacy
# import treetaggerwrapper

nlp = spacy.load("en_core_web_lg")

from models import generate_text, models, find_span, without_child

from pprint import pprint

def preprocess(sentences):
        sentence_dots: str = re.sub(r'\.', '. ', sentences)
        sentence_dots = re.sub(r'\.\s\.\s\.', '...', sentence_dots)
        sentence_dots = re.sub(r'\.\s\.', '..', sentence_dots)
        sentence_dots = re.sub(r'\!', '! ', sentence_dots)
        sentence_dots = re.sub(r'\?', '? ', sentence_dots)
        sentence_dots = re.sub(r'\s\s', ' ', sentence_dots)
        sentence_dots = re.sub(r"'ve", ' have', sentence_dots)
        return sentence_dots

def get_match(text):
    text_ = preprocess(text)
    doc = nlp(text_)
    pp_texts = []
    for _, sentence in enumerate(doc.sents):
        #print(sentence)

        results, errors = match_pp(sentence)
        mistake = None
        if results:
            if errors:
                last_place_mistake = (-1, -1)
                for error in errors:
                    mistake = text[error[0][0]:error[0][1]]
                    if last_place_mistake[0] == error[0][0] and last_place_mistake[1] == error[0][1]:
                        continue
                    last_place_mistake = (error[0][0], error[0][1])
                    #print(error)
                    #print(sentence.text)
                    #print(mistake)
                    #pp_texts.append([mistake, sentence.text.replace('\n', '')])
            #else:
                #pp_texts.append(['-', sentence.text.replace('\n', '')])
            for result in results:
                pp_texts.append([sentence[result[1][-1]].text, sentence[result[1][-2]].text, mistake is not None, sentence.text.replace('\n', '')])
    return pp_texts

def find_wrong_verb_features(pp, sent):
        # проверить лицо и число у слова и глагола
        errors = []
        for match in pp:
            have_index = match[1][-1]
            have_verb = sent[have_index]
            verb = have_verb.head
            children = verb.children
            subj = None
            
            for child in children:
                if child.dep_ in ['relcl', 'nsubj', 'nsubjpass']:
                    subj = child
                    break
            
            if subj == None:
                subj = verb.head
            
            have_number = have_verb.morph.get('Number')
            subj_number = subj.morph.get('Number')
            have_person = have_verb.morph.get('Person')
            subj_person = subj.morph.get('Person')
            
            is_1_2_person = subj_person and (subj_person[0] == '1' or subj_person[0] == '2')
            is_plural = subj_number and subj_number[0] == 'Plur'
            is_sing = subj_number and subj_number[0] == 'Sing'
            
            if (is_1_2_person or is_plural) and have_verb.text != 'have':
                errors.append((have_verb, 'Probably you should write have.'))
                continue 
            
            if not is_1_2_person and is_sing and have_verb.text != 'has':
                errors.append((have_verb, 'Probably you should write has'))
                continue 
        return errors

def clear_matches(matches):
    cur_matches = [match[1][0] for match in matches]
    for i in range(len(cur_matches) - 1):
        for j in range(i + 1, len(cur_matches)):
            if cur_matches[i] == cur_matches[j]:
                del matches[j]
    return matches

def check_got(sentence):
    dep_matcher = DependencyMatcher(vocab=nlp.vocab)
    present_perfect = []
    present_perfect.append([{'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'TAG': 'VBN', 'LEMMA': 'get'}},
                        {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'have',
                        'RIGHT_ATTRS': {'LEMMA': 'have', 'DEP': 'aux', 'TAG': {'IN': ['VBP', 'VBZ']}}},
                        {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'do_verb', 'RIGHT_ATTRS': {'TAG': 'VB'}},
                        {'LEFT_ID': 'do_verb', 'REL_OP': '>', 'RIGHT_ID': 'to', 'RIGHT_ATTRS': {'TAG': 'TO'}},    
                           ])

    
    dep_matcher.add("present_perfect", patterns=present_perfect)
    results = dep_matcher(sentence)
    
    return results

def find_wrong_verb_features(pp, sent):
    # проверить лицо и число у слова и глагола
    errors = []
    for match in pp:
        have_index = match[1][-1]
        have_verb = sent[have_index]
        if have_verb.tag_ != 'aux':
            continue
        verb = have_verb.head
        children = verb.children
        subj = None
        
        for child in children:
            if child.dep_ in ['relcl', 'nsubj', 'nsubjpass']:
                subj = child
                break
        
        if subj == None:
            subj = verb.head
        
        have_number = have_verb.morph.get('Number')
        subj_number = subj.morph.get('Number')
        have_person = have_verb.morph.get('Person')
        subj_person = subj.morph.get('Person')
        
        is_1_2_person = subj_person and (subj_person[0] == '1' or subj_person[0] == '2')
        is_plural = subj_number and subj_number[0] == 'Plur'
        is_sing = subj_number and subj_number[0] == 'Sing'
        
        if (is_1_2_person or is_plural) and have_verb.text != 'have':
            errors.append((have_verb, 'Probably you should write have.'))
            continue 
        
        if not is_1_2_person and is_sing and have_verb.text != 'has':
            errors.append((have_verb, 'Probably you should write has'))
            continue 
    return errors

def match_conj_verb(pp, sent):
    results = []
    
    for match in pp:
        parent = sent[match[1][0]].head
        if sent[match[1][0]].dep_ != 'ROOT':
            results.append((sent[match[1][0]], sent[match[1][0]].tag_ != parent.tag_))
       
    return results

def check_year(pp, sent):
    years = [
        (sent[index], not (1989 <= int(sent[index].text) <= 2022)) for indices in pp 
        for index in indices[1] 
        if sent[index].tag_ in ['NUM', 'CD']
    ]
    return years

def match_pp(sent):
    errors_pp = []
    all_errors = []
    dep_matcher = DependencyMatcher(vocab=nlp.vocab)

    if check_got(sent):
            return [], sent
    
    present_perfect = []
    present_perfect.append([{'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'TAG': 'VBN'}},
                        {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'have',
                        'RIGHT_ATTRS': {'LEMMA': 'have', 'DEP': 'aux', 'TAG': {'IN': ['VBP', 'VBZ']}}}])

    present_perfect.append([{'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'TAG': {'IN' : ['VBN', 'VBG']}}},
                        {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'be',
                        'RIGHT_ATTRS': {'LEMMA': 'be', 'DEP': 'aux', 'TAG': {'IN': ['VBP', 'VBZ', 'VBN']}}},
                       {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'have',
                        'RIGHT_ATTRS': {'LEMMA': 'have', 'DEP': 'aux', 'TAG': {'IN': ['VBP', 'VBZ', 'VBN']}}}
                        ])
    
    
    present_perfect.append([{'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'DEP': 'conj', 
                                                                 'POS': 'VERB'
                                                                }},
                            {'LEFT_ID': 'verb', 'REL_OP': '$--', 'RIGHT_ID': 'have',
                            'RIGHT_ATTRS': {'LEMMA': 'have', 'DEP': 'aux', 'TAG': {'IN': ['VBP', 'VBZ']}}}
                           ])
    
    dep_matcher.add("present_perfect", patterns=present_perfect)
    results = clear_matches(dep_matcher(sent))
    
    if results:
        error_message = 'Present Perfect does not go along with indication of time in the past.'
        all_matches = []
        verbs = []
        dep_matcher_ = DependencyMatcher(vocab=nlp.vocab)

        # in/from/over/between + CD
        patt_one = [{'RIGHT_ID': 'prep', 'RIGHT_ATTRS':
            {'LEMMA': {'IN': ['from', 'in', 'over', 'between']}}},
                    {'LEFT_ID': 'prep', 'REL_OP': '>', 'RIGHT_ID': 'year',
                        'RIGHT_ATTRS': {'TAG': 'CD', 'DEP': 'pobj'}}]

        dep_matcher_.add('prep+cd', patterns=[patt_one])
        dep_matches = dep_matcher_(sent)

        # + у последнего нет детей с леммой recent/last
        if dep_matches and without_child(sent[dep_matches[0][1][1]], {'lemma_': ['last', 'recent']}):
            all_matches.append(dep_matches[0][1][0])
            if sent[dep_matches[0][1][0]].head.tag_ == 'VBN':
                verbs.append(sent[dep_matches[0][1][0]].head)

        # in/from/over/between+year
        patt_two = [{'RIGHT_ID': 'prep', 'RIGHT_ATTRS': {'LEMMA': {'IN': ['at', 'over', 'in']}}},
                    {'LEFT_ID': 'prep', 'REL_OP': '>', 'RIGHT_ID': 'year',
                        'RIGHT_ATTRS': {'LEMMA': 'year', 'DEP': 'pobj'}}]

        dep_matcher_.add('prep+year', patterns=[patt_two])
        # at/in/over + last/recent + NN
        patt_three = [{'RIGHT_ID': 'prep', 'RIGHT_ATTRS': {'LEMMA': {'IN': ['at', 'over', 'in']}}},
                        {'LEFT_ID': 'prep', 'REL_OP': '>', 'RIGHT_ID': 'noun',
                        'RIGHT_ATTRS': {'TAG': 'NN', 'DEP': {'IN': ['pobj', 'npadvmod']}}},
                        {'LEFT_ID': 'noun', 'REL_OP': '>', 'RIGHT_ID': 'adj',
                        'RIGHT_ATTRS': {'LEMMA': {'IN': ['last', 'initial']}}}]

        dep_matcher_.add('prep+noun+last/initial', patterns=[patt_three])
        # at/in/over + ordinal + NN
        patt_four = [{'RIGHT_ID': 'prep', 'RIGHT_ATTRS': {'LEMMA': {'IN': ['at', 'over', 'in']}}},
                        {'LEFT_ID': 'prep', 'REL_OP': '>', 'RIGHT_ID': 'noun',
                        'RIGHT_ATTRS': {'TAG': 'NN', 'DEP': {'IN': ['pobj', 'npadvmod']}}},
                        {'LEFT_ID': 'noun', 'REL_OP': '>', 'RIGHT_ID': 'ord_adj',
                        'RIGHT_ATTRS': {'ENT_TYPE': 'ORDINAL'}}]

        dep_matcher_.add('prep+noun+ordinal', patterns=[patt_four])
        # ago (but not since ...a long ago )
        # I have done it since 2 years ago.
        find_ago = [ago for ago in sent if ago.lemma_ == 'ago']
        for ago_ in find_ago:
            if ago_.head.tag_ == 'VBN':
                verbs.append(ago_.head)

        # last + noun of periods
        patt_five = [{'RIGHT_ID': 'noun', 'RIGHT_ATTRS': {'TAG': 'NN', 'DEP': {'IN': ['pobj', 'npadvmod']}, 'LEMMA':
            {'IN': ['year', 'term', 'week', 'semester', 'century'
                                                        'day', 'month', 'decade', 'spring', 'fall',
                    'autumn', 'winter', 'summer', 'night', 'evening',
                    'morning', 'season', 'stage', 'point', 'phase']}}},
                        {'LEFT_ID': 'noun', 'REL_OP': '>', 'RIGHT_ID': 'last', 'RIGHT_ATTRS': {'LEMMA': 'last'}}]

        dep_matcher_.add('last+nn', patterns=[patt_five])

        # since/from cd to cd
        patt_six = [{'RIGHT_ID': 'vbn', 'RIGHT_ATTRS': {'TAG': 'VBN'}},
                    {'LEFT_ID': 'vbn', 'REL_OP': '>', 'RIGHT_ID': 'prep',
                        'RIGHT_ATTRS': {'LEMMA': {'IN': ['from', 'since']}, 'DEP': 'prep'}},
                    {'LEFT_ID': 'prep', 'REL_OP': '>', 'RIGHT_ID': 'cd',
                        'RIGHT_ATTRS': {'TAG': 'CD', 'DEP': {'IN': ['pobj', 'npadvmod']}}},
                    {'LEFT_ID': 'vbn', 'REL_OP': '>', 'RIGHT_ID': 'to',
                        'RIGHT_ATTRS': {'LEMMA': 'to', 'DEP': 'prep'}},
                    {'LEFT_ID': 'to', 'REL_OP': '>', 'RIGHT_ID': 'cd_two', 'RIGHT_ATTRS': {'TAG': 'CD'}}]

        dep_matcher_.add('since_1998_to_2000', patterns=[patt_six])

        patt_seven = [{'RIGHT_ID': 'prep', 'RIGHT_ATTRS': {'LEMMA': 'at'}},
            {'LEFT_ID': 'prep', 'REL_OP': '>', 'RIGHT_ID': 'cd',
                'RIGHT_ATTRS': {'TAG': {'IN': ['CD', 'NUM']}, 'DEP': 'pobj'}}]
    
        dep_matcher_.add('at_pattern', patterns=[patt_seven])
        
        wrong_years = check_year(dep_matcher_(sent), sent)
        
        for year in wrong_years:
            verb, error = year
            if error:
                all_errors.append([find_span([verb]), error])

        yesterday = [yestdy for yestdy in sent if yestdy.lemma_ == 'yesterday']
        for yestdy in yesterday:
            if yestdy.head.head.head.tag_ == 'VBN':
                verbs.append(yestdy.head.head.head)
        # Verbs
        if dep_matcher_(sent):
            for match_dep in dep_matcher_(sent):
                if sent[match_dep[1][0]].head.tag_ == 'VBN':
                    verbs.append(sent[match_dep[1][0]].head)
                if nlp.vocab[match_dep[0]].text == 'since_1998_to_2000':
                    if sent[match_dep[1][1]].lemma_ == 'since':
                        tok = sent[match_dep[1][1]]
                        all_errors.append([find_span([tok]),
                                            'You may need \'from\' instead of \'since\'.'])

        # have + not от каждого глагола + ошибки
        for verb in verbs:

            if verb.tag_ == 'VBN':
                
                have_not = [have for have in verb.children if have.lemma_ == 'have' and have.dep_ == 'aux'
                            and have.tag_ in {'VBZ', 'VBP'}]
                if have_not:
                    
                    have_not = have_not[0]
                    errors_pp.append(have_not)
                    not_ = [i for i in have_not.children if i.dep_ == 'neg' and i.norm_ == 'not']
                    if not_:
                        errors_pp.append(not_)
                    all_errors.append([find_span(errors_pp), error_message])

        wrong_features = find_wrong_verb_features(results, sent)
        
        for feature in wrong_features:
                verb, error = feature
                all_errors.append([find_span([verb]), error])

        conj_errors = match_conj_verb(results, sent)
    
        for conj in conj_errors:
            verb, is_error = conj
            if is_error:
                all_errors.append([find_span([verb]), 'Wrong conjinction.'])

    return results, all_errors

path = 'F:\Laboratory\exam'

def create_csv():
    dirs = listdir(path)
    files = []
    for dir in dirs:
        files.extend([f'{path}\{dir}\{f}' for f in listdir(f'{path}\{dir}') if isfile(join(f'{path}\{dir}', f)) and '.txt' in f])

    results = {}
    for file in files:
        with open(file) as f:
            text = f.read()
            try:
                #results[file] = generate_text(text)
                match = get_match(text)
                if match:
                    results[file] = match
            except BaseException as err:
                print(f"Unexpected {err=}, {type(err)=}")
            finally:
                continue

    dataset = []
    for file, texts in results.items():
        for text in texts:
            #print(text)
            text.insert(0, basename(file))
            dataset.append(text)

    df = pd.DataFrame(data=dataset, columns=['File', 'Token', 'Word_2', 'Error', 'Sentence'])
    df.to_csv(f'{path}\sentences.csv', index=False, sep=',')
    print('done!')


create_csv()

#from test_reader import get_test_data

#tests = get_test_data()

#print(tests[:10])


