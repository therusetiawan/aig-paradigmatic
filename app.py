from flask import Flask, render_template, request
import spacy, sys, fasttext, re
from nltk.corpus import wordnet as wn
from scipy.spatial.distance import cosine
from operator import itemgetter
import pyinflect
import psutil
from resource import getrusage, RUSAGE_SELF

app = Flask(__name__)

@app.route('/paradigmatic', methods=['GET'])
def paradigmatic():
    return render_template('paradigmatic_form.html')

@app.route('/paradigmatic_result', methods=['GET'])
def paradigmatic_result():
    sp = spacy.load('en_core_web_sm')
    model = fasttext.load_model('models/cc.en.300.bin')

    sentence = request.args.get('sentence')
    stem = request.args.get('stem')
    correct_answer_raw = request.args.get('correct_answer')

    # get target word from stem
    target_word = re.findall('"([^"]*)"', stem)
    target_word = target_word[0]

    sentence_with_target_word = sp(sentence)
    sentence_with_correct_answer = sp(sentence.replace(target_word, correct_answer_raw))

    def cosine_similarity(embedding_1, embedding_2):
            # Calculate the cosine similarity of the two embeddings.
            sim = 1 - cosine(embedding_1, embedding_2)
            return round(sim,3)

    for word in sentence_with_target_word:
        if word.text == target_word:
            target_word_POS = word.pos_
            target_word_TAG = word.tag_

    for word in sentence_with_correct_answer:
        if word.text == correct_answer_raw:
            correct_answer_POS = word.pos_
            correct_answer_TAG = word.tag_
            correct_answer = word.lemma_

    match correct_answer_POS:
        case "VERB":
            pos_code = 'v'
        case "NOUN":
            pos_code = 'n'
        case "ADJ":
            pos_code = 'a'
        case "ADV":
            pos_code = 'r'

    # correct answer preprocessing
    if correct_answer_TAG == "VBG":
        preprocessing='gerund'
    elif correct_answer_TAG == "VBD":
        preprocessing='past_tense'
    elif correct_answer_TAG == "VBN":
        preprocessing='past_participle'
    elif correct_answer_TAG == "NN":
        preprocessing='noun_singular'
    elif correct_answer_TAG == "NNS":
        preprocessing='noun_plural'
    else:
        preprocessing='default'

    correct_answer_embedding = model.get_word_vector(correct_answer)
    # print("This is correct answer embedding")
    # print(correct_answer_embedding)
    # correct_answer_embedding = 0
    synsets = wn.synsets(correct_answer, pos=pos_code)
    number_of_synsets = len(synsets)

    candidates = []
    synsets_with_lemmas = [[]]
    i = 0
    for syn in wn.synsets(correct_answer, pos=pos_code):
        for l in syn.lemmas():

            match preprocessing:
                case "gerund":
                    candidate=sp(l.name())[0]._.inflect('VBG')
                case "past_tense":
                    candidate=sp(l.name())[0]._.inflect('VBD')
                case "past_participle":
                    candidate=sp(l.name())[0]._.inflect('VBN')
                case "noun_singular":
                    candidate=sp(l.name())[0]._.inflect('NN')
                case "noun_plural":
                    candidate=sp(l.name())[0]._.inflect('NNS')
                case "default":
                    candidate=l.name()

            if candidate is None:
                candidate=l.name()
            
            sentence_with_candidate = sp(sentence.replace(target_word, candidate))
            for word in sentence_with_candidate:
                if word.text == candidate:
                    candidate_POS = word.pos_
                    candidate_TAG = word.tag_

            synsets_with_lemmas[i].append(l.name())

            embedding = model.get_word_vector(candidate)
            # if candidate == 'acquisitions':
            #     print("This is distractor acquisitions")
            #     print(embedding)
            sim = cosine_similarity(correct_answer_embedding, embedding)
            # sim = 0

            # skip word that contain target word & correct answer
            if target_word in l.name() or correct_answer in l.name():
                continue

            if candidate_POS == correct_answer_POS:
                candidates.append([candidate, sim, candidate_POS, candidate_TAG])
        
        if i != number_of_synsets:
            synsets_with_lemmas.append([])
            i+=1
    # TODO: remove duplicate candidate

    candidates = sorted(candidates, key=itemgetter(1))
    print("Peak Memory:", int(getrusage(RUSAGE_SELF).ru_maxrss / 1024**2))

    return render_template('paradigmatic_result.html', 
    sentence=sentence, 
    stem=stem,
    correct_answer=correct_answer_raw, 
    target_word_pos=target_word_POS,
    correct_answer_pos=correct_answer_POS,
    target_word_tag=target_word_TAG,
    correct_answer_tag=correct_answer_TAG,
    synsets=synsets,
    number_of_synsets=number_of_synsets,
    synsets_with_lemmas=synsets_with_lemmas,
    candidates=candidates)