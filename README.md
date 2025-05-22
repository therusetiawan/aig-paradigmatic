# AIG Paradigmatic

AIG Paradigmatic is an application that uses a FastText model to generate distractors in English Vocabulary Test. 
This guide will help you set up the environment, prepare the necessary files, and run the application.

## Prerequisites
* Python
* PIP (Python Package Installer)
* Python Dependencies: `flask spacy fasttext nltk scipy pyinflect`
* FastText Model: [cc.en.300.bin](https://fasttext.cc/docs/en/crawl-vectors.html)

## How to Run
* Place the fasttext model file inside the `models` directory. See `app.py` in line 19
* Run flask app
  ```python -m flask --app app.py --debug run```
