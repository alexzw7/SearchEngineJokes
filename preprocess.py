# This code preprocesses the jokes dataset
import json
from lxml import etree
import re

jokes = []
with open("joke-dataset/reddit_jokes.json") as file:
    jokes = json.load(file)

json_processed = []
jokeSet = jokes[:len(jokes) // 2]

# Build TREC files
with open('trec_files/docs/docs.trec', 'wb') as file:
    file.truncate(0)
    for i in range(len(jokeSet)):
        concat_text = jokeSet[i]['title'] + '\n' + jokeSet[i]['body']
        concat_text.strip()
        cleaned_string = re.sub(u'[^\u0020-\uD7FF\u0009\u000A\u000D\uE000-\uFFFD\U00010000-\U0010FFFF]+', '', concat_text)
        doc = etree.Element('DOC')
        docNo = etree.SubElement(doc, 'DOCNO')
        docNo.text = str(jokeSet[i]['id'])
        docText = etree.SubElement(doc, 'TEXT')
        docText.text = cleaned_string
        file.write(etree.tostring(doc, pretty_print=True))

data = []
for i in range(len(jokeSet)):
    concat_text = jokeSet[i]['title'] + '\n' + jokeSet[i]['body']
    concat_text.strip()
    cleaned_string = re.sub(u'[^\u0020-\uD7FF\u0009\u000A\u000D\uE000-\uFFFD\U00010000-\U0010FFFF]+', '', concat_text)
    data.append({'docno' : jokeSet[i]['id'], 'text' : cleaned_string})

with open("docs.json", "w") as processed_file:
    processed_file.truncate(0)
    json.dump(data, processed_file)