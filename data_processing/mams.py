from bs4 import BeautifulSoup
from tqdm import tqdm
from compress_pickle import dump

root = BeautifulSoup(open(r'C:\Users\HongM\PycharmProjects\aspect-sentiment-dashboard\text_data\mams-asca-train.xml'))

categories = []

for ac in root.find_all('aspectcategory'):

    categories.append(ac['category'])

set_of_categories = set(categories)

print(set_of_categories)

examples = []

for sentence in tqdm(root.find_all('sentence')):

    example = {}

    example['text'] = sentence.text.replace('\n', '')

    aspects = {k: 'none' for k in set_of_categories}

    for cat in sentence.find_all('aspectcategory'):
        aspects[cat.get('category')] = cat.get('polarity')

    example['aspects'] = aspects

    examples.append(example)

dump(examples, '../text_data/mams-asca-train.gz')