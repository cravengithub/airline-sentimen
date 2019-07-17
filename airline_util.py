
import re

from nltk.corpus import stopwords


def get_corpus():
    corpus = [
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining and the weather is sweet',
    'Is that good news to day of this sun?']
    return corpus

def tokenizer(text):
    stop = stopwords.words('english')
    text = re.sub('<[^>]*>', '', text)
    hashTagRemover = [w for w in text.split() if not w.startswith('#') ]
    retweetRemover = [w for w in hashTagRemover if not w.startswith('@') ]
    str = ' '.join(retweetRemover)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', str.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

'''
dataFrame = pd.read_csv('train.csv')
sentiment = np.array(dataFrame.get('airline_sentiment'))
print(sentiment)
'''

'''
sample = "#readMore @VirginAmerica I didn't today... Must mean I need to take another trip!"
res = tokenizer(sample)
print(res)
'''