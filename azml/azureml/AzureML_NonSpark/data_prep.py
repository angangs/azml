from argparse import ArgumentParser
from azureml.core import Run
import joblib
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('punkt')
nltk.download('stopwords')


parser = ArgumentParser()
parser.add_argument('--datafolder', type=str, dest='datafolder')
parser.add_argument('--input', type=str, dest='input')
parser.add_argument('--output_name', type=str, dest='output_name')
parser.add_argument('--process', type=str, dest='process')
args = parser.parse_args()

run = Run.get_context()

raw_data = run.input_datasets[args.input]

###### Bag of Words Model ######
### Data Preprocessing ###

filtered_corpus = []
data = raw_data.to_pandas_dataframe()


corpus = list(data["Item Description"].values)
corpus = [word.lower() for word in corpus]
corpus = [word_tokenize(i) for i in corpus]

stopwordlist = stopwords.words("english")
stop_words = set(w.lower() for w in stopwordlist)
porter = PorterStemmer()
for s in corpus:
    lst = [j for j in s if j not in stop_words]  # remove stopwords
    lst = [re.sub(r'[0-9]+', '', i) for i in lst]  # remove numbers
    lst = [re.sub('[^A-Za-z0-9]+', '', i) for i in lst]  # remove special characters, punctuation and keep only alphanumeric
    lst = [porter.stem(i) for i in lst] # stem words
    lst = [i for i in lst if i]  # remove whitespaces from each list
    filtered_corpus.append(lst)


### Document Term Matrix ###

# We are using this analyzer as the corpus is a list of lists with strings
vec = CountVectorizer(analyzer=lambda filtered_corpus: filtered_corpus, min_df=2)

X = vec.fit_transform(filtered_corpus)
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

# Log number of features
run.log("Number of features", len(df.columns))


if args.process == 'training':

    # Register features as pkl to Models for inference
    joblib.dump(value=df.columns, filename='outputs/features.pkl')
    run.upload_file("outputs/features.pkl", "outputs/features.pkl")
    model = run.register_model(model_path='outputs/features.pkl',
                               model_name='Features',
                               description='Number of features')

    # Concatenate corpus terms with each category
    dtm_df = pd.concat([df, data["Category L4"]], axis=1)
else:
    dtm_df = df.copy()

# Output final dataframe for training step
os.makedirs(args.datafolder, exist_ok=True)

# Create the path
path = os.path.join(args.datafolder, args.output_name)

# Write the data preparation output as csv file
dtm_df.to_csv(path, index=False)

run.complete()




