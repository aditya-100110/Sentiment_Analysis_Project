import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("1429_1.csv", encoding="latin1")
df = df[["reviews.text", "reviews.rating"]].dropna()
df.columns = ["text", "rating"]

# Label reviews
df["sentiment"] = df["rating"].apply(lambda x: "positive" if x >= 4 else "negative")

# Balance classes
min_count = df["sentiment"].value_counts().min()
df_balanced = df.groupby("sentiment").apply(lambda x: x.sample(min_count)).reset_index(drop=True)

# Stopword list
stopwords = set("""
a about above after again against all am an and any are aren't as at be because been before
being below between both but by can't cannot could couldn't did didn't do does doesn't doing
don't down during each few for from further had hadn't has hasn't have haven't having he he'd
he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into
is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only
or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't
so some such than that that's the their theirs them themselves then there there's these they they'd
they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're
we've were weren't what what's when when's where where's which while who who's whom why why's with
won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

# Preprocessing
def preprocess(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))  # Remove punctuation/numbers
    tokens = text.lower().split()
    return " ".join(word for word in tokens if word not in stopwords)

df_balanced["cleaned"] = df_balanced["text"].apply(preprocess)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced["cleaned"], df_balanced["sentiment"], test_size=0.2, random_state=42
)

# Vectorizer: Tfidf
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Model
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Evaluation
y_pred = model.predict(X_test_vect)
print(classification_report(y_test, y_pred))

# Save model/vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ model.pkl and vectorizer.pkl saved!")

