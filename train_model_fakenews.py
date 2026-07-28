import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

df = pd.read_csv('news.csv')
df.columns = ['id', 'title', 'text', 'label']
df['label'] = df['label'].str.upper()
df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    min_df=2,
    ngram_range=(1, 2),
    sublinear_tf=True,
)

X_train, X_test, y_train, y_test = train_test_split(
    df['combined_text'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

model = LinearSVC(class_weight='balanced', random_state=42)
model.fit(tfidf_train, y_train)

y_pred = model.predict(tfidf_test)
print(classification_report(y_test, y_pred, digits=3))

with open('tfid.pickle', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

with open('model_fakenews.pickle', 'wb') as f:
    pickle.dump(model, f)