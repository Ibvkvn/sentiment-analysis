import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import movie_reviews

nltk.download("movie_reviews")

documents = [
    (" ".join(movie_reviews.words(field)), category)
    for category in movie_reviews.categories()
    for field in movie_reviews.fileids(category)
]

df = pd.DataFrame(documents, columns=["review", "sentiment"])

vectorizer = CountVectorizer(max_features=2000)
x = vectorizer.fit_transform(df["review"])
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"classification Report:\n{classification_report(y_test, y_pred)}")

def preditct_sentiment(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]

print(preditct_sentiment("This movie is fantastic!"))
print(preditct_sentiment("This movie is terrible!"))
print(preditct_sentiment("This movie is okay, nothing special."))