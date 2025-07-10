import string
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, roc_auc_score

fake_df = pd.read_csv('Fake.csv')  
fake_df['label'] = 1  


real_df = pd.read_csv('True.csv')
real_df['label'] = 0  

# Combine
df = pd.concat([fake_df, real_df], ignore_index=True)

# Shuffle the dataset to avoid ordering bias
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove links
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df['text'] = df['text'].apply(clean_text)

vectorizer = TfidfVectorizer(stop_words='english',
    max_df=0.7,          
    min_df=5,            
    max_features=5000,   
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_proba = nb.predict_proba(X_test)[:, 1]

# Accuracy
acc = accuracy_score(y_test, nb.predict(X_test))
print("Accuracy: {:.2f}%".format(acc * 100))

# Cross-validation to detect overfitting
cv_scores = cross_val_score(nb, X, y, cv=5)
print("Cross-validation accuracy: {:.2f}%".format(cv_scores.mean() * 100))

# ROC AUC
nb_auc = roc_auc_score(y_test, nb_proba)
print("ROC AUC Score: {:.2f}".format(nb_auc))

# Save model and vectorizer
joblib.dump(nb, 'naive_bayes_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')