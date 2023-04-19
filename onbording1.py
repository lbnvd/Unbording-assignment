import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

"""Loading the data"""
#list of columns
columns=["sentiment", "id", "date", "query", "user", "text"]
#load the test data from the CSV file 
df_test = pd.read_csv("testdata.manual.2009.06.14.csv", header=None, names=columns, encoding="ISO-8859-1")
#load the training data from the CSV file
df_train = pd.read_csv("training.1600000.processed.noemoticon.csv",  header=None, names=columns, encoding="ISO-8859-1")
#load the test data from the hand made CSV file 
df_test_hand_made=pd.read_csv("dataset_handmade.csv", header=None, names=["sentiment", "text"],  delimiter=',', encoding="ISO-8859-1")

"""Cleaning the data"""
#drop unnecessary columns
df_train = df_train.drop(["id", "date", "query", "user"], axis=1)
df_test = df_test.drop(["id", "date", "query", "user"], axis=1)

#convert to lowercase all the data
df_train["text"] = df_train["text"].str.lower()  
df_test["text"] = df_test["text"].str.lower()  
df_test_hand_made["text"] = df_test_hand_made["text"].astype(str).str.lower()
#check for missing data
assert not df_train.isna().any().any()
#drop the sentiment neutre in the test df
df_test.drop(df_test[df_test['sentiment'] == 2].index,  inplace = True)
#lemmatization method
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
#function for cleaning the data
def clean_text(text):
    text = re.sub(r"http\S+", "", text)#remove URLs
    text = re.sub(r"@[A-Za-z0-9]+", "", text) #Remove mentions (@username)
    text = re.sub(r"#[A-Za-z0-9]+", "", text)  #remove hashtags (#hashtag)
    text = re.sub(r"\d", "", text) #remove digits
    text = re.sub(r"[^\w\s]+", "", text) #remove non alphanumerique like emogies
    text = re.sub(r"[^a-zA-Z\s]+", "", text) #remove non alphabetic
    text = re.sub(r"\b(?:am|is|are|was|were|been|being)\b", "", text) #remove all form of verb be
    text = re.sub(r"\b(?:do|does|did|doing)\b", "", text) #remove all forme of verb do
    text = re.sub(r"\b(?:go|goes|went|going)\b", "", text) #verb go
    text = re.sub(r"\b(?:get|gets|got|getting)\b", "", text) #verb get
    text = re.sub(r"\b(?:make|makes|made|making)\b", "", text) #verb make
    text = re.sub(r"\b\w+ly\b", "", text) #all adverb
    text = re.sub(r"\b(?:ah|oh|eh|uh)\b", "", text) #all interjections
    # Remove stop words and stem the words
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    text = " ".join(words)
    # Lemmatize the words
    #words = word_tokenize(text)
    #words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    #text = " ".join(words)
    return text 

#apply the function to the dataset
df_train["text"] = df_train["text"].apply(clean_text)
df_test["text"] = df_test["text"].apply(clean_text)
df_test_hand_made["text"] = df_test_hand_made["text"].apply(clean_text)
#check the five first lines of the datasets
print("\n\ndf train : \n", df_train.head())
print("\n\ndf test :\n", df_test.head())
#print("\n\ndf hand made test :\n", df_test_hand_made.head())

"""Visualization"""

#split df into positive and negative sentiment
df_train_positive = df_train[df_train['sentiment'] == 4]
df_train_negative = df_train[df_train['sentiment'] == 0]
#concatenate all the sentiment values into a single string
all_sentiments = " ".join(sentiment for sentiment in df_train["text"])
#generate the word cloud object
cloud_sentiments = WordCloud().generate(all_sentiments)
#concatetane all positive sentiment into a single string
pos_sentiment=" ".join(sentiment for sentiment in df_train_positive["text"])
#generate the word cloud object
cloud_sentiments_pos = WordCloud().generate(pos_sentiment)
#concatenate all negative sentiment into a signle string
neg_sentiment=" ".join(sentiment for sentiment in df_train_negative["text"])
#generate the word cloud object
cloud_sentiments_neg = WordCloud().generate(neg_sentiment)
#create the plot
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
# Create the wordcloud subplots
axes[0].imshow(cloud_sentiments, interpolation="bilinear")
axes[0].set_title("Wordcloud all sentiments")
axes[1].imshow(cloud_sentiments_pos, interpolation="bilinear")
axes[1].set_title("Wordcloud sentiments positifs")
axes[2].imshow(cloud_sentiments_neg, interpolation="bilinear")
axes[2].set_title("Wordcloud sentiments negatifs")
# Count the number of positive and negative tweets
positive_count = df_train[df_train["sentiment"] == 4].count()[0]
negative_count = df_train[df_train["sentiment"] == 0].count()[0]
# Create the bar chart
axes[3].bar(["Positive", "Negative"], [positive_count, negative_count])
axes[3].set_title("Sentiment Distribution")
# Create the pie chart
axes[4].pie([positive_count, negative_count], labels=["Positive", "Negative"])
axes[4].set_title("Sentiment Proportion")
# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.8)
# Display the plot
plt.show()


"""Vectorization"""
#vectorize the text data using CountVectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=1000, ngram_range = (1,1), encoding='latin-1')
#apply on text columns of df
X_train = vectorizer.fit_transform(df_train['text'])
X_test=vectorizer.transform(df_test['text'])
#X_test_hand_made=vectorizer.transform(df_test_hand_made['text'])

"""Split the data"""
#extract the sentiment values
y_train = df_train['sentiment']
y_test = df_test['sentiment'].values
#y_test_hand_made = df_test_hand_made['sentiment'].values
#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
#X_train, X_test_hand_made, y_train, y_test_hand_made = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

"""Models"""
#train a Logistic Regression model
print("\nLOGISTIC REGRESSION MODEL\n")
clf = LogisticRegression()
clf.fit(X_train, y_train)
#make predictions on the test set
y_pred_clf = clf.predict(X_test)
#compute the accuracy of the model
accuracy = accuracy_score(y_test, y_pred_clf)*100
print("Model logistic regression accuracy:", accuracy)
print("Confusion matrix :\n", confusion_matrix(y_test, y_pred_clf))
print("Classification report :\n", classification_report(y_test,y_pred_clf))
"""
#models with the hand made data
y_pred_clf_hand_made = clf.predict(X_test_hand_made)
accuracy_hand_made = accuracy_score(y_test_hand_made, y_pred_clf_hand_made)*100
print("Model logistic regression accuracy:", accuracy_hand_made)
print("Confusion matrix :\n", confusion_matrix(y_test_hand_made, y_pred_clf_hand_made))
print("Classification report :\n", classification_report(y_test_hand_made,y_pred_clf_hand_made))
"""

print("\nRAND FOREST MODEL : \n")
randomForest=RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
randomForest.fit(X_train.toarray(), y_train)
y_pred_randomForest=randomForest.predict(X_test.toarray())
accuracy = accuracy_score(y_test, y_pred_randomForest)*100
print('Random Forest model accuracy:', accuracy)
print("Confusion Matrix :\n", confusion_matrix(y_test, y_pred_randomForest))
print("Classification report :\n", classification_report(y_test,y_pred_randomForest))

print("\nSVC MODEL:\n")
SVCmodel=LinearSVC(max_iter=10000)
SVCmodel.fit(X_train, y_train)
y_pred_SVC = SVCmodel.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_SVC)*100
print('SVC accuracy:', accuracy)
print("Confusion Matrix :\n",confusion_matrix(y_test, y_pred_SVC))
print("Classification report :\n",classification_report(y_test,y_pred_SVC))

"""
print("\nKNN MODEL:\n")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
knn_score = knn.score(X_test, y_test)
knn_accuracy=accuracy_score(y_test, y_pred_knn)
print("Model KNN Classifier accuracy:", knn_accuracy)
"""

"""
print("\nNAIVE BAYES MODEL : \n")
naiveBayes=GaussianNB()
X_train_dense = X_train.toarray()
naiveBayes.fit(X_train_dense, y_train)
y_pred_naiveBayes=naiveBayes.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_naiveBayes)*100
print('Naive Bayes model accuracy:', accuracy)
print("Confusion Matrix :\n",confusion_matrix(y_test, y_pred_naiveBayes))
print("Classification report :\n",classification_report(y_test,y_pred_naiveBayes))
"""