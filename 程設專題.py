import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# 確保下載 NLTK 所需的資源
nltk.download('stopwords')
nltk.download('wordnet')

# 資料集路徑
file_path = r"C:\Users\user\Desktop\程設專題\archive (3)\all-data.csv"

# 讀取數據
data = pd.read_csv(file_path, names=["Sentiment", "News_Headline"], encoding='latin1')

# 清洗文本
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # 移除標點符號
    text = text.lower()  # 轉小寫
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])  # 去停用詞
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])  # 詞形還原
    return text

data['Cleaned_Headline'] = data['News_Headline'].apply(preprocess_text)

# 特徵提取（TF-IDF）
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['Cleaned_Headline'])

# 標籤編碼
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(data['Sentiment'])

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型訓練
model = LogisticRegression()
model.fit(X_train, y_train)

# 評估模型
y_pred = model.predict(X_test)
print("分類報告：")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 混淆矩陣
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 用戶界面功能
def predict_sentiment(input_text):
    cleaned_text = preprocess_text(input_text)
    vectorized_text = tfidf.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    sentiment = le.inverse_transform(prediction)[0]
    return sentiment

# 測試用戶界面
while True:
    user_input = input("輸入一條金融新聞（或輸入 'exit' 離開）：")
    if user_input.lower() == 'exit':
        print("程式結束。")
        break
    sentiment = predict_sentiment(user_input)
    print(f"預測情緒：{sentiment}")
