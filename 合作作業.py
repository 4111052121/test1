import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. 匯入訓練與測試資料
train_file_path = "C:\\Users\\user\\Desktop\\train.csv"
test_file_path = "C:\\Users\\user\\Desktop\\test.csv"

train_df = pd.read_csv(train_file_path, sep=';', header=0)
test_df = pd.read_csv(test_file_path, sep=';', header=0)

# 2. 檢查並顯示缺失值
print("Train Data Missing Values Check:")
missing_train = train_df.isnull().sum()
print(missing_train)
print("\n")

print("Test Data Missing Values Check:")
missing_test = test_df.isnull().sum()
print(missing_test)
print("\n")

# 表格化顯示缺失值總覽
print("Summary of Missing Values in Train and Test Data:")
missing_summary = pd.DataFrame({
    "Train Missing Values": missing_train,
    "Test Missing Values": missing_test
})
print(missing_summary)
print("\n")

# 3. 資料預處理
# 替換二元類別資料為數值
train_df['y'].replace(('yes', 'no'), (1, 0), inplace=True)
train_df['loan'].replace(('yes', 'no'), (1, 0), inplace=True)

test_df['y'].replace(('yes', 'no'), (1, 0), inplace=True)
test_df['loan'].replace(('yes', 'no'), (1, 0), inplace=True)

# 刪除不必要的欄位
columns_to_drop = [
    'job', 'marital', 'education', 'default', 'housing', 
    'contact', 'day', 'month', 'duration', 'campaign', 
    'pdays', 'previous', 'poutcome', 'y'
]

X_train = train_df.drop(columns=columns_to_drop)
y_train = train_df['y']

X_test = test_df.drop(columns=columns_to_drop)
y_test = test_df['y']

# 4. 檢查數據處理後的格式
print("First 5 rows of X_train:")
print(X_train.head())
print("\n")

print("First 5 rows of y_train:")
print(y_train.head())
print("\n")

# 5. 訓練羅吉斯回歸模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. 預測測試資料
y_pred = model.predict(X_test)

# 7. 計算模型準確度
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Model Accuracy: {accuracy:.2f}")

# 8. 可視化分析：繪製長條圖
def plot_horizontal_bars(feature, target='y'):
    count = train_df.groupby([feature, target]).size().unstack()
    bar_height = 0.4
    y = range(len(count))

    plt.figure(figsize=(10, 6))
    plt.barh(y, count[0], height=bar_height, color='skyblue', label='No (Did not subscribe)')
    plt.barh([i + bar_height for i in y], count[1], height=bar_height, color='salmon', label='Yes (Subscribed)')

    for i, (no, yes) in enumerate(zip(count[0], count[1])):
        plt.text(no + 50, i, int(no), va='center', fontsize=8, color='blue')
        plt.text(yes + 50, i + bar_height, int(yes), va='center', fontsize=8, color='red')

    plt.title(f"Count of {target} by {feature}")
    plt.ylabel(feature)
    plt.xlabel("Count")
    plt.yticks([i + bar_height / 2 for i in y], count.index)
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# 繪製多個特徵的長條圖
features_to_plot = ["job", "marital", "education", "loan"]
for feature in features_to_plot:
    plot_horizontal_bars(feature)
