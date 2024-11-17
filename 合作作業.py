import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. 匯入資料
train_file_path = "C:\\Users\\user\\Desktop\\train.csv"
test_file_path = "C:\\Users\\user\\Desktop\\test.csv"

train_df = pd.read_csv(train_file_path, sep=';', header=0)
test_df = pd.read_csv(test_file_path, sep=';', header=0)

# 2. 檢查缺失值和基本資訊
def data_summary(df, dataset_name):
    print(f"=== {dataset_name} Summary ===")
    print("\nMissing Values Check:")
    print(df.isnull().sum())  # 缺失值數量
    print("\nDataset Info:")
    print(df.info())  # 資料集基本資訊
    print("\nDescriptive Statistics:")
    print(df.describe(include='all'))  # 描述性統計資訊
    print("\n================================\n")

# 輸出訓練集和測試集的摘要
data_summary(train_df, "Train Data")
data_summary(test_df, "Test Data")

# 3. 年齡區間分組
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 100]
labels = [f"{bins[i]}~{bins[i+1]}" for i in range(len(bins) - 1)]
train_df['age_group'] = pd.cut(train_df['age'], bins=bins, labels=labels, right=False)

# 4. 繪製水平分組長條圖
def plot_horizontal_bars(df, feature, target='y'):
    unique_values = df[target].unique()
    
    if isinstance(unique_values[0], str):
        count = df.groupby([feature, target]).size().unstack(fill_value=0)
    else:
        count = df.groupby([feature, target]).size().unstack(fill_value=0)

    bar_height = 0.4
    y = range(len(count))

    plt.figure(figsize=(10, 6))
    
    class_names = count.columns
    plt.barh(y, count[class_names[0]], height=bar_height, color='skyblue', label=f'{class_names[0]} (Did not subscribe)')
    plt.barh([i + bar_height for i in y], count[class_names[1]], height=bar_height, color='salmon', label=f'{class_names[1]} (Subscribed)')

    for i, (no, yes) in enumerate(zip(count[class_names[0]], count[class_names[1]])):
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

# 繪製年齡區間的長條圖
plot_horizontal_bars(train_df, "age_group")

# 分析其他特徵
features_to_plot = ["job", "marital", "education", "loan"]
for feature in features_to_plot:
    plot_horizontal_bars(train_df, feature)

# 5. 資料預處理
train_df['y'].replace(('yes', 'no'), (1, 0), inplace=True)
train_df['loan'].replace(('yes', 'no'), (1, 0), inplace=True)

test_df['y'].replace(('yes', 'no'), (1, 0), inplace=True)
test_df['loan'].replace(('yes', 'no'), (1, 0), inplace=True)

X_train = train_df[['age', 'balance', 'loan']]
y_train = train_df['y']

X_test = test_df[['age', 'balance', 'loan']]
y_test = test_df['y']

# 6. 訓練羅吉斯回歸模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 7. 預測測試資料
y_pred = model.predict(X_test)

# 8. 計算模型準確度
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Model Accuracy: {accuracy:.2f}")
