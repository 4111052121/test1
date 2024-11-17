import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: 匯入訓練資料與測試資料，並檢查缺失值
train_file_path = "train.csv"  # 訓練資料檔案路徑
test_file_path = "test.csv"  # 測試資料檔案路徑

# 載入資料
train_data = pd.read_csv(train_file_path, sep=';')  # 使用分號分隔符號
test_data = pd.read_csv(test_file_path, sep=';')  # 使用分號分隔符號

# 清理欄位名稱
train_data.columns = [col.strip().replace('"', '') for col in train_data.columns]
test_data.columns = [col.strip().replace('"', '') for col in test_data.columns]

# 檢查是否有缺失值
print("訓練資料缺失值檢查：")
print(train_data.isnull().sum())
print("測試資料缺失值檢查：")
print(test_data.isnull().sum())

# Step 2: 年齡：分段處理為範圍
train_data['age_group'] = pd.cut(train_data['age'], bins=[0, 20, 40, 60, 80, 100], labels=['0~20', '20~40', '40~60', '60~80', '80~100'])
test_data['age_group'] = pd.cut(test_data['age'], bins=[0, 20, 40, 60, 80, 100], labels=['0~20', '20~40', '40~60', '60~80', '80~100'])

# 繪圖函數：分開顯示未訂閱與已訂閱，並顯示數值
def plot_separate_horizontal_bar_chart(data, feature, title):
    plt.figure(figsize=(10, 6))
    # 計算各分類中是否訂閱的數量
    plot_data = data.groupby([feature, 'y']).size().unstack(fill_value=0)
    
    # 建立位置參數
    y_positions = range(len(plot_data.index))
    bar_width = 0.4
    
    # 繪製「未訂閱」長條
    bar_no = plt.barh([pos - bar_width / 2 for pos in y_positions], plot_data['no'], 
                      height=bar_width, color='skyblue', label='No')
    # 繪製「已訂閱」長條
    bar_yes = plt.barh([pos + bar_width / 2 for pos in y_positions], plot_data['yes'], 
                       height=bar_width, color='salmon', label='Yes')
    
    # 加入數值標籤
    for bar in bar_no:
        plt.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2, 
                 f'{int(bar.get_width())}', va='center', fontsize=10)
    for bar in bar_yes:
        plt.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2, 
                 f'{int(bar.get_width())}', va='center', fontsize=10)
    
    # 加上標籤與圖例
    plt.yticks(y_positions, plot_data.index)
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel(feature.capitalize())
    plt.legend(title="y", labels=["No", "Yes"])
    plt.tight_layout()
    plt.show()

# Step 3: 繪製分開的水平長條圖
# 1. 年齡層 vs 訂閱定期存款 (針對 test.csv)
plot_separate_horizontal_bar_chart(train_data, 'age_group', 'Count of y by Age Group')

# 2. 工作 vs 訂閱定期存款 (針對 test.csv)
plot_separate_horizontal_bar_chart(train_data, 'job', 'Count of y by Job')

# 3. 婚姻狀況 vs 訂閱定期存款 (針對 test.csv)
plot_separate_horizontal_bar_chart(train_data, 'marital', 'Count of y by Marital Status')

# 4. 教育層次 vs 訂閱定期存款 (針對 test.csv)
plot_separate_horizontal_bar_chart(train_data, 'education', 'Count of y by Education')

# 5. 是否有個人貸款 vs 訂閱定期存款 (針對 test.csv)
plot_separate_horizontal_bar_chart(train_data, 'loan', 'Count of y by Personal Loan')

# Step 4: 轉換欄位為數值型別並進行訓練
# 轉換 'loan' 和 'y' 為數值型別
train_data['loan'] = train_data['loan'].apply(lambda x: 1 if x == 'yes' else 0)
train_data['y'] = train_data['y'].apply(lambda x: 1 if x == 'yes' else 0)
test_data['loan'] = test_data['loan'].apply(lambda x: 1 if x == 'yes' else 0)
test_data['y'] = test_data['y'].apply(lambda x: 1 if x == 'yes' else 0)

# 使用 'age', 'balance', 'loan' 作為特徵
X_train = train_data[['age', 'balance', 'loan']]
y_train = train_data['y']
X_test = test_data[['age', 'balance', 'loan']]
y_test = test_data['y']

# 建立羅吉斯回歸模型並訓練
model = LogisticRegression()
model.fit(X_train, y_train)

# 預測測試集並輸出準確度
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"羅吉斯回歸模型測試準確度: {accuracy:.2f}")




