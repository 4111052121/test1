import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 讀取資料
file_path = r'C:\Users\user\Desktop\boston_house_prices.csv'
data = pd.read_csv(file_path, skiprows=1)  # 跳過第一行

# 設定房價的欄位名稱
price_column = 'MEDV'

# 檢查缺失值
print("缺失值檢查：")
print(data.isnull().sum())

# 顯示房價的基本統計數據
print("\n基本統計數據：")
print("最高房價:", data[price_column].max())
print("最低房價:", data[price_column].min())
print("平均房價:", data[price_column].mean())
print("中位數房價:", data[price_column].median())

# 1. 房價分布直方圖（區間為 0~10, 10~20, 20~30, 30~40, 40~50）
bins = [0, 10, 20, 30, 40, 50]
labels = ['0~10', '10~20', '20~30', '30~40', '40~50']
data['price_range'] = pd.cut(data[price_column], bins=bins, labels=labels, right=False)

# 計算每個區間的數量
price_counts = data['price_range'].value_counts(sort=False)

# 繪製房價分布直方圖（不連續的長條圖）
plt.figure(figsize=(8, 6))
plt.bar(price_counts.index, price_counts.values, color='skyblue', edgecolor='black', width=0.6)
plt.title('Distribution of House Price')
plt.xlabel('House Price Range (thousand dollars)')
plt.ylabel('Count')
plt.show()

# 2. 不同 RM 值的平均房價
# RM值四捨五入到個位數
data['RM_rounded'] = data['RM'].round()

# 分析不同RM值的平均房價
avg_price_by_rm = data.groupby('RM_rounded')[price_column].mean()
print("\n不同RM值的平均房價：")
print(avg_price_by_rm)

# 繪製不同RM值的平均房價直方圖
plt.figure(figsize=(8, 6))
avg_price_by_rm.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Boston Housing Prices Group by RM')
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.xticks(rotation=0)
plt.show()

# 3. 房價預測與實際房價比較圖
# 訓練線性回歸模型
feature_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

# 確認特徵欄位是否存在
for col in feature_columns:
    if col not in data.columns:
        print(f"錯誤：找不到特徵欄位 '{col}'，請確認 CSV 文件中的欄位名稱。")
        exit()

X = data[feature_columns]
y = data[price_column]

# 分割資料集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立線性回歸模型
model = LinearRegression()
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 繪製房價預測與實際房價比較圖
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='black')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('房價預測與實際房價比較')
plt.xlabel('實際房價 (千美元)')
plt.ylabel('預測房價 (千美元)')
plt.grid(True)
plt.show()
