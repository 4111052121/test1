import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. 用pandas匯入資料，並檢查是否有缺失值
url = 'https://github.com/scikit-learn/scikit-learn/raw/main/sklearn/datasets/data/boston_house_prices.csv'
data = pd.read_csv(url, header=1)  # 指定第二行作為標題行

# 檢查缺失值
print("缺失值檢查：")
print(data.isnull().sum())
  
# 檢查數據前幾行和列名
print(data.head())
print(data.columns)

# 清理列名
data.columns = data.columns.str.strip()

# 2. 列出最高房價、最低房價、平均房價、中位數房價
max_price = data['MEDV'].max()
min_price = data['MEDV'].min()
mean_price = data['MEDV'].mean()
median_price = data['MEDV'].median()

print(f"最高房價: {max_price}")
print(f"最低房價: {min_price}")
print(f"平均房價: {mean_price}")
print(f"中位數房價: {median_price}")

# 3. 計算每個區間的房價計數，並繪製條形圖
price_bins = range(0, 60, 10)  # 定義區間
data['Price_Range'] = pd.cut(data['MEDV'], bins=price_bins)

# 計算每個區間的計數
price_distribution = data['Price_Range'].value_counts().sort_index()

# 繪製條形圖
plt.figure(figsize=(10, 6))
plt.bar(["0~10", "10~20", "20~30", "30~40", "40~50"], price_distribution.values, color='skyblue')
plt.title('Distributions of House Prices')
plt.xlabel('House Price Ranges (thousand dollars)')
plt.ylabel('Count')
plt.xticks(rotation=0)
#plt.grid(axis='y')
plt.show()

# 4. RM為每間住宅的平均房間數，將RM的值四捨五入到個位數，並分析不同RM值的平均房價
data['RM_rounded'] = data['RM'].round()
average_price_by_rm = data.groupby('RM_rounded')['MEDV'].mean()

# 繪製RM與平均房價的直方圖
average_price_by_rm.plot(kind='bar', figsize=(10, 6), color='skyblue')
plt.title('Distribution of Boston Housing Prices Group by RM')
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.xticks(rotation=0)
#plt.grid(axis='y')
plt.show()

# 5. 使用線性回歸模型來預測房價
X = data[['RM']]  # 特徵
y = data['MEDV']  # 目標變數

# 創建線性回歸模型
model = LinearRegression()
model.fit(X, y)

# 預測
predictions = model.predict(X)

# 繪製預測結果
plt.figure(figsize=(10, 6))
plt.scatter(data['RM'], y, color='skyblue', label='Actual House Price')
plt.plot(data['RM'], predictions, linewidth=2, label='Predicted House Price')
plt.title('Linear Regression of Room Number and House Price')
plt.xlabel('RM')
plt.ylabel('House Price (thousand dollars)')
plt.legend()
plt.show()
