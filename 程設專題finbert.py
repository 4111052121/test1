import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

# 設置設備（GPU優先）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用的設備:", device)

# 自定義數據集
class FinancialNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

if __name__ == '__main__':
    # 1. 資料讀取與預處理
    file_path = r"C:\Users\user\Desktop\程設專題\archive (3)\all-data.csv"
    data = pd.read_csv(file_path, names=["Sentiment", "News_Headline"], encoding='latin1')

    # 將標籤轉換為數字
    label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    data['Sentiment'] = data['Sentiment'].map(label_mapping)

    # 分割數據集
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data['News_Headline'], data['Sentiment'], test_size=0.2, random_state=42
    )

    # 2. 載入 BERT 模型與 Tokenizer
    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone", num_labels=3)
    model.to(device)  # 將模型移動到 GPU

    # 創建 DataLoader
    train_dataset = FinancialNewsDataset(train_texts, train_labels, tokenizer)
    test_dataset = FinancialNewsDataset(test_texts, test_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # 3. 訓練模型
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    for epoch in range(3):  # 訓練 3 個 Epoch
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)  # 將資料移動到 GPU
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    # 4. 評估模型
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)  # 將資料移動到 GPU
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # 5. 分類報告
    print("分類報告：")
    print(classification_report(true_labels, predictions, target_names=label_mapping.keys()))

    # 6. 繪製混淆矩陣
    conf_matrix = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # 7. 用戶界面輸入
    def predict_sentiment(text):
        model.eval()
        with torch.no_grad():
            encoding = tokenizer(
                text,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(F.softmax(logits, dim=1), dim=1).item()

        sentiment = {0: "negative", 1: "neutral", 2: "positive"}
        return sentiment[pred]

    while True:
        user_input = input("輸入一條金融新聞（或輸入 'exit' 離開）：")
        if user_input.lower() == "exit":
            print("程式結束。")
            break
        print(f"預測情緒：{predict_sentiment(user_input)}")
