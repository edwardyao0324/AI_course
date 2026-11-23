import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
from collections import Counter
import time

TRAIN_ROOT = r"C:\Users\user\Desktop\0\train"
TEST_ROOT  = r"C:\Users\user\Desktop\0\test"
BATCH_SIZE   = 32
FOLDS_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_num_threads(4)
print(f"系統啟動中... 正在使用硬體: {DEVICE}")

try:
    font_path = "C:/Windows/Fonts/msjh.ttc"
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Microsoft JhengHei'
except:
    pass
plt.rcParams['axes.unicode_minus'] = False

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

val_test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset_full_raw = datasets.ImageFolder(root=TRAIN_ROOT, transform=train_transform)

class_indices = {0: [], 1: [], 2: [], 3: []}
for idx, target in enumerate(train_dataset_full_raw.targets):
    class_indices[target].append(idx)

balanced_indices = []
random.seed(42)
for cls, indices in class_indices.items():
    sampled_indices = random.sample(indices, 3000)
    balanced_indices.extend(sampled_indices)

train_dataset_full = Subset(train_dataset_full_raw, balanced_indices)
targets     = [train_dataset_full_raw.targets[i] for i in balanced_indices]
class_names = train_dataset_full_raw.classes

test_dataset = datasets.ImageFolder(root=TEST_ROOT, transform=val_test_transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"成功載入！標籤映射為: {train_dataset_full_raw.class_to_idx}")
print(f"平衡後訓練集數量: {len(train_dataset_full)} | 測試集數量: {len(test_dataset)}")

print("\n繪製資料集類別分佈圖...")
train_counts = Counter(targets)
test_counts  = Counter(test_dataset.targets)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = ['#EF4444', '#22C55E', '#3B82F6', '#F59E0B']

for ax, counts, title in zip(axes,
                              [train_counts, test_counts],
                              ['訓練集類別分佈', '測試集類別分佈']):
    bars = ax.bar([class_names[i] for i in sorted(counts)],
                  [counts[i] for i in sorted(counts)],
                  color=colors, edgecolor='white', linewidth=1.2)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('情緒類別', fontsize=12)
    ax.set_ylabel('樣本數量', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 30,
                str(int(bar.get_height())),
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('dataset_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   dataset_distribution.png 已儲存")

class EarlyClassroomCNN(nn.Module):
    def __init__(self):
        super(EarlyClassroomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies  = []
best_model_path  = ""
overall_best_acc = 0.0

all_folds_preds  = []
all_folds_labels = []
all_val_accs     = []
fold_times       = []

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
    print(f"\n{'='*50}")
    print(f"正在執行第 {fold+1} 折 (Fold {fold+1}/5) 訓練")
    print('='*50)

    fold_start = time.time()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc  = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    EARLY_STOPPING_PATIENCE = 7

    fold_train_dataset = Subset(train_dataset_full, train_idx)
    
    val_dataset_clean_raw = datasets.ImageFolder(root=TRAIN_ROOT, transform=val_test_transform)
    val_dataset_clean     = Subset(val_dataset_clean_raw, balanced_indices) 
    fold_val_dataset      = Subset(val_dataset_clean, val_idx)

    fold_train_loader = DataLoader(fold_train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    fold_val_loader   = DataLoader(fold_val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = EarlyClassroomCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    for epoch in range(FOLDS_EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, lbls in fold_train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            _, preds    = torch.max(outputs, 1)
            train_total   += lbls.size(0)
            train_correct += (preds == lbls).sum().item()

        epoch_train_loss = train_loss / len(fold_train_loader.dataset)
        epoch_train_acc  = (train_correct / train_total) * 100

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, lbls in fold_val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                outputs    = model(imgs)
                loss       = criterion(outputs, lbls)
                val_loss  += loss.item() * imgs.size(0)
                _, preds   = torch.max(outputs, 1)
                val_total   += lbls.size(0)
                val_correct += (preds == lbls).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(lbls.cpu().numpy())

        epoch_val_loss = val_loss / len(fold_val_loader.dataset)
        epoch_val_acc  = (val_correct / val_total) * 100
        scheduler.step(epoch_val_loss)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch [{epoch+1:02d}/{FOLDS_EPOCHS}] | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            current_model_path = f'best_model_fold{fold+1}.pt'
            torch.save(model.state_dict(), current_model_path)
            if best_val_acc > overall_best_acc:
                overall_best_acc = best_val_acc
                best_model_path  = current_model_path

        if epoch_val_loss < best_val_loss:
            best_val_loss    = epoch_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early Stopping 觸發！提早結束第 {fold+1} 折。")
                break

    fold_elapsed = (time.time() - fold_start) / 60
    fold_times.append(fold_elapsed)
    print(f"Fold {fold+1} 訓練耗時: {fold_elapsed:.1f} 分鐘")

    all_folds_preds.extend(all_preds)
    all_folds_labels.extend(all_labels)
    all_val_accs.append(history['val_acc'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Fold {fold+1} 學習曲線', fontsize=14, fontweight='bold')

    ax1.plot(history['train_loss'], label='Train Loss', color='#2E75B6', linewidth=2)
    ax1.plot(history['val_loss'],   label='Val Loss',   color='#E74C3C', linewidth=2)
    ax1.set_title('Loss 曲線')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(history['train_acc'], label='Train Accuracy', color='#2E75B6', linewidth=2)
    ax2.plot(history['val_acc'],   label='Val Accuracy',   color='#E74C3C', linewidth=2)
    ax2.set_title('Accuracy 曲線')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'learning_curve_fold{fold+1}.png', dpi=150, bbox_inches='tight')
    plt.close()

    fold_accuracies.append(best_val_acc)
    print(f"Fold {fold+1} 最高驗證準確率: {best_val_acc:.2f}%")

print(f"\n{'='*20}")
print(f"啟動最終 Test Set 盲測！載入最強權重: {best_model_path}")

final_model = EarlyClassroomCNN().to(DEVICE)
final_model.load_state_dict(torch.load(best_model_path))
final_model.eval()

test_correct, test_total = 0, 0
test_preds, test_labels  = [], []
test_confidences_correct = []
test_confidences_wrong   = []

with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        outputs    = final_model(imgs)
        probs      = torch.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, 1)

        test_total   += lbls.size(0)
        test_correct += (preds == lbls).sum().item()
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(lbls.cpu().numpy())

        for c, p, l in zip(conf.cpu().numpy(),
                            preds.cpu().numpy(),
                            lbls.cpu().numpy()):
            if p == l:
                test_confidences_correct.append(c)
            else:
                test_confidences_wrong.append(c)

test_acc = (test_correct / test_total) * 100
print(f"獨立測試集最終準確率: {test_acc:.2f}%\n")
print(classification_report(test_labels, test_preds, target_names=class_names))

print("\n正在生成所有報告圖表...")

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar([f'Fold {i+1}' for i in range(5)],
              fold_accuracies,
              color=['#2E75B6', '#28A745', '#DC3545', '#FFC107', '#6F42C1'],
              edgecolor='white', linewidth=1.5, width=0.55)

mean_acc = np.mean(fold_accuracies)
ax.axhline(mean_acc, color='#E74C3C', linestyle='--', linewidth=2,
           label=f'平均: {mean_acc:.2f}%')

for bar, acc in zip(bars, fold_accuracies):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.2,
            f'{acc:.2f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_title('5-Fold 交叉驗證最佳驗證準確率', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('折次', fontsize=12)
ax.set_ylabel('最佳驗證準確率 (%)', fontsize=12)
ax.set_ylim(0, 100)
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('5fold_accuracy_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("   5fold_accuracy_bar.png 已儲存")

cm = confusion_matrix(test_labels, test_preds)
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 14, 'weight': 'bold'}, ax=ax)
ax.set_title(f'測試集混淆矩陣（準確率 {test_acc:.2f}%）',
             fontsize=14, fontweight='bold', pad=12)
ax.set_ylabel('真實標籤', fontsize=12)
ax.set_xlabel('預測標籤', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix_test_final.png', dpi=150, bbox_inches='tight')
plt.close()
print("   confusion_matrix_test_final.png 已儲存")

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax  = plt.subplots(figsize=(8, 7))
sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 13, 'weight': 'bold'}, ax=ax)
ax.set_title('測試集混淆矩陣（各類別百分比）',
             fontsize=14, fontweight='bold', pad=12)
ax.set_ylabel('真實標籤', fontsize=12)
ax.set_xlabel('預測標籤', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix_test_percent.png', dpi=150, bbox_inches='tight')
plt.close()
print("   confusion_matrix_test_percent.png 已儲存")

total_cm = confusion_matrix(all_folds_labels, all_folds_preds)
fig, ax  = plt.subplots(figsize=(8, 7))
sns.heatmap(total_cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 14, 'weight': 'bold'}, ax=ax)
ax.set_title('五折合併總混淆矩陣（驗證集）',
             fontsize=14, fontweight='bold', pad=12)
ax.set_ylabel('真實標籤', fontsize=12)
ax.set_xlabel('預測標籤', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix_ALL_FOLDS.png', dpi=150, bbox_inches='tight')
plt.close()
print("   confusion_matrix_ALL_FOLDS.png 已儲存")

report_dict = classification_report(
    test_labels, test_preds, target_names=class_names, output_dict=True
)

metrics    = ['precision', 'recall', 'f1-score']
metrics_zh = ['Precision', 'Recall', 'F1-Score']
x     = np.arange(len(class_names))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
colors_m = ['#2E75B6', '#28A745', '#E74C3C']

for i, (m, mz, c) in enumerate(zip(metrics, metrics_zh, colors_m)):
    vals = [report_dict[cls][m] for cls in class_names]
    bars = ax.bar(x + i * width, vals, width,
                  label=mz, color=c, edgecolor='white', linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{v:.2f}',
                ha='center', va='bottom', fontsize=9)

ax.set_title('各類別 Precision / Recall / F1-Score（測試集）',
             fontsize=14, fontweight='bold', pad=12)
ax.set_xticks(x + width)
ax.set_xticklabels(class_names, fontsize=12)
ax.set_ylabel('分數', fontsize=12)
ax.set_ylim(0, 1.12)
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('per_class_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("   per_class_metrics.png 已儲存")

fig, ax = plt.subplots(figsize=(11, 5))
fold_colors = ['#2E75B6', '#28A745', '#DC3545', '#FFC107', '#6F42C1']

for i, val_accs in enumerate(all_val_accs):
    ax.plot(val_accs, label=f'Fold {i+1}',
            color=fold_colors[i], linewidth=2, alpha=0.85)

ax.axhline(np.mean(fold_accuracies), color='black', linestyle='--',
           linewidth=1.5, label=f'平均最佳: {np.mean(fold_accuracies):.2f}%')
ax.set_title('五折驗證準確率收斂曲線', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Val Accuracy (%)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('val_acc_all_folds.png', dpi=150, bbox_inches='tight')
plt.close()
print("   val_acc_all_folds.png 已儲存")

train_sample_counts = [train_counts[i] for i in range(len(class_names))]
f1_scores = [report_dict[cls]['f1-score'] for cls in class_names]

fig, ax1 = plt.subplots(figsize=(10, 5))
bar_colors = ['#EF4444', '#22C55E', '#3B82F6', '#F59E0B']
bars = ax1.bar(class_names, train_sample_counts,
               color=bar_colors, alpha=0.6, edgecolor='white',
               linewidth=1.2, label='訓練樣本數')
ax1.set_ylabel('訓練樣本數量', fontsize=12)
ax1.set_xlabel('情緒類別', fontsize=12)

for bar, cnt in zip(bars, train_sample_counts):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 50,
             str(cnt),
             ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2 = ax1.twinx()
ax2.plot(class_names, f1_scores, 'ko-',
         linewidth=2.5, markersize=10, label='F1-Score', zorder=5)
ax2.set_ylabel('F1-Score', fontsize=12)
ax2.set_ylim(0, 1.1)

for i, (cls, f1) in enumerate(zip(class_names, f1_scores)):
    ax2.text(i, f1 + 0.03, f'{f1:.2f}',
             ha='center', fontsize=11, fontweight='bold', color='black')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='upper right', fontsize=11)

plt.title('各類別訓練樣本數 vs F1-Score', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('sample_vs_f1.png', dpi=150, bbox_inches='tight')
plt.close()
print("   sample_vs_f1.png 已儲存")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(test_confidences_correct, bins=30, alpha=0.65,
        color='#22C55E', label=f'預測正確（共 {len(test_confidences_correct)} 筆）',
        edgecolor='white')
ax.hist(test_confidences_wrong, bins=30, alpha=0.65,
        color='#EF4444', label=f'預測錯誤（共 {len(test_confidences_wrong)} 筆）',
        edgecolor='white')

ax.axvline(np.mean(test_confidences_correct), color='#16A34A',
           linestyle='--', linewidth=2,
           label=f'正確平均信心度: {np.mean(test_confidences_correct):.3f}')
ax.axvline(np.mean(test_confidences_wrong), color='#DC2626',
           linestyle='--', linewidth=2,
           label=f'錯誤平均信心度: {np.mean(test_confidences_wrong):.3f}')

ax.set_title('模型預測信心度分布（Softmax 最高分數）',
             fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('信心度', fontsize=12)
ax.set_ylabel('樣本數量', fontsize=12)
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('confidence_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   confidence_distribution.png 已儲存")

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar([f'Fold {i+1}' for i in range(5)],
              fold_times,
              color=['#2E75B6', '#28A745', '#DC3545', '#FFC107', '#6F42C1'],
              edgecolor='white', linewidth=1.5, width=0.55)

total_time = sum(fold_times)
ax.axhline(np.mean(fold_times), color='#E74C3C', linestyle='--',
           linewidth=2, label=f'平均: {np.mean(fold_times):.1f} 分鐘')

for bar, t in zip(bars, fold_times):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            f'{t:.1f} 分',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_title(f'各折訓練時間（總計 {total_time:.1f} 分鐘）',
             fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('折次', fontsize=12)
ax.set_ylabel('訓練時間（分鐘）', fontsize=12)
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('fold_training_time.png', dpi=150, bbox_inches='tight')
plt.close()
print("   fold_training_time.png 已儲存")

print(f"\n{'='*20}")
print("5-Fold 交叉驗證與最終測試結果（48×48 最佳版）")
for i, acc in enumerate(fold_accuracies):
    print(f"   Fold {i+1} 最佳驗證準確率: {acc:.2f}%  |  訓練時間: {fold_times[i]:.1f} 分鐘")
print(f"交叉驗證平均準確率: {np.mean(fold_accuracies):.2f} ± {np.std(fold_accuracies):.2f}%")
print(f"系統實戰盲測準確率 (Test): {test_acc:.2f}%")
print(f"總訓練時間: {sum(fold_times):.1f} 分鐘")
print(f"\n產生圖表清單：")
charts = [
    'dataset_distribution.png          → 資料集類別分佈',
    'learning_curve_fold1~5.png        → 各折學習曲線',
    '5fold_accuracy_bar.png            → 5-Fold 準確率長條圖',
    'confusion_matrix_test_final.png   → 測試集混淆矩陣（數字）',
    'confusion_matrix_test_percent.png → 測試集混淆矩陣（百分比）',
    'confusion_matrix_ALL_FOLDS.png    → 五折合併總混淆矩陣',
    'per_class_metrics.png             → 各類別 Precision/Recall/F1',
    'val_acc_all_folds.png             → 五折 Val Acc 合併曲線',
    'sample_vs_f1.png                  → 樣本數 vs F1-Score',
    'confidence_distribution.png       → 信心度分布圖',
    'fold_training_time.png            → 各折訓練時間',
]
for c in charts:
    print(f"   {c}")