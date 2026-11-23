import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Target-80 版本
# 目標：
# 1. 讓 5-Fold 其中一折更有機會超過 80%
# 2. 降低 angry / sad 混淆
# 3. 特別觀察 sad -> angry、angry -> sad 比例
# ============================================================

# ── 中文字體設定 ──────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 路徑設定：依你的資料夾位置 ───────────────────────
TRAIN_ROOT = r"C:\Users\Edward\Desktop\train"
TEST_ROOT  = r"C:\Users\Edward\Desktop\test"

# ── 基本參數 ────────────────────────────────────────
BATCH_SIZE   = 32
FOLDS_EPOCHS = 70
EARLY_STOPPING_PATIENCE = 15
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)

print(f"系統啟動中... 正在使用硬體: {DEVICE}")


# ============================================================
# 影像前處理
# 重點：
# 原本 64x64 + 強增強，容易破壞 angry/sad 的眉眼嘴角細節
# 改成 96x96 + 較輕的 augmentation
# ============================================================
IMG_SIZE = 96

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),

    # 先放大再裁切，保留更多臉部細節
    transforms.Resize((112, 112)),
    transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    # angry/sad 對細節敏感，不建議旋轉太大
    transforms.RandomRotation(8),

    # translate 從 0.1 降到 0.04
    transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),

    # 亮度/對比也降一點
    transforms.ColorJitter(brightness=0.12, contrast=0.12),

    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

val_test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


# ============================================================
# Dataset
# ============================================================
train_dataset_full = datasets.ImageFolder(root=TRAIN_ROOT, transform=train_transform)
targets = train_dataset_full.targets
class_names = train_dataset_full.classes

test_dataset = datasets.ImageFolder(root=TEST_ROOT, transform=val_test_transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"成功載入！標籤映射為: {train_dataset_full.class_to_idx}")
print(f"訓練集數量: {len(train_dataset_full)} | 測試集數量: {len(test_dataset)}")


# ============================================================
# 模型架構
# 保留你原本的 EarlyClassroomCNN
# 因為有 AdaptiveAvgPool2d，所以輸入從 64x64 改成 96x96 不需要改 Linear
# ============================================================
class EarlyClassroomCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(EarlyClassroomCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ============================================================
# Loss 權重
# 你目前最大問題：angry -> sad 偏高
# 所以 angry 權重稍微提高，sad 也略提高，但 sad 不要拉太高
# ============================================================
def build_class_weights(class_names):
    weight_map = {
        "angry": 1.30,
        "happy": 1.00,
        "sad": 1.15,
        "surprise": 1.00,
    }

    weights = [weight_map.get(cls, 1.0) for cls in class_names]
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


class_weights = build_class_weights(class_names)
print(f"使用 class weights: {dict(zip(class_names, class_weights.detach().cpu().numpy().round(3)))}")


def make_criterion():
    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
    )


# ============================================================
# 工具：輸出 angry/sad 混淆比例
# ============================================================
def print_angry_sad_error(cm, title):
    if "angry" not in class_names or "sad" not in class_names:
        print("找不到 angry 或 sad 類別，無法計算 angry/sad 混淆比例。")
        return

    angry_idx = class_names.index("angry")
    sad_idx = class_names.index("sad")

    angry_total = cm[angry_idx].sum()
    sad_total = cm[sad_idx].sum()

    angry_to_sad = cm[angry_idx, sad_idx] / angry_total if angry_total > 0 else 0
    sad_to_angry = cm[sad_idx, angry_idx] / sad_total if sad_total > 0 else 0

    angry_recall = cm[angry_idx, angry_idx] / angry_total if angry_total > 0 else 0
    sad_recall = cm[sad_idx, sad_idx] / sad_total if sad_total > 0 else 0

    print(f"\n[{title}] angry / sad 混淆分析")
    print(f"angry recall: {angry_recall * 100:.2f}%")
    print(f"sad recall: {sad_recall * 100:.2f}%")
    print(f"angry 被判成 sad 比例: {angry_to_sad * 100:.2f}%")
    print(f"sad 被判成 angry 比例: {sad_to_angry * 100:.2f}%\n")


# ============================================================
# 5-Fold Training
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

fold_accuracies = []
best_model_path = ""
overall_best_acc = 0.0

all_val_preds = []
all_val_labels = []

best_fold_history = None
best_fold_number = 0

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):

    print(f"\n{'=' * 50}")
    print(f"正在執行第 {fold + 1} 折 (Fold {fold + 1}/5) 訓練")
    print("=" * 50)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0

    # Train 使用 augmentation
    fold_train_dataset = Subset(train_dataset_full, train_idx)

    # Val 不使用 augmentation
    val_dataset_clean = datasets.ImageFolder(
        root=TRAIN_ROOT,
        transform=val_test_transform
    )
    fold_val_dataset = Subset(val_dataset_clean, val_idx)

    fold_train_loader = DataLoader(
        fold_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    fold_val_loader = DataLoader(
        fold_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = EarlyClassroomCNN(num_classes=len(class_names)).to(DEVICE)

    criterion = make_criterion()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0003,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    current_model_path = f"best_model_fold{fold + 1}.pt"

    fold_val_preds = []
    fold_val_labels = []

    for epoch in range(FOLDS_EPOCHS):

        # ---------------- Train ----------------
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for imgs, lbls in fold_train_loader:
            imgs = imgs.to(DEVICE)
            lbls = lbls.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, lbls)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

            _, preds = torch.max(outputs, 1)

            train_total += lbls.size(0)
            train_correct += (preds == lbls).sum().item()

        epoch_train_loss = train_loss / len(fold_train_loader.dataset)
        epoch_train_acc = train_correct / train_total * 100

        # ---------------- Val ----------------
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        epoch_preds = []
        epoch_labels = []

        with torch.no_grad():
            for imgs, lbls in fold_val_loader:
                imgs = imgs.to(DEVICE)
                lbls = lbls.to(DEVICE)

                outputs = model(imgs)
                loss = criterion(outputs, lbls)

                val_loss += loss.item() * imgs.size(0)

                _, preds = torch.max(outputs, 1)

                val_total += lbls.size(0)
                val_correct += (preds == lbls).sum().item()

                epoch_preds.extend(preds.cpu().numpy())
                epoch_labels.extend(lbls.cpu().numpy())

        epoch_val_loss = val_loss / len(fold_val_loader.dataset)
        epoch_val_acc = val_correct / val_total * 100

        scheduler.step(epoch_val_loss)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1:02d}/{FOLDS_EPOCHS}] | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Train Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_acc:.2f}%"
        )

        # 以 Val Acc 保存最佳模型
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            fold_val_preds = epoch_preds.copy()
            fold_val_labels = epoch_labels.copy()
            torch.save(model.state_dict(), current_model_path)

        # Early stopping 用 Val Loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early Stopping 觸發！提早結束本折訓練。")
                break

    fold_accuracies.append(best_val_acc)
    all_val_preds.extend(fold_val_preds)
    all_val_labels.extend(fold_val_labels)

    print(f"Fold {fold + 1} 最佳驗證準確率: {best_val_acc:.2f}%")

    # 記錄整體最佳 fold，用來畫 Loss / Accuracy 曲線
    if best_val_acc > overall_best_acc:
        overall_best_acc = best_val_acc
        best_model_path = current_model_path
        best_fold_history = {k: v.copy() for k, v in history.items()}
        best_fold_number = fold + 1


# ============================================================
# 圖 1：最佳折 Loss 收斂曲線
# ============================================================
epochs_range = range(1, len(best_fold_history["train_loss"]) + 1)

plt.figure(figsize=(12, 4))
plt.plot(epochs_range, best_fold_history["train_loss"], "b-", label="訓練 Loss")
plt.plot(epochs_range, best_fold_history["val_loss"], "r--", label="驗證 Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"第 {best_fold_number} 折 Loss 收斂曲線")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("01_Loss收斂曲線.png", dpi=150)
plt.close()
print("✅ 已儲存：01_Loss收斂曲線.png")


# ============================================================
# 圖 2：最佳折 Accuracy 曲線
# ============================================================
plt.figure(figsize=(12, 4))
plt.plot(epochs_range, best_fold_history["train_acc"], "b-", label="訓練準確率")
plt.plot(epochs_range, best_fold_history["val_acc"], "r--", label="驗證準確率")
plt.xlabel("Epoch")
plt.ylabel("準確率 (%)")
plt.title(f"第 {best_fold_number} 折 準確率曲線")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("02_準確率曲線.png", dpi=150)
plt.close()
print("✅ 已儲存：02_準確率曲線.png")


# ============================================================
# 圖 3：五折合併總混淆矩陣
# ============================================================
cm_val = confusion_matrix(all_val_labels, all_val_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_val,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("五折合併總混淆矩陣（驗證集）")
plt.ylabel("真實標籤")
plt.xlabel("預測標籤")
plt.tight_layout()
plt.savefig("03_五折合併總混淆矩陣_驗證集.png", dpi=150)
plt.close()
print("✅ 已儲存：03_五折合併總混淆矩陣_驗證集.png")

print_angry_sad_error(cm_val, "五折合併驗證集")


# ============================================================
# 圖 4：5-Fold 最佳驗證準確率
# ============================================================
mean_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)

plt.figure(figsize=(10, 6))
bars = plt.bar(
    [f"Fold {i + 1}" for i in range(5)],
    fold_accuracies,
    color="steelblue",
    alpha=0.85
)

for bar, acc in zip(bars, fold_accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f"{acc:.2f}%",
        ha="center",
        va="bottom",
        fontsize=11
    )

plt.axhline(
    y=mean_acc,
    color="royalblue",
    linestyle="--",
    linewidth=1.5,
    label=f"平均: {mean_acc:.2f}%"
)

plt.xlabel("折次")
plt.ylabel("最佳驗證準確率 (%)")
plt.title(f"5-Fold 交叉驗證最佳驗證準確率\n平均: {mean_acc:.2f}% ± {std_acc:.2f}%")
plt.ylim(0, 100)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("04_5Fold交叉驗證最佳驗證準確率.png", dpi=150)
plt.close()
print("✅ 已儲存：04_5Fold交叉驗證最佳驗證準確率.png")


# ============================================================
# 最終 Test Set：載入 5-Fold 中最佳權重
# ============================================================
print(f"\n{'🚀' * 20}")
print(f"啟動最終 Test Set 盲測！載入最強權重: {best_model_path}")
print(f"{'🚀' * 20}")

final_model = EarlyClassroomCNN(num_classes=len(class_names)).to(DEVICE)
final_model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
final_model.eval()

test_correct = 0
test_total = 0
test_preds = []
test_labels = []

with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs = imgs.to(DEVICE)
        lbls = lbls.to(DEVICE)

        outputs = final_model(imgs)

        _, preds = torch.max(outputs, 1)

        test_total += lbls.size(0)
        test_correct += (preds == lbls).sum().item()

        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(lbls.cpu().numpy())

test_acc = test_correct / test_total * 100

print(f"\n獨立測試集 (Test Set) 最終準確率: {test_acc:.2f}%\n")


# ============================================================
# 圖 5：最終測試集混淆矩陣
# ============================================================
cm_test = confusion_matrix(test_labels, test_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_test,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("最終測試集混淆矩陣（Test Set）")
plt.ylabel("真實標籤")
plt.xlabel("預測標籤")
plt.tight_layout()
plt.savefig("05_最終測試集混淆矩陣_TestSet.png", dpi=150)
plt.close()
print("✅ 已儲存：05_最終測試集混淆矩陣_TestSet.png")

print_angry_sad_error(cm_test, "最終測試集")


# ============================================================
# 圖 6：各類別訓練樣本數 vs Test F1-Score
# ============================================================
f1_per_class = f1_score(test_labels, test_preds, average=None)

# 計算各類別訓練樣本數
class_counts = []
for i in range(len(class_names)):
    count = sum(1 for t in targets if t == i)
    class_counts.append(count)

fig, ax1 = plt.subplots(figsize=(10, 6))

bars = ax1.bar(
    class_names,
    class_counts,
    color="steelblue",
    alpha=0.6,
    label="訓練樣本數"
)

for bar, cnt in zip(bars, class_counts):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 10,
        str(cnt),
        ha="center",
        va="bottom",
        fontsize=11
    )

ax1.set_xlabel("情緒類別")
ax1.set_ylabel("訓練樣本數")
ax1.set_ylim(0, max(class_counts) * 1.2)

ax2 = ax1.twinx()
ax2.plot(
    class_names,
    f1_per_class,
    "ko-",
    linewidth=2,
    markersize=8,
    label="Test F1-Score"
)

for i, (cls, f1) in enumerate(zip(class_names, f1_per_class)):
    ax2.text(
        i,
        f1 + 0.02,
        f"{f1:.2f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

ax2.set_ylabel("F1-Score")
ax2.set_ylim(0, 1.1)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.title("平衡後各類別訓練樣本數 vs 測試集 F1-Score")
fig.tight_layout()
plt.savefig("06_各類別訓練樣本數_vs_TestF1Score.png", dpi=150)
plt.close()
print("✅ 已儲存：06_各類別訓練樣本數_vs_TestF1Score.png")


# ============================================================
# 最終結果輸出
# ============================================================
print("\n專題期末報告：5-Fold 交叉驗證與最終測試結果")

for i, acc in enumerate(fold_accuracies):
    print(f"   Fold {i + 1} 最佳驗證準確率 (Val): {acc:.2f}%")

print(f"交叉驗證平均準確率: {np.mean(fold_accuracies):.2f} ± {np.std(fold_accuracies):.2f}%")
print(f"系統實戰盲測準確率 (Test): {test_acc:.2f}%")

print("\n各類別 Test F1-Score:")
for cls, f1 in zip(class_names, f1_per_class):
    print(f"   {cls}: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(test_labels, test_preds, target_names=class_names, digits=4))
