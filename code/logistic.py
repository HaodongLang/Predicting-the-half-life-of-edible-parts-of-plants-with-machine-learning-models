# -*- coding = utf-8 -*-
# @Time :2025/6/12 13:52
# @Author :郎皓东
# @File ：logistic.py
# @Software:PyCharm
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.metrics import matthews_corrcoef

# 数据预处理
def preprocess_data(df):
    # 物理化学特征标准化
    scaler = StandardScaler()
    physchem_cols = ['logKow', 'MW(g/mol)', 'Polar Surface Area (A2)', 'temperature']
    df[physchem_cols] = scaler.fit_transform(df[physchem_cols])

    # 处理分类特征
    # Chiral Center Atom 编码
    chiral_categories = ['C', 'P', 'S', '-']
    df['chiral_encoded'] = df['chiral center atom'].apply(
        lambda x: [1 if x == cat else 0 for cat in chiral_categories]
    )
    # Compartment独热编码
    compartment_categories = ['fruit', 'fruit surface', "curd", "flesh", "grain"]  # 根据实际数据补充所有类别
    df['compartment_encoded'] = df['compartment'].apply(
        lambda x: [1 if x == cat else 0 for cat in compartment_categories]
    )
    # Plant Class独热编码
    plant_class_categories = ['fruits', 'vegetables', 'root crops', 'cereals', 'leafy crops']  # 根据实际数据补充
    df['plant_class_encoded'] = df['plant class'].apply(
        lambda x: [1 if x == cat else 0 for cat in plant_class_categories]
    )
    # Stereochemical Configuration 编码
    def encode_stereo(config):
        if config == 'complicated':
            return [0, 0, 0, 1, 0, 0]
        elif config == 'mixed':
            return [0, 0, 0, 0, 1, 0]
        elif config == '-':
            return [0, 0, 0, 0, 0, 1]
        elif config == 'R':
            return [0, 0, 0, 0, 0, 0]
        elif config == "RR":
            return [0, 0, 1, 0, 0, 0]
        elif config == "RRR":
            return [0, 1, 0, 0, 0, 0]
        elif config == "RRS":
            return [0, 1, 1, 0, 0, 0]
        elif config == "RS":
            return [1, 0, 0, 0, 0, 0]
        elif config == "RSS":
            return [1, 0, 1, 0, 0, 0]
        elif config == "S":
            return [1, 1, 0, 0, 0, 0]

    df['stereo_encoded'] = df['Stereochemical configuration'].apply(encode_stereo)

    return df




# 特征提取函数
def extract_features(df):
    """
    提取所有特征并组合成特征矩阵
    """
    features = []

    for idx, row in df.iterrows():
        # 提取分子特征（使用Morgan指纹代替GNN）
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol:
            # 使用Morgan指纹（1024位）作为分子特征
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            mol_features = np.array(fp)
        else:
            mol_features = np.zeros(1024)

        # 物理化学特征（已标准化）
        physchem = [
            row['logKow'],
            row['MW(g/mol)'],
            row['Polar Surface Area (A2)'],
            row['temperature']
        ]

        # 分类特征（已编码）
        categorical = np.concatenate([
            row['chiral_encoded'],
            row['compartment_encoded'],
            row['plant_class_encoded'],
            row['stereo_encoded']
        ])

        # 合并所有特征
        all_features = np.concatenate([physchem, categorical])
        features.append(all_features)

    return np.array(features)


def train_and_evaluate(data_path='pesticide_data.csv', test_size=0.2, random_state=42):
    """
    训练和评估逻辑回归模型
    参数:
        data_path: 数据文件路径
        test_size: 测试集比例
        random_state: 随机种子
    返回:
        dict: 包含评估指标和特征重要性的字典
    """
    # 读取和处理数据
    df = pd.read_csv(data_path)
    df = preprocess_data(df)

    # 创建标签
    y = np.where(df['half life[day]'] <= 4, 0, 1)

    # 提取特征
    X = extract_features(df)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 训练模型
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # 预测和评估
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 计算指标
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUROC': roc_auc_score(y_test, y_proba),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }

    # 特征重要性
    importance = model.coef_[0]
    sorted_idx = np.argsort(np.abs(importance))[::-1]

    return metrics
# 主流程
if __name__ == "__main__":
    # 读取数据
    df = pd.read_csv('data/pesticide_data.csv')
    df = preprocess_data(df)

    # 创建标签（二分类）
    y = np.where(df['half life[day]'] <= 4, 0, 1)

    # 提取特征
    X = extract_features(df)

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 特征标准化（物理化学特征已标准化，但分子指纹需要额外处理）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 创建并训练逻辑回归模型
    model = LogisticRegression(
        penalty='l2',  # L2正则化
        C=1.0,  # 正则化强度
        solver='lbfgs',  # 优化算法
        max_iter=1000,  # 最大迭代次数
        class_weight='balanced',  # 平衡类别权重
        random_state=42
    )

    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # 正类的概率

    # 评估
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_proba)
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"MCC: {mcc:.4f}")

    # 特征重要性分析
    importance = model.coef_[0]
    print("\nFeature Importance (Top 20):")
    sorted_idx = np.argsort(np.abs(importance))[::-1]
    for idx in sorted_idx[:20]:
        print(f"Feature {idx}: {importance[idx]:.4f}")


