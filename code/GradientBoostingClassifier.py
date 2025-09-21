# -*- coding = utf-8 -*-
# @Time :2025/6/12 14:17
# @Author :郎皓东
# @File ：GradientBoostingClassifier.py
# @Software:PyCharm
import pandas as pd
import numpy as np
import shap
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

# 设置显示选项以避免省略
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# 数据预处理 - 与原始代码相同
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
    compartment_categories = ['fruit', 'fruit surface', "curd", "flesh", "grain"]
    df['compartment_encoded'] = df['compartment'].apply(
        lambda x: [1 if x == cat else 0 for cat in compartment_categories]
    )
    # Plant Class独热编码
    plant_class_categories = ['fruits', 'vegetables', 'root crops', 'cereals', 'leafy crops']
    df['plant_class_encoded'] = df['plant class'].apply(
        lambda x: [1 if x == cat else 0 for cat in plant_class_categories]
    )

    # Stereochemical Configuration 编码
    def encode_stereo(config):
        if config == 'complicated':
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        elif config == 'mixed':
            return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif config == '-':
            return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif config == 'R':
            return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif config == "RR":
            return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif config == "RRR":
            return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif config == "RRS":
            return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif config == "RS":
            return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif config == "RSS":
            return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif config == "S":
            return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 默认值

    df['stereo_encoded'] = df['Stereochemical configuration'].apply(encode_stereo)

    return df


# 生成ECFP指纹
def generate_ecfp(smiles, radius=2, n_bits=1024):
    """生成ECFP4指纹（半径=2）"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits))


# 特征提取函数
def extract_features(df):
    """
    提取所有特征并组合成特征矩阵
    """
    # 生成ECFP指纹
    ecfp_features = np.vstack(df['SMILES'].apply(lambda x: generate_ecfp(x)))

    # 物理化学特征
    physchem_features = df[['logKow', 'MW(g/mol)', 'Polar Surface Area (A2)', 'temperature']].values

    # 分类特征
    def extract_encoded_features(row):
        return np.concatenate([
            row['chiral_encoded'],
            row['compartment_encoded'],
            row['plant_class_encoded'],
            row['stereo_encoded']
        ])

    categorical_features = np.vstack(df.apply(extract_encoded_features, axis=1))
    # 合并所有特征
    X = np.hstack([ecfp_features, physchem_features, categorical_features])

    # 创建特征名称（用于解释性分析）
    feature_names = (
            [f'ECFP_{i}' for i in range(ecfp_features.shape[1])] +
            ['logKow', 'MW', 'PSA', 'temperature'] +
            ['-', 'C', 'P', 'S'] +
            ['curd', 'flesh', 'fruit', 'fruit surface', 'grain'] +
            ['cereals', 'fruits', 'leafy crops', 'root crops', 'vegetables'] +
            ['-', 'complicated', 'mixed', 'S', 'RRS', 'R', 'RR', 'RRR', 'RS', 'RSS']
    )

    return X, feature_names


# 主流程
if __name__ == "__main__":
    # 读取数据
    df = pd.read_csv('data/pesticide_data.csv')
    df = preprocess_data(df)

    # 创建标签（二分类）
    y = np.where(df['half life[day]'] <= 4, 0, 1)

    # 提取特征
    X, feature_names = extract_features(df)

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建并训练GBRT模型
    model = GradientBoostingClassifier(
        n_estimators=300,  # 树的数量
        learning_rate=0.05,  # 学习率
        max_depth=5,  # 树的最大深度
        min_samples_split=10,  # 分裂内部节点所需的最小样本数
        min_samples_leaf=5,  # 叶节点所需的最小样本数
        subsample=0.8,  # 用于拟合个体基础学习器的样本比例
        max_features='sqrt',  # 寻找最佳分割时要考虑的特征数量
        random_state=42,
        verbose=1  # 显示训练进度
    )

    print("开始训练模型...")
    model.fit(X_train, y_train)
    print("模型训练完成")

    # 预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # 正类的概率

    # 评估
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_proba)
    mcc = matthews_corrcoef(y_test, y_pred)

    print("\n模型性能评估:")
    print(f"准确率 (Accuracy): {acc:.4f}")
    print(f"F1 分数: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"MCC: {mcc:.4f}")

    # 特征重要性分析
    print("\n特征重要性分析:")
    importance = model.feature_importances_
    sorted_idx = importance.argsort()[::-1]
    print(len(feature_names))

    # 打印最重要的20个特征
    print("\nTop 20 重要特征:")
    for i in range(20):
        if sorted_idx[i] < len(feature_names):
            print(f"{feature_names[sorted_idx[i]]}: {importance[sorted_idx[i]]:.4f}")

    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    plt.bar(range(20), importance[sorted_idx[:20]])
    plt.xticks(range(20), [feature_names[i] for i in sorted_idx[:20]], rotation=90)
    plt.title("Top 20 重要特征")
    plt.ylabel("重要性分数")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()

    # 学习曲线分析（使用OOB改进估计）
    oob_improvement = model.oob_improvement_
    if oob_improvement is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(oob_improvement)) + 1, oob_improvement)
        plt.xlabel("Boosting Iterations")
        plt.ylabel("OOB Improvement")
        plt.title("OOB Improvement by Boosting Iterations")
        plt.grid(True)
        plt.savefig("oob_improvement.png")
        plt.show()

    # 训练过程中的损失变化
    train_score = model.train_score_
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(train_score)) + 1, train_score)
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Training Loss")
    plt.title("Training Loss by Boosting Iterations")
    plt.grid(True)
    plt.savefig("training_loss.png")

    # ======================= SHAP分析 =======================

    # 1. 创建SHAP解释器 - 使用TreeExplainer
    explainer = shap.TreeExplainer(model)

    # 2. 计算SHAP值（使用测试集的一个子集以提高效率）
    sample_indices = np.random.choice(len(X_test), min(100, len(X_test)), replace=False)
    X_sample = X_test[sample_indices]
    y_sample = y_test[sample_indices]

    shap_values = explainer.shap_values(X_test)

    # 3. 全局特征重要性分析
    print("\n全局特征重要性分析:")
    # 平均绝对SHAP值作为特征重要性度量
    global_importance = np.mean(np.abs(shap_values), axis=0)

    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': global_importance
    })

    # 按重要性排序
    importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)

    # 打印最重要的20个特征
    print(importance_df.head(20))

    # 可视化全局特征重要性
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'][:20][::-1], importance_df['Importance'][:20][::-1])
    plt.xlabel('平均绝对SHAP值')
    plt.title('Top 20 重要特征 (基于SHAP值)')
    plt.tight_layout()
    plt.savefig('shap_global_importance.png')
    plt.show()

    print(f"SHAP值矩阵形状: {np.array(shap_values).shape}")  # 应为 (n_samples, n_features)
    print(f"特征矩阵形状: {X_sample.shape}")
    print(f"特征名数量: {len(feature_names)}")
    # 4. 总结图 - 展示特征影响分布
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values[:100], X_sample, feature_names=feature_names, plot_type="dot", max_display=20)
    plt.title('特征影响分布')
    plt.tight_layout()
    # plt.savefig('shap_summary_dot.png')
    plt.show()

    # 5. 特定样本解释 - 选择一个代表性样本
    sample_idx = np.argmax(np.abs(shap_values).sum(axis=1))  # 选择最具解释性的样本
    sample_data = X_sample[sample_idx]
    sample_label = y_sample[sample_idx]

    print(f"\n样本详细解释 (真实标签: {'>4天' if sample_label == 1 else '≤4天'})")

    # # 力导向图
    # shap.force_plot(
    #     explainer.expected_value,
    #     shap_values[sample_idx],
    #     sample_data,
    #     feature_names=feature_names,
    #     matplotlib=True,
    #     text_rotation=15
    # )
    # plt.title(f"样本预测解释 (预测概率: {model.predict_proba([sample_data])[0][1]:.2f})")
    # plt.tight_layout()
    # # plt.savefig('shap_force_plot.png')
    # plt.show()

    # 决策图
    plt.figure(figsize=(12, 8))
    shap.decision_plot(
        explainer.expected_value,
        shap_values[sample_idx],
        sample_data,
        feature_names=feature_names,
        feature_order='importance'
    )
    plt.title(f"样本决策路径 (预测概率: {model.predict_proba([sample_data])[0][1]:.2f})")
    plt.tight_layout()
    plt.savefig('shap_decision_plot.png')
    plt.show()

    # 6. 特征依赖分析 - 对最重要特征进行详细分析
    top_feature = importance_df.iloc[0]['Feature']

    print(f"\n对关键特征 '{top_feature}' 的依赖分析")
    print(len(shap_values))
    print(len(X_sample))


    # # 依赖图
    # sample_indices = np.random.choice(len(X_test), size=361, replace=False)
    # X_sample = X_test[sample_indices]
    # plt.figure(figsize=(12, 8))
    # shap.dependence_plot(
    #     top_feature,
    #     shap_values,
    #     X_sample,
    #     feature_names=feature_names,
    #     interaction_index=None
    # )
    # plt.title(f"特征 '{top_feature}' 依赖关系")
    # plt.tight_layout()
    # # plt.savefig('shap_dependence_plot.png')
    # plt.show()
