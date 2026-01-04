#!/usr/bin/env python3
# 模型训练脚本

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. 数据加载
print("正在加载数据...")
data = pd.read_csv('./student_data_adjusted_rounded.csv')
print(f"数据加载完成，共 {len(data)} 行，{len(data.columns)} 列")

# 2. 数据预处理
print("正在预处理数据...")
le_major = LabelEncoder()
le_gender = LabelEncoder()
data['专业'] = le_major.fit_transform(data['专业'])
data['性别'] = le_gender.fit_transform(data['性别'])

# 特征和目标变量
X = data[['性别', '专业', '每周学习时长（小时）', '上课出勤率', '期中考试分数', '作业完成率']]
y = data['期末考试分数']

# 训练测试集分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")

# 3. 算法选择和训练
algorithms = {
    "linear_regression": LinearRegression(),
    "ridge_regression": Ridge(),
    "lasso_regression": Lasso(),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

best_model = None
best_score = 0
results = []

print("\n正在训练模型...")
for name, model in algorithms.items():
    print(f"\n训练 {name}...")
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 模型评估
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 保存结果
    results.append({
        "algorithm": name,
        "mae": mae,
        "mse": mse,
        "r2": r2
    })
    
    print(f"{name} 性能:")
    print(f"  MAE: {mae:.2f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  R²: {r2:.4f}")
    
    # 选择最佳模型
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_model_name = name

# 4. 保存结果到CSV
results_df = pd.DataFrame(results)
results_df.to_csv('D:\streamlit_env\model_results.csv', index=False)
print(f"\n模型结果已保存到 D:\streamlit_env\model_results.csv")

# 5. 保存最佳模型和编码器
print(f"\n最佳模型: {best_model_name} (R²: {best_score:.4f})")
print("正在保存最佳模型和编码器...")

# 创建模型目录
import os
model_dir = 'D:\streamlit_env\models'
os.makedirs(model_dir, exist_ok=True)

# 保存模型
joblib.dump(best_model, os.path.join(model_dir, 'best_model.pkl'))
joblib.dump(le_major, os.path.join(model_dir, 'le_major.pkl'))
joblib.dump(le_gender, os.path.join(model_dir, 'le_gender.pkl'))

print("\n所有模型训练完成!")
