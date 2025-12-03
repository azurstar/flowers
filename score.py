#!/usr/bin/env python3
"""
评分脚本
功能:
1. 对比预测文件和真实标签文件
2. 计算准确率 (Accuracy)
3. 生成分类报告 (Precision/Recall/F1)
4. 将预测错误的样本保存为 CSV，方便错误分析
"""

import pandas as pd
import argparse
import os
import sys
from sklearn.metrics import accuracy_score, classification_report

def score_predictions(pred_file, truth_file, error_output_file=None):
    print(f"正在加载预测文件: {pred_file}")
    print(f"正在加载真实标签: {truth_file}")

    if not os.path.exists(pred_file):
        print(f"错误: 找不到预测文件 {pred_file}")
        return
    if not os.path.exists(truth_file):
        print(f"错误: 找不到真实标签文件 {truth_file}")
        return

    # 1. 读取 CSV
    try:
        df_pred = pd.read_csv(pred_file)
        df_true = pd.read_csv(truth_file)
    except Exception as e:
        print(f"读取 CSV 失败: {e}")
        return

    # 2. 检查列名
    required_cols = ['filename', 'category_id']
    for col in required_cols:
        if col not in df_pred.columns:
            print(f"错误: 预测文件缺少列 '{col}'")
            return
        if col not in df_true.columns:
            print(f"错误: 真实标签文件缺少列 '{col}'")
            return

    # 3. 数据合并 (基于 filename)
    # 使用 inner join，确保只比较两个文件中都存在的图片
    merged = pd.merge(df_pred, df_true, on='filename', suffixes=('_pred', '_true'))
    
    if len(merged) == 0:
        print("错误: 两个文件中没有匹配的文件名！请检查 filename 列是否一致。")
        return

    print(f"\n成功匹配 {len(merged)} 个样本。")
    if len(merged) < len(df_pred):
        print(f"警告: 预测文件中有 {len(df_pred) - len(merged)} 个样本在真实标签中未找到，将被忽略。")

    # 4. 数据类型统一 (防止 '1' 和 1 不匹配)
    y_pred = merged['category_id_pred'].astype(str)
    y_true = merged['category_id_true'].astype(str)

    # 5. 计算准确率
    acc = accuracy_score(y_true, y_pred)
    print("\n" + "="*30)
    print(f" >> 总体准确率 (Accuracy): {acc:.4f} ({acc*100:.2f}%)")
    print("="*30)

    # 6. 生成详细分类报告 (可选，如果类别太多可能会刷屏)
    # print("\n分类报告:")
    # print(classification_report(y_true, y_pred, zero_division=0))

    # 7. 提取错误样本
    errors = merged[y_pred != y_true].copy()
    
    if len(errors) > 0:
        print(f"\n共有 {len(errors)} 个预测错误。")
        
        # 整理错误输出格式
        error_df = errors[['filename', 'category_id_true', 'category_id_pred']]
        if 'confidence' in errors.columns:
            error_df['confidence'] = errors['confidence']
            
        # 打印前 10 个错误
        print("\n部分错误示例:")
        print(error_df.head(10).to_string(index=False))

        # 保存错误文件
        if error_output_file:
            error_df.to_csv(error_output_file, index=False)
            print(f"\n[Info] 所有错误样本已保存至: {error_output_file}")
            print("您可以打开该文件分析模型混淆了哪些类别。")
    else:
        print("\n完美！没有预测错误！")

def main():
    parser = argparse.ArgumentParser(description='评测预测结果准确率')
    parser.add_argument('pred_csv', type=str, help='预测结果 CSV 文件路径')
    parser.add_argument('truth_csv', type=str, help='真实标签 CSV 文件路径')
    parser.add_argument('--output', type=str, default='error_analysis.csv', help='保存预测错误的文件路径')
    
    args = parser.parse_args()
    
    score_predictions(args.pred_csv, args.truth_csv, args.output)

if __name__ == '__main__':
    main()