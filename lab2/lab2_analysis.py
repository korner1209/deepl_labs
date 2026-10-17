import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from pathlib import Path
import os

from lab2_starter import load_predictions

def fast_threshold_search(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    desc_idx = np.argsort(scores)[::-1]
    y_sorted = y_true[desc_idx]
    s_sorted = scores[desc_idx]
    
    total_p = np.sum(y_true == 1)
    total_n = np.sum(y_true == 0)
    
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    
    distinct_idx = np.where(np.diff(s_sorted) != 0)[0]
    distinct_idx = np.append(distinct_idx, len(s_sorted) - 1)
    
    t_vals = s_sorted[distinct_idx]
    tp_vals = tp_cum[distinct_idx]
    fp_vals = fp_cum[distinct_idx]
    
    fn_vals = total_p - tp_vals
    tn_vals = total_n - fp_vals
    
    t_vals = np.insert(t_vals, 0, np.inf)
    tp_vals = np.insert(tp_vals, 0, 0)
    fp_vals = np.insert(fp_vals, 0, 0)
    fn_vals = np.insert(fn_vals, 0, total_p)
    tn_vals = np.insert(tn_vals, 0, total_n)
    
    c_vals = 10 * fn_vals + fp_vals
    
    return pd.DataFrame({
        'threshold': t_vals, 'TP': tp_vals, 'FP': fp_vals, 'FN': fn_vals, 'TN': tn_vals, 'Cost': c_vals
    })

def direct_threshold_search(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    candidates = np.unique(scores)
    candidates = np.append(candidates, np.inf)
    candidates = np.sort(candidates)[::-1]
    
    results = []
    for t in candidates:
        preds = (scores >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        tn = np.sum((preds == 0) & (y_true == 0))
        c = 10 * fn + fp
        results.append((t, tp, fp, fn, tn, c))
        
    return pd.DataFrame(results, columns=['threshold', 'TP', 'FP', 'FN', 'TN', 'Cost'])

def find_optimal_threshold(df: pd.DataFrame) -> pd.Series:
    min_cost = df['Cost'].min()
    best_rows = df[df['Cost'] == min_cost]
    return best_rows.sort_values('threshold', ascending=False).iloc[0]

def print_test_errors(y_true, scores, amount, row_ids, threshold):
    preds = (scores >= threshold).astype(int)
    
    all_fp_idx = np.where((preds == 1) & (y_true == 0))[0]
    all_fn_idx = np.where((preds == 0) & (y_true == 1))[0]
    
    fp_sorted_by_row = all_fp_idx[np.argsort(row_ids[all_fp_idx])]
    fn_sorted_by_row = all_fn_idx[np.argsort(row_ids[all_fn_idx])]
    
    fp_indices = fp_sorted_by_row[:5]
    fn_indices = fn_sorted_by_row[:5]
    
    print("\nРядок CSV | Тип помилки | Мітка | Оцінка s | Amount | Відстань |")
    print("-" * 65)
    for idx in fp_indices:
        dist = abs(scores[idx] - threshold)
        print(f"{row_ids[idx]:<9} | FP          | {y_true[idx]:<5} | {scores[idx]:.6f} | {amount[idx]:<6} | {dist:.6f} |")
    for idx in fn_indices:
        dist = abs(scores[idx] - threshold)
        print(f"{row_ids[idx]:<9} | FN          | {y_true[idx]:<5} | {scores[idx]:.6f} | {amount[idx]:<6} | {dist:.6f} |")

def main():
    results_dir = Path("results")
    if not results_dir.exists():
        os.makedirs(results_dir)

    data = load_predictions(results_dir)
    
    print("=" * 60)
    print("1: Навчання та збереження прогнозів")
    print("=" * 60)
    for name in ("train", "validation", "test"):
        y = data[f"{name}_y"]
        total = len(y)
        pos = int(np.count_nonzero(y == 1))
        frac = pos / total
        print(f"{name.capitalize():<10} : {total:>7} об'єктів | {pos:>4} шахрайських | Частка: {frac:.6f} ({frac*100:.4f}%)")


    print("\n" + "=" * 60)
    print("2: Вибір порога на валідаційній вибірці")
    print("=" * 60)
    val_y = data["validation_y"]
    val_scores = data["validation_scores"]
    
    fpr, tpr, _ = roc_curve(val_y, val_scores)
    auc_score = roc_auc_score(val_y, val_scores)
    precision_curve, recall_curve, _ = precision_recall_curve(val_y, val_scores)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-крива')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(recall_curve, precision_curve)
    plt.xlabel('Recall (Повнота)')
    plt.ylabel('Precision (Точність)')
    plt.title('PR-крива')
    plt.tight_layout()
    plt.savefig(results_dir / 'validation_curves.png')
    
    print(f"ROC-AUC: {auc_score:.6f}")
    
    val_candidates = fast_threshold_search(val_y, val_scores)
    val_candidates.to_csv(results_dir / "validation_thresholds.csv", index=False)
    
    opt_row = find_optimal_threshold(val_candidates)
    t_star = opt_row['threshold']
    
    default_row = val_candidates[val_candidates['threshold'] <= 0.5].iloc[0] if not val_candidates[val_candidates['threshold'] <= 0.5].empty else val_candidates.iloc[-1]
    all_neg_row = val_candidates[val_candidates['threshold'] == np.inf].iloc[0]
    
    summary_df = pd.DataFrame([default_row, opt_row, all_neg_row])
    summary_df = summary_df[['threshold', 'TP', 'FP', 'FN', 'TN', 'Cost']]
    summary_df.columns = ['Поріг', 'TP', 'FP', 'FN', 'TN', 'C']
    summary_df.index = ['Стандартний поріг', 'Мінімум вартості', 'Усі негативні']
    summary_df.index.name = 'Правило на validation'
    
    pd.set_option('display.float_format', lambda x: '%.6f' % x)
    print(summary_df.to_string())
    print(f"\nЗафіксований оптимальний поріг (t*): {t_star:.6f}")

    print("\n" + "=" * 60)
    print("3: Перевірка реалізації пошуку порога")
    print("=" * 60)
    A_y = np.array([0, 1, 0, 1, 0, 0])
    A_s = np.array([0.9, 0.7, 0.7, 0.4, 0.2, 0.1])
    
    B_y = np.array([0, 0, 0])
    B_s = np.array([0.9, 0.6, 0.3])
    
    C_y = np.array([1, 1] + [0]*10 + [0])
    C_s = np.array([0.8] + [0.5]*11 + [0.2])
    
    for name, y, s in [('A', A_y, A_s), ('B', B_y, B_s), ('C', C_y, C_s)]:
        print(f"\nНабір {name}:")
        
        df_fast = fast_threshold_search(y, s)
        opt_fast = find_optimal_threshold(df_fast)
        
        df_direct = direct_threshold_search(y, s)
        opt_direct = find_optimal_threshold(df_direct)
        
        print("Таблиця кандидатів (повільний перебір):")
        print(df_direct.to_string(index=False))
        
        frames_match = df_fast.equals(df_direct) or (df_fast.values == df_direct.values).all()
        thresholds_match = (opt_fast['threshold'] == opt_direct['threshold'])
        
        if frames_match and thresholds_match:
            print(f"-> Усі лічильники (TP, FP, FN, TN), вартість (C) та обраний поріг (t* = {opt_direct['threshold']:.1f})")
            print("   збігаються точно між швидким алгоритмом та прямим перебором!")
            print("   Правило розв'язання нічиєї (найбільший поріг) спрацювало ідентично.")
        else:
            print("-> ПОМИЛКА: результати швидкого і повільного алгоритмів розходяться!")

    print("\n" + "=" * 60)
    print("4: Тестова оцінка та аналіз помилок")
    print("=" * 60)
    test_y = data["test_y"]
    test_scores = data["test_scores"]
    
    preds = (test_scores >= t_star).astype(int)
    TP = np.sum((preds == 1) & (test_y == 1))
    FP = np.sum((preds == 1) & (test_y == 0))
    FN = np.sum((preds == 0) & (test_y == 1))
    TN = np.sum((preds == 0) & (test_y == 0))
    Cost = 10 * FN + FP
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    print(f"Матриця плутанини: TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    print(f"Вартість (Cost): {Cost}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    
    if (TP + FP) == 0:
        print("Пояснення: Precision не визначена (знаменник нульовий), бо модель не передбачила жодного позитивного класу (TP+FP=0).")
    
    print_test_errors(test_y, test_scores, data["test_amount"], data["test_row_ids"], t_star)

if __name__ == "__main__":
    main()