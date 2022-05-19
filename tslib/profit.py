import numpy as np
import pandas as pd


def calculate_profit(trues, preds, day_deposit_rate, night_deposit_rate, night_loan_rate):

    trues = trues.copy()
    profit = np.zeros(trues.shape)

    # Начало дня
    positive_pred_mask = (preds > 0)
    # Вкладываем средства по прогнозу
    profit[positive_pred_mask] += day_deposit_rate * preds[positive_pred_mask]
    # Из-за этого уменьшается ликвидность
    trues[positive_pred_mask] -= preds[positive_pred_mask]
    # Покрываем дефицит ликвидности в течение дня, если прогноз < 0
    trues[~positive_pred_mask] -= preds[~positive_pred_mask]

    # Конец дня
    positive_balance_mask = (trues > 0)
    # Если остается положительная ликвидность, вкладываем
    profit[positive_balance_mask] += night_deposit_rate * trues[positive_balance_mask]
    # Если есть дефицит, занимаем
    profit[~positive_balance_mask] += night_loan_rate * trues[~positive_balance_mask]

    return profit