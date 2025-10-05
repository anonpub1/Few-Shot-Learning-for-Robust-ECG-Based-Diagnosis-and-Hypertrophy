import torch
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from model import TCNEncoder

def evaluate_with_logging(encoder, segments, labels, ids, device, N=5, K=5, Q=5, episodes=100, temp=1.0, save_path='evaluation_results.csv'):
    encoder.eval()
    all_true, all_pred, all_probs = [], [], []
    
    for ep in range(episodes):
        try:
            pass
        except Exception as e:
            print(f"Skipping episode {ep} due to error: {e}")
            continue

    df = pd.DataFrame({
        'true_label': all_true,
        'pred_label': all_pred,
    })
    probs_df = pd.DataFrame(all_probs, columns=[f'prob_class{i}' for i in range(all_probs[0].shape[0])])
    df = pd.concat([df, probs_df], axis=1)
    df.to_csv(save_path, index=False)
    print(f" Detailed evaluation saved to {save_path}. You can now use it for confusion matrices, ROC, F1, etc.")
