import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class TCNFeatureExtractor:
    def __init__(self, encoder, device, feature_dir='tcn_activations'):
        self.encoder = encoder
        self.device = device
        self.feature_dir = feature_dir
        self.outputs_by_class = defaultdict(list)
        os.makedirs(feature_dir, exist_ok=True)

    def extract_features(self, segments, labels, ids, N=5, K=5, Q=5, episodes=100):
        self.encoder.eval()

        for ep in range(episodes):
            try:
                sx, sy, qx, qy = create_episode(segments, labels, ids, N=N, K=K, Q=Q)
                if len(qx) == 0:
                    continue

                qx = torch.tensor(qx, dtype=torch.float32).to(self.device)
                qyt = torch.tensor(qy, dtype=torch.long).to(self.device)

                output = self.encoder(qx)

                for i in range(qx.size(0)):
                    self.outputs_by_class[qyt[i].item()].append(output[i].cpu().detach().numpy())

            except Exception as e:
                print(f"Skipping episode {ep} due to: {e}")

        self.save_outputs()

    def save_outputs(self):
        for cls_idx, output_list in self.outputs_by_class.items():
            avg_output = np.mean(np.stack(output_list), axis=0)
            np.save(os.path.join(self.feature_dir, f'class{cls_idx}_avg_output.npy'), avg_output)

            top = np.argsort(avg_output)[-5:][::-1]
            plt.figure(figsize=(10, 4))
            plt.bar(range(len(top)), avg_output[top], tick_label=[f'F{t}' for t in top])
            plt.title(f"Class {cls_idx} - Top Active Features")
            plt.ylabel("Mean Feature Value")
            plt.tight_layout()
            plt.savefig(os.path.join(self.feature_dir, f'class{cls_idx}_top_features.png'))
            plt.close()

        print(f"TCN features saved in: {os.path.abspath(self.feature_dir)}")

feature_extractor = TCNFeatureExtractor(encoder=encoder, device=device, feature_dir='tcn_activations')
feature_extractor.extract_features(segments=all_segments, labels=record_labels, ids=record_ids, N=5, K=5, Q=1, episodes=20)