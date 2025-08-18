import matplotlib.pyplot as plt
import numpy as np

# Confusion matrix data
conf_matrices = {
    "Environment Dataset": np.array([[2153, 411],
                                     [224, 354]]),
    "External Dataset": np.array([[297, 12],
                                  [334, 357]])
}

# Labels
classes = ["Actual Positive", "Actual Negative"]
preds = ["Predicted Positive", "Predicted Negative"]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for ax, (title, cm) in zip(axes, conf_matrices.items()):
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=16, pad=15)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(preds, rotation=30, ha='right', fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)
    #ax.set_xlabel('Predicted Label', fontsize=12)
    #ax.set_ylabel('True Label', fontsize=12)
    ax.grid(False)

    # Annotate values inside matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    fontsize=13, fontweight='bold',
                    color="white" if cm[i, j] > thresh else "black")

    # Compute metrics
    TP, FN, FP, TN = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    accuracy = (TP + TN) / cm.sum()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    fnr = FN / (TP + FN) if (TP + FN) > 0 else 0
    tnr = TN / (FP + TN) if (FP + TN) > 0 else 0

    # Stats as caption under each matrix
    ax.text(0.5, -0.25,
        f"Acc: {accuracy:.2f} | Prec: {precision:.2f} | Rec: {recall:.2f} | "
        f"F1: {f1:.2f} | FPR: {fpr:.2f} | FNR: {fnr:.2f} | TNR: {tnr:.2f}",
        transform=ax.transAxes,
        ha='center', va='top', fontsize=11,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Add vertical colorbar for each subplot
    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("confusion_matrices_with_stats.png", dpi=300, bbox_inches='tight')
plt.show()
