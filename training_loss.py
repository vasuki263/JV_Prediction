import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Load history
# ============================================================

history = np.load(
    "runs/2026-09-10_15-13-14_SunPower_5-1/ae/training_history.npz"
)


epochs = np.arange(
    1,
    len(history["train_total"]) + 1
)


# ============================================================
# Total loss
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(
    epochs,
    history["train_total"],
    label="Training loss"
)

plt.plot(
    epochs,
    history["val_total"],
    label="Validation loss"
)

plt.xlabel("Epoch")
plt.ylabel("Total loss")

plt.legend()

plt.tight_layout()

# plt.savefig(
#     "training_validation_loss.png",
#     dpi=300,
#     bbox_inches="tight"
# )

plt.show()