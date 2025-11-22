import matplotlib.pyplot as plt
import numpy as np

def plot_curves():
    # ---------------------------------------------------------
    # 📝 REAL DATA (Extracted from your Colab Logs)
    # ---------------------------------------------------------
    epochs = np.arange(1, 16) # Epochs 1 to 15
    
    # Training Loss (Blue Line) - Steadily decreasing (Model memorizing)
    train_loss = [
        0.8670, 0.0574, 0.0313, 0.0211, 0.0219, 
        0.0079, 0.0072, 0.0069, 0.0046, 0.0043, 
        0.0045, 0.0039, 0.0035, 0.0032, 0.0030
    ]
    
    # Validation Loss (Red Line) - The "U" Shape (Overfitting check)
    val_loss = [
        0.1195, 0.0479, 0.0479, 0.0454, 0.0531, 
        0.0566, 0.0588, 0.0595, 0.0648, 0.0772, 
        0.0738, 0.0802, 0.0796, 0.0830, 0.0839
    ]

    # ---------------------------------------------------------
    # 🎨 PLOTTING LOGIC
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Plot curves
    plt.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r--s', label='Validation Loss', linewidth=2)
    
    # Highlight the Best Model (Epoch 4)
    best_epoch = 4
    best_val_loss = 0.0454
    
    # Draw a green line/dot to show where you "saved" the model
    plt.axvline(x=best_epoch, color='green', linestyle=':', alpha=0.8, label='Best Model Checkpoint (Epoch 4)')
    plt.scatter([best_epoch], [best_val_loss], color='green', s=150, zorder=5, edgecolors='black')
    
    # Add a helpful annotation text pointing to the dot
    plt.annotate(f'Lowest Loss\n(0.045)', 
                 xy=(best_epoch, best_val_loss), 
                 xytext=(best_epoch+1, best_val_loss+0.02),
                 arrowprops=dict(facecolor='green', shrink=0.05))

    # Labels and Titles
    plt.title('DistilBERT Training Dynamics: Loss vs Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.xticks(epochs) # Show every epoch number on X-axis
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save image for your report
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=300)
    print("✅ Graph saved as 'training_curves.png'")
    plt.show()

if __name__ == "__main__":
    plot_curves()