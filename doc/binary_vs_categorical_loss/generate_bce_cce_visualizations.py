"""
Visualization Generator for Binary vs Categorical Cross-Entropy Report

This script generates all visualizations for the comprehensive report on
Binary Cross-Entropy vs Categorical Cross-Entropy.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)

# ============================================================================
# 1. Binary Cross-Entropy Loss Landscape
# ============================================================================

def plot_bce_loss_landscape():
    """Plot BCE loss for different predicted probabilities"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Loss when true label is 1
    y_pred = np.linspace(0.001, 0.999, 1000)
    loss_y1 = -np.log(y_pred)
    
    axes[0].plot(y_pred, loss_y1, 'b-', linewidth=2.5, label='BCE Loss (y=1)')
    axes[0].axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Uncertainty (0.5)')
    axes[0].axhline(y=-np.log(0.5), color='r', linestyle='--', alpha=0.5)
    axes[0].scatter([0.5], [-np.log(0.5)], color='red', s=100, zorder=5, label='Uncertainty point')
    axes[0].scatter([0.9], [-np.log(0.9)], color='green', s=100, zorder=5, label='Confident (0.9)')
    axes[0].scatter([0.1], [-np.log(0.1)], color='orange', s=100, zorder=5, label='Wrong (0.1)')
    axes[0].set_xlabel('Predicted Probability (ŷ)', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Binary Cross-Entropy Loss (True Label = 1)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 5)
    
    # Right: Loss when true label is 0
    loss_y0 = -np.log(1 - y_pred)
    
    axes[1].plot(y_pred, loss_y0, 'r-', linewidth=2.5, label='BCE Loss (y=0)')
    axes[1].axvline(x=0.5, color='b', linestyle='--', alpha=0.5, label='Uncertainty (0.5)')
    axes[1].axhline(y=-np.log(0.5), color='b', linestyle='--', alpha=0.5)
    axes[1].scatter([0.5], [-np.log(0.5)], color='blue', s=100, zorder=5, label='Uncertainty point')
    axes[1].scatter([0.1], [-np.log(0.9)], color='green', s=100, zorder=5, label='Confident (0.1)')
    axes[1].scatter([0.9], [-np.log(0.1)], color='orange', s=100, zorder=5, label='Wrong (0.9)')
    axes[1].set_xlabel('Predicted Probability (ŷ)', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Binary Cross-Entropy Loss (True Label = 0)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 5)
    
    plt.tight_layout()
    plt.savefig('bce_loss_landscape.png', dpi=300, bbox_inches='tight')
    print("Saved: bce_loss_landscape.png")
    plt.close()

# ============================================================================
# 2. Categorical Cross-Entropy Loss Landscape
# ============================================================================

def plot_cce_loss_landscape():
    """Plot CCE loss for 3-class example"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # True label is class 0
    y_true = np.array([1, 0, 0])
    y_pred_base = np.linspace(0.001, 0.998, 1000)
    
    losses_class0 = []
    for p0 in y_pred_base:
        # Ensure probabilities sum to 1
        p1 = (1 - p0) / 2
        p2 = (1 - p0) / 2
        y_pred = np.array([p0, p1, p2])
        loss = -np.sum(y_true * np.log(y_pred + 1e-10))
        losses_class0.append(loss)
    
    axes[0].plot(y_pred_base, losses_class0, 'b-', linewidth=2.5)
    axes[0].axvline(x=1/3, color='r', linestyle='--', alpha=0.5, label='Uniform (1/3)')
    axes[0].scatter([1/3], [-np.log(1/3)], color='red', s=100, zorder=5)
    axes[0].scatter([0.8], [-np.log(0.8)], color='green', s=100, zorder=5, label='Confident (0.8)')
    axes[0].scatter([0.1], [-np.log(0.1)], color='orange', s=100, zorder=5, label='Wrong (0.1)')
    axes[0].set_xlabel('P(Class 0)', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('CCE Loss (True Label = Class 0)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_ylim(0, 5)
    
    # True label is class 1
    y_true = np.array([0, 1, 0])
    losses_class1 = []
    for p1 in y_pred_base:
        p0 = (1 - p1) / 2
        p2 = (1 - p1) / 2
        y_pred = np.array([p0, p1, p2])
        loss = -np.sum(y_true * np.log(y_pred + 1e-10))
        losses_class1.append(loss)
    
    axes[1].plot(y_pred_base, losses_class1, 'g-', linewidth=2.5)
    axes[1].axvline(x=1/3, color='r', linestyle='--', alpha=0.5, label='Uniform (1/3)')
    axes[1].scatter([1/3], [-np.log(1/3)], color='red', s=100, zorder=5)
    axes[1].scatter([0.8], [-np.log(0.8)], color='green', s=100, zorder=5, label='Confident (0.8)')
    axes[1].scatter([0.1], [-np.log(0.1)], color='orange', s=100, zorder=5, label='Wrong (0.1)')
    axes[1].set_xlabel('P(Class 1)', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('CCE Loss (True Label = Class 1)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim(0, 5)
    
    # True label is class 2
    y_true = np.array([0, 0, 1])
    losses_class2 = []
    for p2 in y_pred_base:
        p0 = (1 - p2) / 2
        p1 = (1 - p2) / 2
        y_pred = np.array([p0, p1, p2])
        loss = -np.sum(y_true * np.log(y_pred + 1e-10))
        losses_class2.append(loss)
    
    axes[2].plot(y_pred_base, losses_class2, 'orange', linewidth=2.5)
    axes[2].axvline(x=1/3, color='r', linestyle='--', alpha=0.5, label='Uniform (1/3)')
    axes[2].scatter([1/3], [-np.log(1/3)], color='red', s=100, zorder=5)
    axes[2].scatter([0.8], [-np.log(0.8)], color='green', s=100, zorder=5, label='Confident (0.8)')
    axes[2].scatter([0.1], [-np.log(0.1)], color='orange', s=100, zorder=5, label='Wrong (0.1)')
    axes[2].set_xlabel('P(Class 2)', fontsize=12)
    axes[2].set_ylabel('Loss', fontsize=12)
    axes[2].set_title('CCE Loss (True Label = Class 2)', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_ylim(0, 5)
    
    plt.tight_layout()
    plt.savefig('cce_loss_landscape.png', dpi=300, bbox_inches='tight')
    print("Saved: cce_loss_landscape.png")
    plt.close()

# ============================================================================
# 3. Sigmoid vs Softmax Comparison
# ============================================================================

def plot_sigmoid_vs_softmax():
    """Compare sigmoid and softmax activation functions"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Sigmoid function
    z = np.linspace(-10, 10, 1000)
    sigmoid = 1 / (1 + np.exp(-z))
    
    axes[0].plot(z, sigmoid, 'b-', linewidth=2.5, label='Sigmoid: σ(z)')
    axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Decision threshold')
    axes[0].axvline(x=0, color='g', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('z (logit)', fontsize=12)
    axes[0].set_ylabel('σ(z)', fontsize=12)
    axes[0].set_title('Sigmoid Activation Function', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_ylim(-0.1, 1.1)
    
    # Right: Softmax for 3 classes
    z_values = np.linspace(-5, 5, 1000)
    z0 = np.zeros_like(z_values)
    z1 = z_values
    z2 = -z_values
    
    # Calculate softmax probabilities
    exp_z0 = np.exp(z0)
    exp_z1 = np.exp(z1)
    exp_z2 = np.exp(z2)
    sum_exp = exp_z0 + exp_z1 + exp_z2
    
    softmax_0 = exp_z0 / sum_exp
    softmax_1 = exp_z1 / sum_exp
    softmax_2 = exp_z2 / sum_exp
    
    axes[1].plot(z_values, softmax_0, 'b-', linewidth=2.5, label='P(Class 0)')
    axes[1].plot(z_values, softmax_1, 'g-', linewidth=2.5, label='P(Class 1)')
    axes[1].plot(z_values, softmax_2, 'orange', linewidth=2.5, label='P(Class 2)')
    axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('z₁ - z₀ (logit difference)', fontsize=12)
    axes[1].set_ylabel('Probability', fontsize=12)
    axes[1].set_title('Softmax Activation (3 Classes)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig('sigmoid_vs_softmax.png', dpi=300, bbox_inches='tight')
    print("Saved: sigmoid_vs_softmax.png")
    plt.close()

# ============================================================================
# 4. Probability Distribution Comparison
# ============================================================================

def plot_probability_comparison():
    """Compare probability outputs from sigmoid vs softmax"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Example scores/logits
    z_binary = 1.5  # Single logit for binary
    z_multiclass = np.array([1.5, 0.8, -0.5])  # Logits for 3 classes
    
    # Binary: Sigmoid
    sigmoid_prob = 1 / (1 + np.exp(-z_binary))
    
    ax1 = fig.add_subplot(gs[0, 0])
    classes = ['Class 0', 'Class 1']
    probs = [1 - sigmoid_prob, sigmoid_prob]
    colors = ['lightcoral', 'lightblue']
    bars = ax1.bar(classes, probs, color=colors, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title('Binary: Sigmoid Output\n(Sum = {:.2f})'.format(sum(probs)), 
                  fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{prob:.3f}', ha='center', fontsize=11, fontweight='bold')
    
    # Multiclass: Softmax
    exp_z = np.exp(z_multiclass)
    softmax_probs = exp_z / np.sum(exp_z)
    
    ax2 = fig.add_subplot(gs[0, 1])
    classes = ['Class 0', 'Class 1', 'Class 2']
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    bars = ax2.bar(classes, softmax_probs, color=colors, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_title('Multiclass: Softmax Output\n(Sum = {:.2f})'.format(np.sum(softmax_probs)), 
                  fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, prob in zip(bars, softmax_probs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{prob:.3f}', ha='center', fontsize=11, fontweight='bold')
    
    # Problem: Using sigmoid for multiclass (WRONG)
    sigmoid_multiclass = 1 / (1 + np.exp(-z_multiclass))
    
    ax3 = fig.add_subplot(gs[0, 2])
    bars = ax3.bar(classes, sigmoid_multiclass, color=colors, edgecolor='black', 
                   linewidth=2, alpha=0.7)
    ax3.set_ylabel('Probability', fontsize=12)
    ax3.set_title('WRONG: Sigmoid for Multiclass\n(Sum = {:.2f} ≠ 1!)'.format(np.sum(sigmoid_multiclass)), 
                  fontsize=13, fontweight='bold', color='red')
    ax3.set_ylim(0, 1.5)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Required sum = 1')
    for bar, prob in zip(bars, sigmoid_multiclass):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{prob:.3f}', ha='center', fontsize=11, fontweight='bold')
    ax3.legend()
    
    # Loss comparison
    # Binary example
    y_true_binary = 1
    y_pred_binary = sigmoid_prob
    bce_loss = -(y_true_binary * np.log(y_pred_binary) + 
                 (1 - y_true_binary) * np.log(1 - y_pred_binary))
    
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(['BCE Loss'], [bce_loss], color='steelblue', edgecolor='black', linewidth=2)
    ax4.set_ylabel('Loss Value', fontsize=12)
    ax4.set_title('Binary Cross-Entropy Loss\n(y=1, ŷ={:.3f})'.format(y_pred_binary), 
                  fontsize=13, fontweight='bold')
    ax4.text(0, bce_loss + 0.01, f'{bce_loss:.3f}', ha='center', 
            fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Categorical example
    y_true_cat = np.array([0, 1, 0])  # Class 1 is true
    cce_loss = -np.sum(y_true_cat * np.log(softmax_probs + 1e-10))
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(['CCE Loss'], [cce_loss], color='darkgreen', edgecolor='black', linewidth=2)
    ax5.set_ylabel('Loss Value', fontsize=12)
    ax5.set_title('Categorical Cross-Entropy Loss\n(y=[0,1,0], ŷ={})'.format(
        [f'{p:.3f}' for p in softmax_probs]), fontsize=13, fontweight='bold')
    ax5.text(0, cce_loss + 0.01, f'{cce_loss:.3f}', ha='center', 
            fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Gradient flow comparison
    ax6 = fig.add_subplot(gs[1, 2])
    methods = ['Binary\n(Sigmoid+BCE)', 'Multiclass\n(Softmax+CCE)']
    gradient_complexity = [1, 3]  # Number of outputs
    colors_grad = ['steelblue', 'darkgreen']
    bars = ax6.bar(methods, gradient_complexity, color=colors_grad, 
                   edgecolor='black', linewidth=2)
    ax6.set_ylabel('Number of Outputs', fontsize=12)
    ax6.set_title('Gradient Complexity', fontsize=13, fontweight='bold')
    ax6.set_ylim(0, 4)
    ax6.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, gradient_complexity):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val} output(s)', ha='center', fontsize=11, fontweight='bold')
    
    plt.suptitle('Binary vs Multiclass: Probability Distributions and Loss Functions', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig('probability_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: probability_comparison.png")
    plt.close()

# ============================================================================
# 5. Gradient Flow Visualization
# ============================================================================

def plot_gradient_flow():
    """Visualize how gradients flow in BCE vs CCE"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Binary Cross-Entropy Gradient Flow
    ax1 = axes[0]
    ax1.text(0.5, 0.9, 'Binary Cross-Entropy Gradient Flow', 
            ha='center', fontsize=14, fontweight='bold', transform=ax1.transAxes)
    
    # Draw flow diagram
    # Input
    ax1.add_patch(plt.Rectangle((0.1, 0.6), 0.15, 0.15, fill=True, 
                                facecolor='lightblue', edgecolor='black', linewidth=2))
    ax1.text(0.175, 0.675, 'X\n(m×n)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Weights
    ax1.add_patch(plt.Rectangle((0.3, 0.6), 0.15, 0.15, fill=True, 
                                facecolor='lightgreen', edgecolor='black', linewidth=2))
    ax1.text(0.375, 0.675, 'w\n(n,)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Output
    ax1.add_patch(plt.Rectangle((0.5, 0.6), 0.15, 0.15, fill=True, 
                                facecolor='lightyellow', edgecolor='black', linewidth=2))
    ax1.text(0.575, 0.675, 'ŷ\n(m,)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Error
    ax1.add_patch(plt.Rectangle((0.5, 0.3), 0.15, 0.15, fill=True, 
                                facecolor='lightcoral', edgecolor='black', linewidth=2))
    ax1.text(0.575, 0.375, 'Error\nŷ-y\n(m,)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Gradient
    ax1.add_patch(plt.Rectangle((0.3, 0.3), 0.15, 0.15, fill=True, 
                                facecolor='plum', edgecolor='black', linewidth=2))
    ax1.text(0.375, 0.375, '∇w\n(n,)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    ax1.arrow(0.25, 0.675, 0.05, 0, head_width=0.02, head_length=0.02, 
              fc='black', ec='black', linewidth=2)
    ax1.arrow(0.45, 0.675, 0.05, 0, head_width=0.02, head_length=0.02, 
              fc='black', ec='black', linewidth=2)
    ax1.arrow(0.575, 0.6, 0, -0.15, head_width=0.02, head_length=0.02, 
              fc='black', ec='black', linewidth=2)
    ax1.arrow(0.5, 0.375, -0.15, 0, head_width=0.02, head_length=0.02, 
              fc='black', ec='black', linewidth=2)
    
    ax1.set_xlim(0, 0.8)
    ax1.set_ylim(0.2, 0.9)
    ax1.axis('off')
    
    # Categorical Cross-Entropy Gradient Flow
    ax2 = axes[1]
    ax2.text(0.5, 0.9, 'Categorical Cross-Entropy Gradient Flow', 
            ha='center', fontsize=14, fontweight='bold', transform=ax2.transAxes)
    
    # Draw flow diagram
    # Input
    ax2.add_patch(plt.Rectangle((0.1, 0.6), 0.15, 0.15, fill=True, 
                                facecolor='lightblue', edgecolor='black', linewidth=2))
    ax2.text(0.175, 0.675, 'X\n(m×n)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Weights
    ax2.add_patch(plt.Rectangle((0.3, 0.6), 0.15, 0.15, fill=True, 
                                facecolor='lightgreen', edgecolor='black', linewidth=2))
    ax2.text(0.375, 0.675, 'W\n(K×n)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Output
    ax2.add_patch(plt.Rectangle((0.5, 0.6), 0.15, 0.15, fill=True, 
                                facecolor='lightyellow', edgecolor='black', linewidth=2))
    ax2.text(0.575, 0.675, 'ŷ\n(m×K)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Error
    ax2.add_patch(plt.Rectangle((0.5, 0.3), 0.15, 0.15, fill=True, 
                                facecolor='lightcoral', edgecolor='black', linewidth=2))
    ax2.text(0.575, 0.375, 'Error\nŷ-Y\n(m×K)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Gradient
    ax2.add_patch(plt.Rectangle((0.3, 0.3), 0.15, 0.15, fill=True, 
                                facecolor='plum', edgecolor='black', linewidth=2))
    ax2.text(0.375, 0.375, '∇W\n(K×n)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    ax2.arrow(0.25, 0.675, 0.05, 0, head_width=0.02, head_length=0.02, 
              fc='black', ec='black', linewidth=2)
    ax2.arrow(0.45, 0.675, 0.05, 0, head_width=0.02, head_length=0.02, 
              fc='black', ec='black', linewidth=2)
    ax2.arrow(0.575, 0.6, 0, -0.15, head_width=0.02, head_length=0.02, 
              fc='black', ec='black', linewidth=2)
    ax2.arrow(0.5, 0.375, -0.15, 0, head_width=0.02, head_length=0.02, 
              fc='black', ec='black', linewidth=2)
    
    ax2.set_xlim(0, 0.8)
    ax2.set_ylim(0.2, 0.9)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('gradient_flow.png', dpi=300, bbox_inches='tight')
    print("Saved: gradient_flow.png")
    plt.close()

# ============================================================================
# 6. Loss Comparison Table Visualization
# ============================================================================

def plot_loss_comparison_table():
    """Create a visual comparison table"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    table_data = [
        ['Aspect', 'Binary Cross-Entropy', 'Categorical Cross-Entropy'],
        ['Activation', 'Sigmoid: σ(z) = 1/(1+e^(-z))', 'Softmax: exp(z_i)/Σexp(z_j)'],
        ['Output Shape', 'Single probability (m,)', 'K probabilities (m, K)'],
        ['Sum Constraint', 'Implicit: P(1) + P(0) = 1', 'Explicit: ΣP(k) = 1'],
        ['Loss Function', 'J = -[y·log(ŷ) + (1-y)·log(1-ŷ)]', 'J = -Σ y_k·log(ŷ_k)'],
        ['Parameters', 'w: (n,), b: scalar', 'W: (K, n), b: (K, 1)'],
        ['Gradient Shape', 'dj_dw: (n,)', 'dW: (K, n)'],
        ['Use Case', 'Binary classification', 'Multiclass classification'],
        ['Class Competition', 'No (independent)', 'Yes (compete for probability)'],
        ['Number of Models', '1 per binary problem', '1 for all classes'],
    ]
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                    cellLoc='left', loc='center',
                    colWidths=[0.25, 0.375, 0.375])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8F5E9')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    # Highlight key differences
    key_rows = [1, 2, 4, 6, 8]  # Activation, Output, Loss, Gradient, Competition
    for row in key_rows:
        for j in range(3):
            table[(row, j)].set_facecolor('#FFF9C4')
    
    plt.title('Binary Cross-Entropy vs Categorical Cross-Entropy: Comprehensive Comparison', 
             fontsize=16, fontweight='bold', pad=20)
    plt.savefig('loss_comparison_table.png', dpi=300, bbox_inches='tight')
    print("Saved: loss_comparison_table.png")
    plt.close()

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("Generating visualizations for BCE vs CCE report...")
    print("=" * 60)
    
    plot_bce_loss_landscape()
    plot_cce_loss_landscape()
    plot_sigmoid_vs_softmax()
    plot_probability_comparison()
    plot_gradient_flow()
    plot_loss_comparison_table()
    
    print("=" * 60)
    print("All visualizations generated successfully!")
    print("\nGenerated files:")
    print("  1. bce_loss_landscape.png")
    print("  2. cce_loss_landscape.png")
    print("  3. sigmoid_vs_softmax.png")
    print("  4. probability_comparison.png")
    print("  5. gradient_flow.png")
    print("  6. loss_comparison_table.png")

