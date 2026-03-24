import matplotlib.pyplot as plt
import shap
import numpy as np

# plt.rcParams.update({
#     'font.family': 'serif', 
#     'font.serif': ['DejaVu Serif'],
# })  # 有衬线字体

plt.rcParams.update({
    'font.family': 'sans-serif', 
    'font.sans-serif': ['DejaVu Sans'],
})

shap_concat = np.load('shap_values.npy')
X_concat = np.load('shap_X_val.npy')
feature_names = [f'Dye{i}' for i in range(1, 9)]
shap.summary_plot(shap_concat, features=X_concat, feature_names=feature_names, show=False)

plt.xlabel("SHAP value", fontsize=12)
plt.ylabel("Feature value", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("SHAP Summary Plot", fontsize=16, fontweight='bold')
plt.savefig(f'shap_summary_plot_customized.svg', bbox_inches="tight")
