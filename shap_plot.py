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
})  # 无衬线字体

shap_concat = np.load('/data/code/mywork/gas_sensor/weights/RGB_diff_refine_label_new_n2/shap_values.npy')
X_concat = np.load('/data/code/mywork/gas_sensor/weights/RGB_diff_refine_label_new_n2/shap_X_val.npy')
feature_names = [f'Dye{i}' for i in range(1, 9)]  # 假设特征名称为 Dye1 到 Dye8
shap.summary_plot(shap_concat, features=X_concat, feature_names=feature_names, show=False)

# 调整 colorbar 的标题和刻度字体大小
fig = plt.gcf()
cbar_ax = fig.axes[-1]
cbar_ax.set_ylabel("Feature value", fontsize=16, fontweight='bold')
cbar_ax.tick_params(labelsize=16)

for tick_label in cbar_ax.get_yticklabels():
    tick_label.set_fontweight('bold')

plt.xlabel("SHAP value", fontsize=12) # 横坐标标题
plt.ylabel("Feature", fontsize=14) # 纵坐标标题
plt.xticks(fontsize=12) # 横坐标刻度
plt.yticks(fontsize=12) # 纵坐标刻度
plt.title("SHAP Summary Plot", fontsize=16, fontweight='bold') # 图标题
plt.savefig(f'weights/RGB_diff_refine_label_new_n2/shap_summary_plot_customized2.svg', bbox_inches="tight")  # 保存图像
