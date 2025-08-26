import numpy as np
import matplotlib.pyplot as plt

# 设置字体为 Arial
plt.rcParams['font.family'] = 'Times New Roman'

# RUM data
alpha_rum = np.array([0.00, 0.10, 0.20, 0.30, 0.40, 0.50])
deploy_rum = np.array([0.767530, 0.791810, 0.814870, 0.839950, 0.862100, 0.886300])
attack_rum = np.array([0.232470, 0.208190, 0.185130, 0.160050, 0.137900, 0.113700])

# UUM data
alpha_uum = np.array([0.00, 0.10, 0.20, 0.30, 0.40, 0.50])
deploy_uum = np.array([0.360880, 0.422890, 0.486320, 0.549550, 0.612920, 0.677680])
attack_uum = np.array([0.639120, 0.577110, 0.513680, 0.450450, 0.387080, 0.322320])

# SUUM data
alpha_suum = np.array([0.00, 0.10, 0.20, 0.30, 0.40, 0.50])
deploy_suum = np.array([0.359262, 0.288309, 0.219697, 0.164247, 0.120530, 0.087381])
downgrade_suum = np.array([0.640738, 0.390660, 0.229692, 0.133782, 0.075009, 0.041607])
attack_suum = 1 - deploy_suum - downgrade_suum

# 计算角度
angles = np.linspace(0, 2 * np.pi, len(alpha_rum), endpoint=False).tolist()
angles += angles[:1]  # 闭合雷达图

# 创建画布
fig, axes = plt.subplots(1, 3, subplot_kw=dict(polar=True), figsize=(15, 5))

# 设置通用样式
line_width = 3
marker_size = 10
font_size = 20

# 绘制RUM雷达图
deploy_rum = np.concatenate((deploy_rum, [deploy_rum[0]]))
attack_rum = np.concatenate((attack_rum, [attack_rum[0]]))
axes[0].plot(angles, deploy_rum, label='Deployment State Probability', linewidth=line_width, marker='<', markersize=marker_size)
axes[0].plot(angles, attack_rum, label='Attack State Probability', linewidth=line_width, linestyle='--', marker='d', markersize=marker_size)
axes[0].set_theta_zero_location("N")
axes[0].set_theta_direction(-1)
axes[0].set_title('RUM Steady-state Probability', fontsize=font_size, pad=30)  # 增加pad值
axes[0].set_xticks(angles[:-1])
axes[0].set_xticklabels(alpha_rum, fontsize=20)  # 设置标签字体大小为20
axes[0].set_yticks(np.arange(0, 1.1, 0.2))  # 设置y轴刻度从0到1，间隔0.2
axes[0].set_xlabel('Adversary Relative Power $\\alpha$', fontsize=20)

# 绘制UUM雷达图
deploy_uum = np.concatenate((deploy_uum, [deploy_uum[0]]))
attack_uum = np.concatenate((attack_uum, [attack_uum[0]]))
axes[1].plot(angles, deploy_uum, label='Deployment State Probability', linewidth=line_width, marker='<', markersize=marker_size)
axes[1].plot(angles, attack_uum, label='Attack State Probability', linewidth=line_width, linestyle='--', marker='d', markersize=marker_size)
axes[1].set_theta_zero_location("N")
axes[1].set_theta_direction(-1)
axes[1].set_title('UUM Steady-state Probability', fontsize=font_size, pad=30)  # 增加pad值
axes[1].set_xticks(angles[:-1])
axes[1].set_xticklabels(alpha_uum, fontsize=20)  # 设置标签字体大小为20
axes[1].set_yticks(np.arange(0, 1.1, 0.2))  # 设置y轴刻度从0到1，间隔0.2
axes[1].set_xlabel('Adversary Relative Power $\\alpha$', fontsize=20)

# 绘制SUUM雷达图
deploy_suum = np.concatenate((deploy_suum, [deploy_suum[0]]))
downgrade_suum = np.concatenate((downgrade_suum, [downgrade_suum[0]]))
attack_suum = np.concatenate((attack_suum, [attack_suum[0]]))
axes[2].plot(angles, deploy_suum, label='Deployment State', linewidth=line_width, marker='<', markersize=marker_size)
axes[2].plot(angles, attack_suum, label='Attack State', linewidth=line_width, linestyle='--', marker='d', markersize=marker_size)
axes[2].plot(angles, downgrade_suum, label='Downgrade State', linewidth=line_width, linestyle=':', marker='>', markersize=marker_size)
axes[2].set_theta_zero_location("N")
axes[2].set_theta_direction(-1)
axes[2].set_title('SUUM Steady-state Probability', fontsize=font_size, pad=30)  # 增加pad值
axes[2].set_xticks(angles[:-1])
axes[2].set_xticklabels(alpha_suum, fontsize=20)  # 设置标签字体大小为20
axes[2].set_yticks(np.arange(0, 1.1, 0.2))  # 设置y轴刻度从0到1，间隔0.2
axes[2].set_xlabel('Adversary Relative Power $\\alpha$', fontsize=20)

# 只保留SUUM的图例，放到左下角
axes[2].legend(fontsize=20, loc='lower left', bbox_to_anchor=(0.1, 0.1), framealpha=0.5)

plt.tight_layout()
plt.show()