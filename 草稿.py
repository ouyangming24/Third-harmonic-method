import numpy as np
import matplotlib.pyplot as plt

# 定义 x 范围
x = np.linspace(-6, 6, 400)
y = np.tanh(x)

# 绘图
plt.figure(figsize=(6, 4))
plt.plot(x, y, label='tanh(x)', color='steelblue')

# 辅助线
plt.axhline(1, color='gray', linestyle='--')
plt.axhline(-1, color='gray', linestyle='--')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# 标注
plt.title('tanh Function')
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.legend()
plt.grid(True)
plt.show()