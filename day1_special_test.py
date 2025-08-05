#特殊任务：用AI生成鼓励海报
#print("Hello AI World!")
name=input("请输入你的名字：")


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams["font.family"]=['SimHei']
plt.figure(figsize=(6,4))
plt.text(0.5,0.5,f"欢迎{name}同学第一天进度达成！",
    fontsize=20,ha='center')
plt.savefig('day1_achevement.png')
plt.show()