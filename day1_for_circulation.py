print("本周深度学习任务清单：")
topics=["变量","条件判断","循环","函数"]
        
#way1 for
for i in range(len(topics)):
    print(f"第{i+1}天：掌握{topics[i]}")

#way2 while
print("每日进度跟踪：")
day=1
while day<=7:
    print(f"DAY{day}:已完成" if day<=4 else f"DAY{day}:进行中")
    day+=1

books=["python入门","机器学习实战","深度学习导论"]

for index,book in enumerate(books,1):
    print(f"第{index}周：{book}")