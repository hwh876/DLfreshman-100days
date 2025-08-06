# 房价预测线性回归模型（整合版）
# 核心逻辑：通过梯度下降法找到最优线性公式 y = w*x + b 拟合房价与面积的关系

###参考答案
'''

def neuron_predict(area, weight, bias):
    """神经元预测函数：根据面积、权重和偏置计算预测房价"""
    # 线性公式：预测房价 = 权重×面积 + 偏置
    return weight * area + bias

def train_neuron(areas, prices, epochs=100, learning_rate=0.0001):
    """训练函数：通过梯度下降法优化权重和偏置"""
    # 1. 初始化参数（初始猜测值）
    weight = 0.1  # 权重w
    bias = 1      # 偏置b
    
    # 2. 打印初始状态
    print(f"初始参数：权重W={weight:.3f}, 偏置b={bias:.3f}")
    print("="*50)
    
    # 3. 开始训练（迭代优化）
    for epoch in range(epochs):
        # 3.1 计算当前预测值
        predictions = [neuron_predict(area, weight, bias) for area in areas]
        
        # 3.2 计算误差（预测值 - 真实值）
        errors = [pred - price for pred, price in zip(predictions, prices)]
        
        # 3.3 计算权重和偏置的调整量（梯度下降核心）
        # 权重调整：与误差和面积的乘积相关
        weight_adjust = sum(error * area for error, area in zip(errors, areas)) / len(areas)
        # 偏置调整：只与误差相关
        bias_adjust = sum(errors) / len(areas)
        
        # 3.4 更新参数（向误差减小的方向调整）
        weight -= learning_rate * weight_adjust
        bias -= learning_rate * bias_adjust
        
        # 3.5 定期打印训练进度（每10轮）
        if (epoch + 1) % 10 == 0:
            print(f"第{epoch+1}轮训练：W={weight:.3f}, b={bias:.3f}")
    
    return weight, bias

# 4. 准备训练数据（面积和对应房价）
house_areas = [80, 120, 90]    # 房屋面积（平方米）
house_prices = [420, 630, 473] # 对应房价（万元）

# 5. 训练模型（学习房价规律）
trained_weight, trained_bias = train_neuron(
    areas=house_areas,
    prices=house_prices,
    epochs=100,        # 训练100轮
    learning_rate=0.0001 # 学习率（步长）
)

# 6. 验证训练结果
print("="*50)
print(f"最终学习到的参数：W={trained_weight:.3f}, b={trained_bias:.3f}")
print(f"理论最优参数：W=4.2, b=84")

# 7. 预测新数据（100平方米房价）
predicted_price = neuron_predict(100, trained_weight, trained_bias)
print(f"\n预测100平方米房价：{predicted_price:.0f}万元（理论值：504万元）")
'''

###V1,公式推导版调参公式，这个好像只找到局部最优解，要多一点有代表性的样本才能有用
'''
def predict(area,weight,bias):
    return weight*area+bias


def train(areas,prices,epochs=100,learn_rate=0.001):
    weight=0.1
    bias=1

    print(f"初始参数为w={weight},b={bias}")
    print("="*50)

    for epoch in range(epochs):

        pridicts=[predict(area,weight,bias) for area in areas]

        errors=[price-pridict for price,pridict in zip(prices,pridicts)]

        weight+=learn_rate*sum(error*area for error,area in zip(errors,areas))/len(areas)
        bias+=learn_rate*sum(errors)/len(areas)

        if (epoch%1000==0):
            print(f"第{epoch}次训练，w={weight:.3f},b={bias:.3f},误差：{errors}")

    return weight,bias


house_areas =[80,120,90]
house_prices =[420,630,473]
w_train,b_train=train(areas=house_areas,prices=house_prices,epochs=10000000,learn_rate=0.0001)

print("="*50)
print(f"最终学习的参数w={w_train:.3f},b={b_train:.3f}")
print(f"理论最优参数：W=4.2, b=84")

predict_price=predict(100,w_train,b_train)
print(f"\n预测100平方米房价：{predict_price:.0f}万元（理论值：504万元）")

'''
#V2 增加数据标准化
'''
def nomolize(data):
    max_value=max(data)
    min_value=min(data)
    return [(x-min_value)/(max_value-min_value) for x in data]


def neuron_predict(area, weight, bias):
    return weight * area + bias

def train_neuron(areas, prices, epochs=100, learning_rate=0.01):
    """训练函数：通过梯度下降法优化权重和偏置"""
    # 1. 初始化参数（初始猜测值）
    weight = 0.1  # 权重w
    bias = 1      # 偏置b
    
    # 2. 打印初始状态
    print(f"初始参数：权重W={weight:.3f}, 偏置b={bias:.3f}")
    print("="*50)
    
    # 3. 开始训练（迭代优化）
    for epoch in range(epochs):

        predictions = [neuron_predict(area, weight, bias) for area in areas]
        

        errors = [pred - price for pred, price in zip(predictions, prices)]
        
        # 3.3 计算权重和偏置的调整量（梯度下降核心）
        # 权重调整：与误差和面积的乘积相关
        weight_adjust = sum(error * area for error, area in zip(errors, areas)) / len(areas)
        # 偏置调整：只与误差相关
        bias_adjust = sum(errors) / len(areas)
        
        # 3.4 更新参数（向误差减小的方向调整）
        weight -= learning_rate * weight_adjust
        bias -= learning_rate * bias_adjust
        
        # 3.5 定期打印训练进度（每10轮）
        if (epoch + 1) % 10 == 0:
            print(f"第{epoch+1}轮训练：W={weight:.3f}, b={bias:.3f},误差为{errors}")
    
    return weight, bias

# 4. 准备训练数据（面积和对应房价）
house_areas = [80, 120, 90]    
house_prices = [420, 630, 473] 
nom_areas=nomolize(house_areas)
nom_prices=nomolize(house_prices)
print("标准化后的输入为:",nom_areas,nom_prices)

# 5. 训练模型（学习房价规律）
trained_weight, trained_bias = train_neuron(
    areas=nom_areas,
    prices=nom_prices,
    epochs=1000,        # 训练1000轮
    learning_rate=0.01 # 学习率（步长）
)

# 6. 验证训练结果
print("="*50)
print(f"最终学习到的参数：W={trained_weight:.3f}, b={trained_bias:.3f}")

# 7. 预测新数据（100平方米房价输入也要标准化）
predicted_price_nom = neuron_predict((100-min(house_areas))/(max(house_areas)-min(house_areas)), trained_weight, trained_bias)


#8.标准化预测结果反标准化
predicted_price=predicted_price_nom*(max(house_prices)-min(house_prices))+min(house_prices)
print(f"\n预测100平方米房价：{predicted_price:.0f}万元（理论值：504万元）")
'''


# v3 增加动态学习率：
def nomolize(data):
    max_value=max(data)
    min_value=min(data)
    return [(x-min_value)/(max_value-min_value) for x in data]


def neuron_predict(area, weight, bias):
    return weight * area + bias

def train_neuron(areas, prices, epochs=10000, initial_lr=0.01,decay=0.5):
    """训练函数：通过梯度下降法优化权重和偏置"""
    # 1. 初始化参数（初始猜测值）
    weight = 0.1  # 权重w
    bias = 1      # 偏置b
    
    # 2. 打印初始状态
    print(f"初始参数：权重W={weight:.3f}, 偏置b={bias:.3f}")
    print("="*50)
    
    # 3. 开始训练（迭代优化）
    for epoch in range(epochs):
        #5000轮衰减一次
        learning_rate=initial_lr*(1/(1+decay*(epoch//5000)))

        predictions = [neuron_predict(area, weight, bias) for area in areas]
        

        errors = [pred - price for pred, price in zip(predictions, prices)]
        
        # 3.3 计算权重和偏置的调整量（梯度下降核心）
        # 权重调整：与误差和面积的乘积相关
        weight_adjust = sum(error * area for error, area in zip(errors, areas)) / len(areas)
        # 偏置调整：只与误差相关
        bias_adjust = sum(errors) / len(areas)
        
        # 3.4 更新参数（向误差减小的方向调整）
        weight -= learning_rate * weight_adjust
        bias -= learning_rate * bias_adjust
        
        avg_abs_error=sum(abs(e) for e in errors)/len(errors)
        # 3.5 定期打印训练进度（每100轮）
        if (epoch + 1) % 100 == 0:
            print(f"第{epoch+1}轮训练：W={weight:.3f}, b={bias:.3f},误差为{[round(e,4) for e in errors]},平均绝对误差为{avg_abs_error:.3f},学习率为{learning_rate}")
    
    return weight, bias

# 4. 准备训练数据（面积和对应房价）
house_areas = [80, 120, 90]    
house_prices = [420, 630, 473] 
nom_areas=nomolize(house_areas)
nom_prices=nomolize(house_prices)
print("标准化后的输入为:",nom_areas,nom_prices)

# 5. 训练模型（学习房价规律）
trained_weight, trained_bias = train_neuron(
    areas=nom_areas,
    prices=nom_prices,
    epochs=10000,        # 训练10000轮
    initial_lr=0.01 # 学习率（步长）
)

# 6. 验证训练结果
print("="*50)
print(f"最终学习到的参数：W={trained_weight:.3f}, b={trained_bias:.3f}")

# 7. 预测新数据（100平方米房价输入也要标准化）
predicted_price_nom = neuron_predict((100-min(house_areas))/(max(house_areas)-min(house_areas)), trained_weight, trained_bias)


#8.标准化预测结果反标准化
predicted_price=predicted_price_nom*(max(house_prices)-min(house_prices))+min(house_prices)
print(f"\n预测100平方米房价：{predicted_price:.0f}万元（理论值：504万元）")



