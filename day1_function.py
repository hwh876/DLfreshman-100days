def bmi_calculator(height,weight):
    bmi=weight/(height**2)
    if bmi<18.5:return f"bmi={bmi:.2f},偏瘦，建议增加营养"
    elif bmi<24:return f"bmi={bmi:.2f},健康，继续保持"
    else:return f"bmi={bmi:.2f},肥胖，加强运动"

#use
print(bmi_calculator(1.70,63))

def valuation(day):
    percent=day/100*100
    if percent<30: return "起步阶段，继续坚持"
    elif percent<60: return "稳步前进，量变产生质变"
    else: return "突破临界点，蜕变在即"

print(valuation(2))
