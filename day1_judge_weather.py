weather=input('输入今天天气：（晴/雨/雪）：')
if weather=='晴':
    outfit='T恤+帽衫'
elif weather=='雨':
    outfit='雨衣+雨靴'
elif weather=='雪':
    outfit='羽绒服+棉鞋'
else :
    outfit='卫衣+牛仔裤'

temperature=int(input("输入今日温度（摄氏度）："))
if temperature>=30:
    outfit+="+冰袖"
print("推荐装备：",outfit)