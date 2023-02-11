# -*- coding: utf-8 -*-
"""
time: 2021.12.18
@author: 吴才朋
"""

import re
import requests
from urllib import error
from bs4 import BeautifulSoup
import os

num = 0
numPicture = 0
file = ''
List = [] # 存储图片地址


# A是会话对象，url是网址部分
def Find(url, A):
    global List
    print('正在检测图片总数，请稍等.....')
    t = 0
    s = 0
    while t < 3000:		# 这里可以改变搜索到的照片数量
        Url = url + str(t) # t为页数
        try:
            # 获取访问回来的结果
            Result = A.get(Url, timeout=7, allow_redirects=False)
        except BaseException:
            t = t + 60 # 每一页60张相片
            continue
        else:
            # 获得结果的文本
            result = Result.text
            # 先利用正则表达式找到图片url
            pic_url = re.findall('"objURL":"(.*?)",', result, re.S)
            s += len(pic_url)  # 图片的个数
            if len(pic_url) == 0:  # 如果没有图片，则退出
                break
            else:
                # 把列表加入图片的地址
                List.append(pic_url)
                t = t + 60  # 每一页60张相片
    return s # 返回图片的个数

def recommend(url):
    Re = []
    try:
        html = requests.get(url, allow_redirects=False) # 获取图片的地址
    except error.HTTPError as e: # 如果访问有错误，则退出
        return
    else:
        html.encoding = 'utf-8' # 设置中文
        bsObj = BeautifulSoup(html.text, 'html.parser')
        div = bsObj.find('div', id='topRS') # 查找
        if div is not None:
            listA = div.findAll('a') # 查找html中所有a的标签
            for i in listA:
                if i is not None:
                    Re.append(i.get_text()) # 添加入列表中
        return Re


def dowmloadPicture(html, keyword):
    global num # 定义一个全局变量
    pic_url = re.findall('"objURL":"(.*?)",', html, re.S)  # 先利用正则表达式找到图片url
    print('找到关键词:' + keyword + '的图片，即将开始下载图片...')
    for each in pic_url: # 迭代下载
        print('正在下载第' + str(num + 1) + '张图片，图片地址:' + str(each))
        try:
            if each is not None:
                pic = requests.get(each, timeout=7) # 获取地址
            else:
                continue
        except BaseException:
            print('错误，当前图片无法下载')
            continue
        else:
            string = file + r'\\' + keyword + '_' + str(num) + '.jpg' # 下载文件到本地的地址
            fp = open(string, 'wb') # 打开文件
            fp.write(pic.content)   # 写入数据
            fp.close() # 关闭文件
            num += 1 # 图片个数加一
        if num >= numPicture: # 如果达到需要的图片，则退出
            return

# 主函数入口
if __name__ == '__main__':
    # header是服务器以HTTP协议传HTML资料到浏览器前所送出的字串
    headers = {
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0',
        'Upgrade-Insecure-Requests': '1'
    }
    # 初始化requests.sessionh()会话对象
    A = requests.Session()
    A.headers = headers

    # 输入需要搜索的名字
    word = input("请输入搜索关键词(鱼类名等等): ")
    # add = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=%E5%BC%A0%E5%A4%A9%E7%88%B1&pn=120'
    # 百度搜索图片的网址，通用的
    url = 'https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + word + '&pn='

    # 调用查询函数，返回图片个数tot，获得图片的地址放入List中
    tot = Find(url, A)
    Recommend = recommend(url)  # 记录相关推荐
    print('经过检测%s类图片共有%d张' % (word, tot))
    numPicture = int(input('请输入想要下载的图片数量： '))
    file = input('请建立一个存储图片的文件夹，输入文件夹名称即可：')
    y = os.path.exists(file) # 判断文件是否为空
    if y == 1:
        str1=input('该文件已存在，是否保存在已存在文件中(y/n)：')
        if(str1=='N' or str1=='n'):
            file = input('请建立一个存储图片的文件夹，)输入文件夹名称即可:')
            os.mkdir(file)
    else:
        os.mkdir(file) # 建立文件
    t = 0
    tmp = url
    while t < numPicture: # numPicture为图片个数
        try:
            url = tmp + str(t)

            # 获取访问回来的结果
            result = A.get(url, timeout=10, allow_redirects=False)
        except error.HTTPError as e:
            print('网络错误，请调整网络后重试')
            t = t + 60 # 每一页60张相片
        else:
            dowmloadPicture(result.text, word)
            t = t + 60 # 每一页60张相片
    print("下载成功！！")
    for re in Recommend:
        print(re, end='  ')