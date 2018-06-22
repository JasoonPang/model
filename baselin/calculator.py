
# coding: utf-8

import re 


"""
本脚本只进行简单的加减乘除计算，目的是熟悉正则匹配规则
"""

# 乘除模块

def mul_div(s):
    while ('*' in s) or ('/' in s):        
        str_s = re.search('(\d+\.?\d*)+([*/])+(\d+\.?\d*)+',s) # 匹配出*/和其左右的数字
        if str_s.group(2) == '*':
            mul_div = float(str_s.group(1)) * float(str_s.group(3))  
        else:
            mul_div = float(str_s.group(1)) / float(str_s.group(3))  

        mul_div = '{}'.format(mul_div)
        s = s.replace(str_s.group(), mul_div) # 用计算后的结果替换正则匹配的结果，以下同
    s = nosub(s)
    return s


# 加减模块
    
def add_sub(s):
    while ('+' in s) or ('-' in s) and s[0] != '-':
        str_s = re.search('(\d+\.?\d*)+([+-])(\d+\.?\d*)+',s) # 匹配出+-和其左右的数字
        if s[0] != '-':           # 判断是否是-作为计算式子的开头
            if str_s.group(2) == '+':
                add_sub = float(str_s.group(1)) + float(str_s.group(3))
            else:
                add_sub = float(str_s.group(1)) - float(str_s.group(3))
        else:            
            if str_s.group(2) == '+':
                add_sub = float(str_s.group(1)) - float(str_s.group(3))
            else:
                add_sub = float(str_s.group(1)) + float(str_s.group(3))
        s = s.replace(str_s.group(),'{}'.format(add_sub))
    return s

# 替换算符接减号的情况
def nosub(s):
    while re.search('[/*]-',s):
        str_s = re.search('([+-])(\d+\.?\d*)+([/*])-(\d+\.?\d*)+',s)   # 匹配出*-，/-和其左右的数字
        if str_s.group(1) == '+': # 以下为根据str_s的开头决定替换符号
            s = s.replace(str_s.group(), '-' + str_s.group(2) + str_s.group(3) + str_s.group(4))
        else:
            s = s.replace(str_s.group(), '+' + str_s.group(2) + str_s.group(3) + str_s.group(4))
    s = s.replace('+-','-')
    s = s.replace('--','+')
    return s


# 抽取括号里的内容并计算

def bracket(s):
    while '(' in s:    # 只要有括号，就重复执行
        s = nosub(s)
        #print(s)
        str_s = re.search('\([^()]+\)',s).group()        # 匹配出最里层括号内的字符串
        s2 = str_s[1:-1]
        #print(s2)
        s2 = mul_div(s2)
        #print(s2)
        s2 = add_sub(s2)
        #print(s2)
        s = s.replace(str_s,'{}'.format(s2))
    s = nosub(s)
    return s   


s = '1 + (6 - 12 * 2 + 4 - 5 * (10 / 2 - (4 + 21 - 5 * 3 / 4 - 19) * 2 + 11)) + 1'
flag = re.search('[^0-9-+/*(). ]', s)  # 匹配非计算字符
if flag:
    print('{} is invalid input,please check it.'.format(flag.group()))
else:
    s_tmp = s.replace(' ','')
    s_tmp = bracket(s_tmp)
    s_tmp = mul_div(s_tmp)
    s_tmp = add_sub(s_tmp)
    s = s_tmp
print(s)









