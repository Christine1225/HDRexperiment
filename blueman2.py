# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 19:06:13 2020

@author: Abigail
"""
import numpy as np
Rgain, Bgain= 559,551

with open("vi_dev_0_2592_1944_1_10bits_vc1.raw",'rb') as fr:
    long = np.frombuffer(fr.read(2592*1944*2),dtype=np.uint16)
with open("vi_dev_0_2592_1944_1_10bits_vc0.raw",'rb') as fr2:
    short = np.frombuffer(fr2.read(2592*1944*2),dtype=np.uint16)
short = np.array(short).reshape(1944,2592) 
long = np.array(long).reshape(1944,2592)    
Gb1,B1,R1,Gr1 = short[::2,::2],short[::2,1::2],short[1::2, ::2],short[1::2,1::2]
 
#print(short.max(),long.max(),short.mean(),long.mean())
u16ShortThresh,u16LongThresh = 0xf00,0xc00
Gb2,B2,R2,Gr2 = long[::2,::2],long[::2,1::2],long[1::2, ::2],long[1::2,1::2]
ratio1 = (Gb2.mean()+B2.mean()+ R2.mean()+Gr2.mean() - 200)/(Gb1.mean()+B1.mean()+R1.mean()+Gr1.mean() - 200)
ratio2 = (long.mean() - 50)/(short.mean() - 50)
##linear combination short prior
##step 1 left shift 6bit to 16bit raw
#Gb1,B1,R1,Gr1 = Gb1*32,B1*32,R1*32,Gr1*32
#Gb2,B2,R2,Gr2 = Gb2*32,B2*32,R2*32,Gr2*32
short,long = short*64,long*64
##step 2 to make equal lum
#Gb2,B2,R2,Gr2 = Gb2/ratio1,B2/ratio1,R2/ratio1,Gr2/ratio1
longc = long//ratio1
Gb1,B1,R1,Gr1 = short[::2,::2],short[::2,1::2],short[1::2, ::2],short[1::2,1::2]
Gb2,B2,R2,Gr2 = long[::2,::2],long[::2,1::2],long[1::2, ::2],long[1::2,1::2]
result = np.zeros((1944,2592),dtype = 'uint16')
result2 = np.zeros((1944,2592),dtype = 'uint16')#long.copy()
#B-result = np.zeros(1944,2592)
#R-result = np.zeros(1944,2592)
#Gr-result = np.zeros(1944,2592)
alf ,cnt = 0.5, 0
for i in range(0,result.shape[0],2):###Gb
    for j in range(0,result.shape[1],2):
        if long[i][j] > u16ShortThresh*16:
            result2[i][j] = short[i][j]
#            print("i-{},j-{}".format(i,j))
        elif long[i][j] > u16LongThresh*16:
            cnt = cnt + 1
            result2[i][j] = alf * short[i][j] + (1-alf) * longc[i][j]
#            result[i][j] = long[i][j]
#            print("short-{} longc-{} long-{}".format(short[i][j],longc[i][j],long[i][j]))
        else:
            result2[i][j] = long[i][j]
#            print("i-{},j-{},p-{}".format(i,j,long[i][j]))
#        if(result[i][j]) 
for i in range(1,result.shape[0],2):###Gr
    for j in range(1,result.shape[1],2):
        if long[i][j] > u16ShortThresh*16:
            result2[i][j] = short[i][j]
#            print("i-{},j-{}".format(i,j))
        elif long[i][j] > u16LongThresh*16:
            cnt = cnt + 1
            result2[i][j] = alf * short[i][j] + (1-alf) * longc[i][j]
#            result[i][j] = long[i][j]
#            print("short-{} longc-{} long-{}".format(short[i][j],longc[i][j],long[i][j]))
        else:
            result2[i][j] = long[i][j]
B2r = B2.copy()
R2r = R2.copy()            
for i in range(result.shape[0]//2):
    for j in range(result.shape[1]//2):
        if B2[i][j] >= u16ShortThresh*16:
            B2r[i][j] = B1[i][j]###B2[i][j] //Bgain//256
        elif B2[i][j] > u16LongThresh*16:
            B2r[i][j] = alf *B1[i][j] + (1-alf) * B2[i][j]/ratio1//Bgain//256
#        else: B2r[i][j] = B2[i][j]
for i in range(result.shape[0]//2):
    for j in range(result.shape[1]//2):
        if R2[i][j] > u16ShortThresh*16:
            R2r[i][j] = R1[i][j]###R2[i][j] //Rgain //256       
        elif R2[i][j] > u16LongThresh*16:
            R2r[i][j] = alf *R1[i][j] + (1-alf) * R2[i][j]/ratio1//Rgain //256 
#        else: R2r[i][j] = R2[i][j]  
result2[::2,1::2] = B2r
result2[1::2, ::2] = R2r       
#result10 = result//64
#print(result.max(),result.argmax(),"cnt-{}".format(cnt))
print(result2.max(),result2.argmax(),"cnt-{}".format(cnt))
with open("all-base-alpha0.5.raw",'wb') as f:
    f.write(result2)
            
#ratio2 = (Gb2.mean()+Gr2.mean() - 100)/(Gb1.mean()+Gr1.mean() - 100)
#ratio2 = (Gb2.mean()+Gr2.mean() - 100)/(Gb1.mean()+Gr1.mean() - 100)