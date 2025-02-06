# HW-6-陈宸

#### Q1.卷积核的手动计算

**证明A、B等价：**

设A中的三个矩阵分别为a,b,c，B中的两个矩阵分别为a,d，A=(a*b)\*c，B=a\*d

观察到d为空间可分离卷积，d=(b*c)，代入原式则有B=a\*(b\*c)，根据卷积运算的结合律，(a\*b)\*c=a\*(b\*c)

A等价于B得证。

**计算结果：**

![image-20241104103920839](C:\Users\33030\AppData\Roaming\Typora\typora-user-images\image-20241104103920839.png)

**算法复杂度：**

A算式的第一步运算对输入矩阵的每行每列都做三次乘法后加和，得到与输入矩阵维数一样的输出矩阵，再做类似的乘法运算，复杂度为$O(3N^2+3N^2)=O(6N^2)$ 。对于kernel为k\*k的情况，复杂度为$O(2k*N^2)$ 

B算式对输入矩阵的每行每列都做9此乘法后加和。复杂度为$O(9N^2)$ 。对于kernel为k\*k的情况，复杂度为$O(k^2*N^2)$ 

#### Q2.平移图片

调节卷积核尺寸如下。

```python
# separable translation kernel
H_row = torch.zeros(1, 200).float()
H_row[0, 199] = 1
H_col = torch.zeros(200, 1).float()
H_col[199, 0] = 1
H_row = H_row.expand(3, 1, 1, 200)
H_col = H_col.expand(3, 1, 200, 1)
```

​	在我实现的代码中，将列卷积核的最后的一个值置为1，其余全部置为0，将其扩张至rgb三个通道，则在卷积运算中，卷积核镜像翻转后对原图做滑动窗口运算，使整体图片向上平移了200个单位像素。

​	将行卷积核的最后的一个值置为1，其余全部置为0，将其扩张至rgb三个通道，对原图做滑动窗口运算后使整体图片向左平移了200个单位像素。

平移后的图像如下。

![su7_ultra_move](C:\Users\33030\PycharmProjects\pythonProject\deep_learning\hw06\su7_ultra_move.png)

程序输出：

![image-20241102171005640](C:\Users\33030\AppData\Roaming\Typora\typora-user-images\image-20241102171005640.png)
