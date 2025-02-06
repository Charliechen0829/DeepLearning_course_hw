# HW-8-陈宸

#### Q1.RNN QA系统 -- 代码理解

调整模型learning rate = 0.001，训练模型。

![image-20241126221302368](C:\Users\33030\AppData\Roaming\Typora\typora-user-images\image-20241126221302368.png)

 python generate.py -q "Name a good sport." --cuda 

![image-20241126221524697](C:\Users\33030\AppData\Roaming\Typora\typora-user-images\image-20241126221524697.png)

designed Q&A:

q1：![image-20241126222323087](C:\Users\33030\AppData\Roaming\Typora\typora-user-images\image-20241126222323087.png)

q2：![image-20241126222953958](C:\Users\33030\AppData\Roaming\Typora\typora-user-images\image-20241126222953958.png)

q3：![image-20241126223326141](C:\Users\33030\AppData\Roaming\Typora\typora-user-images\image-20241126223326141.png)

q4：![image-20241126223651819](C:\Users\33030\AppData\Roaming\Typora\typora-user-images\image-20241126223651819.png)

q5：![image-20241126224117044](C:\Users\33030\AppData\Roaming\Typora\typora-user-images\image-20241126224117044.png)

rediculous Q&A:

q1：![image-20241126223834920](C:\Users\33030\AppData\Roaming\Typora\typora-user-images\image-20241126223834920.png)

q2：![image-20241126221740765](C:\Users\33030\AppData\Roaming\Typora\typora-user-images\image-20241126221740765.png)

q3：![image-20241126224329256](C:\Users\33030\AppData\Roaming\Typora\typora-user-images\image-20241126224329256.png)

q4：![image-20241126224456533](C:\Users\33030\AppData\Roaming\Typora\typora-user-images\image-20241126224456533.png)

q5：![image-20241126224606628](C:\Users\33030\AppData\Roaming\Typora\typora-user-images\image-20241126224606628.png)

原因：回答较好的问题与训练数据中相似程度较高，使得模型能够准确地回答，且问题较为具体、明确，答案也较为标准和固定。而离谱的问题从语义上较为复杂或抽象，模型难以理解，或是如q3（视野较好的天气）只是删去了not，但答案却与不删去的完全相同，一定程度上也说明了模型存在类似过拟合的情况。

**train.py 中的 TODO 注释解释：**

**Explain1：**在前向传播过程中，在模型预测期望输出之前，去掉了目标序列表示序列结束的特殊标记（如 [END]），以避免模型仅学习预测到序列的末尾字符。   

**Explain2:** 为了排除提示序列开始的特殊标记（如 [BEG]），以确保模型学会在生成答案时从正确的起始点开始。 

**Explain3:** 每个问题和其答案长度不同，数据的批处理存在不便（例如需要使用填充），故可利用累计梯度的办法使模型能够有效处理不同长度的序列，由于难以将它们放入批处理中，因此在每个（问题，答案）对上都需要累积梯度，最终执行一次优化步骤。

**Explain4:** 两种计算方法是等价的， 而方法一中累计梯度更新更加适合变长问题序列的情况，方法二批量处理虽能更好的利用并行计算资源，但不适用于变长序列的情况。

**model.py 中的 TODO 注释解释：**

**Explain1:**分阶段的处理方式能够使模型更容易训练和调试，且使构造出的问答系统使用具有不同结构的的编码器和解码器，进一步使得模型具有更高的灵活性。例如在作业的样例中使用了不同的网络结构如GRU、LSTM等分别作为编码器和解码器。

**Explain2:**在解码阶段，根据编码阶段生成的上下文向量来逐步生成答案。这个过程可以类比为从一个初始输入开始，逐步生成下一个字符，直到生成整个答案或者遇到End Token。

**generate.py 中的 TODO 注释解释：**

**Explain1:** [BEG]token 作为解码器的初始输入，表明生成序列的开始，用于帮助模型识别从哪里开始生成答案。

**Explain2:** 编码过程中，首先将输入问句转换成词嵌入向量，使得输入数据可以被 RNN 处理。再使用LSTM 或 GRU编码器将输入序列编码成一个上下文向量。后将编码器的最后的隐状态作为解码器的初始隐状态，使得解码器在生成答案时，可以利用输入问句的上下文信息。

编码阶段还处理了输入序列的变长问题，将变长的输入编码成固定长度的上下文向量。

**训练阶段和推理阶段的不同执行方式：**

训练阶段：通过数据对数据迭代训练，使模型进行前向传播与反向传播，累计梯度后更新模型参数。

推理阶段：首先通过编码器对输入序列进行编码，并使用最后的隐藏状态来初始化解码器初始状态。基于训练阶段的模型的预测，迭代生成下一个预测的结果，直到遇到终止标记，最后将模型推理得到的预测序列转换回文本数据，生成最终的回答字符串。

**Word-Based QA structure：**

![image-20241127092010068](C:\Users\33030\AppData\Roaming\Typora\typora-user-images\image-20241127092010068.png)



#### Q2.LSTM 训练sin(x)函数拟合和补全预测

```python
for p in range(predict_len):
    # TODO: use last hidden and output to feed lstm, update h_t+1, write a line of code here
    output, (hn, cn) = self.lstm(last_output, (hn, cn))

    # TODO: update y_t+1 as last_output, write a line of code here
    last_output = self.linear(output)
```



out/predict14.png

![predict14](C:\Users\33030\PycharmProjects\pythonProject\deep_learning\hw08_RNN_QA\out\predict14.png)

补全代码后能够较好的拟合正弦函数。

**训练过程与推理过程拟合任务的区别：**

​	训练阶段主要是构建优化模型参数后的模型，每一步训练中，通过前向传播计算输出，同时计算损失并进行反向传播来更新模型参数，使模型能够准确预测时间序列中的下一个值。

​	推理阶段的则是使用训练阶段调优参数后的模型对未来的时间序列进行预测。与训练阶段不同，推理阶段不再更新模型参数，而是根据已有的输入数据生成未来的预测值，初始输入input_prefix通过 LSTM 网络，计算出隐藏状态hn和细胞状态cn，以及最后的输出last_output作为下一步的输入，循环predict_len次，逐步预测未来的值。在每一步预测中，前一次预测的输出被用作下一次预测的输入，不断更新隐藏状态和细胞状态。

**使用for循环的原因：**

​	结合上文中推理过程的拟合任务，我们需要确保每一步的预测都是逐步进行的，每一步都依赖于前一步的last_output来更新隐藏状态和细胞状态，使每一步的预测结果都基于当前最准确的模型状态。

**与Q1的处理结构的不同：**

​	Q1 处理问答任务，需要encoder-decoder结构根据输入查询生成响应。而相对的，Q2 处理时间序列预测，利用 LSTM 模型预测正弦波序列中的未来值，侧重于顺序预测，而不像 Q1 中的 采用明确的e-d结构，其结构可能更偏向于decoder-only。
