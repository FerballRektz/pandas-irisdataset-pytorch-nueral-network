# %%
import numpy as numpy
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as f
from sklearn.model_selection import train_test_split

# %%
link = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
df = pd.read_csv(link)

# %%
df.head()

# %%
df.columns

# %%
df["variety"].unique()

# %%
X_values = torch.Tensor(df.drop("variety", axis= 1).copy().values)
Y_values = torch.LongTensor(pd.Categorical(df["variety"].copy()).codes)

# %%
X_train, x_test , Y_train, y_test = train_test_split(X_values,Y_values,test_size= 0.2)

# %%
class My_Neural(nn.Module):
    def __init__(self :object , input_layer :int = 4, layer1 :int = 6, layer2 :int = 6, output_layer :int = 3) -> object:
        super().__init__()
        self.fc1 = nn.Linear(input_layer,layer1)
        self.fc2 = nn.Linear(layer1,layer2)
        self.out = nn.Linear(layer2,output_layer)
    def forward(self :object, X :object):
        x = f.relu(self.fc1(X))
        x = f.relu(self.fc2(x))
        x = self.out(x)
        return x

        

# %%
model =  My_Neural()

# %%
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr= 0.01)

# %%
epoch :int = 100
loss_list :list = list() 
for i in range(epoch):
    y_pred = model.forward(X_train)
    loss = criterion(y_pred,Y_train)
    loss_list.append(loss)
    if i % 10 == 0:
        print(f"Epoch {i}, loss: {loss}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# %%
with torch.no_grad():
    y_eval =  model.forward(x_test)
    loss_eval = criterion(y_eval,y_test)

loss_eval

# %%
correct_list :list  = list()
for index, data in enumerate(x_test):
    y_val = model.forward(data)
    test :int
    if y_val.argmax().item() == y_test[index]:
        test = 1
        correct_list.append(test)
    else:
        test = 0
        correct_list.append(test)
    print(f"index{index+1}: y_test:{y_test[index]} y_pred:{y_val.argmax().item()} correct?:{test}")
print(f"total correct: {numpy.sum(correct_list)} length: {index+1}")


