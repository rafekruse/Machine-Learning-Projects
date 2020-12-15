import numpy as np
import time
import pickle
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable

def create_loader(x, y, sampler, batch_size):
    my_dataset = torch.utils.data.TensorDataset(torch.stack([torch.Tensor(i) for i in x]),torch.Tensor(y))
    return(torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, sampler=sampler, num_workers=2))
    
def trainNet(net, x, y, hard, train_sampler, val_sampler, optimizer, loss_function, batch_size, n_epochs, learning_rate):
        
    print("----params----")
    print("hard training : ", hard)
    print("batch_size : ", batch_size)
    print("epochs : ", n_epochs)
    print("learning_rate : ", learning_rate)
    print("--------------")
        
    t_load = create_loader(x, y, train_sampler, batch_size)

    n_batches = len(t_load)   
    total_start_time = time.time()   
    
    for epoch in range(n_epochs):
        
        current_loss = 0.0
        print_frequency = n_batches // 5
        total_loss = 0
        
        for i, data in enumerate(t_load, 0):
            
            inputs, labels = data 
            inputs, labels = Variable(inputs), Variable(labels)
            
            optimizer.zero_grad()
            
            outputs = net(inputs)
           

            loss_size = loss_function(outputs, labels.long())
            loss_size.backward()
            optimizer.step()
            
            current_loss += loss_size.data.item()
            total_loss += loss_size.data.item()
            
            if (i + 1) % (print_frequency + 1) == 0:
                print("Epoch {}#, {:d}% \t train_loss: {:.3f}".format(
                        epoch+1, int(100 * (i+1) / n_batches), current_loss / print_frequency))
                current_loss = 0.0
            
        v_loss = 0
        v_load = create_loader(x, y, val_sampler, batch_size)
        for inputs, labels in v_load:
            
            inputs, labels = Variable(inputs), Variable(labels)
            
            v_outputs = net(inputs)        
            max_results = np.zeros(len(v_outputs))
            for i in range(0, len(v_outputs)):
                max_results[i] = np.argmax(v_outputs.detach().numpy()[i])            
            print("Percent Error:" + str(np.mean(max_results != labels.long().detach().numpy()) * 100) + "%")
            
            val_loss_size = loss_function(v_outputs, labels.long())
            v_loss += val_loss_size.data.item()
            
        print("Val loss = {:.3f}".format(v_loss / len(v_load)))
        
    print("Done. {:.3f}s".format(time.time() - total_start_time))
    
def load_pkl(fname):
    with open(fname,'rb') as f:
        return pickle.load(f)
    
def scale(data, x, y):
    for p in range(0, len(data)):
        data[p] = [x for x in data[p] if any(x)]  
        
        for i in range(len(data[p][1]) - 1, - 1, -1):
            count = 0
            for j in data[p]: 
                if j[i] == 1:
                    count += 1
            if count == 0: 
                for j in range (len(data[p]) -1,-1,-1):                     
                    if isinstance(data[p][j], np.ndarray):
                        data[p][j] = np.delete(data[p][j], i)
                    else:
                        del data[p][j][i]           
        
        output = np.zeros((54,54))
        x_pro = len(data[p]) / float(x)
        y_pro = len(data[p][0]) / float(y)
        for i in range(0,x):
            for j in range(0,y):  
                if(data[p][min(len(data[p]) - 1, round(i * x_pro))][min(len(data[p][0])-1, round(j * y_pro))] == 1):
                    output[i][j] = 1
        data[p] = output     
        if p % 100 == 0:
            print("{:d}% ".format(int(100 * (p) / len(data))))

class easyCNN(torch.nn.Module):
    
    size = 0
    
    def __init__(self, size):
        super(easyCNN, self).__init__()
        
        self.size = size
        self.conv1 = torch.nn.Conv2d(1, 18, kernel_size=3, stride=1, padding=1)      
        self.conv2 = torch.nn.Conv2d(18, 64, kernel_size=3, stride=1, padding=1)
                      
        self.fc1 = torch.nn.Linear(64 * self.size * self.size, 128)        
        self.fc2 = torch.nn.Linear(128, 32)        
        self.fc3 = torch.nn.Linear(32, 2)
    
    
    def forward(self, x):
        
        y = self.conv1(x)
        x = torch.nn.functional.relu(y)        
        y = self.conv2(x)
        x = torch.nn.functional.relu(y)
        
        x = x.view(-1, 64 * self.size * self.size)
        
        x = torch.nn.functional.relu(self.fc1(x))        
        x = torch.nn.functional.relu(self.fc2(x))        
        x = self.fc3(x)
        return(x)

class hardCNN(torch.nn.Module):
    
    size = 0
    
    def __init__(self, size):
        super(hardCNN, self).__init__()
        
        self.size = size
        self.conv1 = torch.nn.Conv2d(1, 18, kernel_size=3, stride=1, padding=1)      
        self.conv2 = torch.nn.Conv2d(18, 64, kernel_size=3, stride=1, padding=1)
                      
        self.fc1 = torch.nn.Linear(64 * self.size * self.size, 128)        
        self.fc2 = torch.nn.Linear(128, 36)        
        self.fc3 = torch.nn.Linear(36, 9)
    
    
    def forward(self, x):
        
        y = self.conv1(x)
        x = torch.nn.functional.relu(y)        
        y = self.conv2(x)
        x = torch.nn.functional.relu(y)
        
        x = x.view(-1, 64 * self.size * self.size)
        
        x = torch.nn.functional.relu(self.fc1(x))        
        x = torch.nn.functional.relu(self.fc2(x))        
        x = self.fc3(x)
        return(x)
    

def load_data(data_path, labels_path, imageDimensions):
    X = load_pkl(data_path)
    scale(X,imageDimensions,imageDimensions)
    X = np.array([np.array(xi) for xi in X])
    y = np.load(labels_path)
    
    
    X = np.stack(X)
    
    if(len(X.shape) < 4):
        X = np.expand_dims(X, axis = 1)

    if(len(y.shape) > 1):
        y = np.concatenate(y)

    y = [x - 1 for x in y]
    print(np.unique(y))
    return X,y


def train(hard = False, data_path = "easy_data.pkl", labels_path = "easy_labels.npy", imageDimensions = 54, n_train_samples = 1400, n_val_samples = 200, batch_size = 64, n_epochs = 10,learning_rate = 0.001, save_CNN = False, cnn_save_path = 'model.pt'):
    print("Initializing...")#Just to indicate train is running properly
    
    model = hardCNN(imageDimensions) if hard else easyCNN(imageDimensions)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_sampler = SubsetRandomSampler(np.arange(n_train_samples, dtype = np.int64))
    val_sampler = SubsetRandomSampler(np.arange(n_train_samples, dtype = np.int64))
    print("Processing data...")
    X, y = load_data(data_path, labels_path, imageDimensions)
    trainNet(model, X, y, hard, train_sampler, val_sampler, optimizer, loss_function, batch_size=batch_size, n_epochs = n_epochs, learning_rate = learning_rate)

    if save_CNN:
        torch.save(model, cnn_save_path)

