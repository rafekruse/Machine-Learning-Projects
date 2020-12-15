import numpy as np
import time
import pickle
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable

def load_pkl(fname):
    with open(fname,'rb') as f:
        return pickle.load(f)

def save_pkl(fname,obj):
    with open(fname,'wb') as f:
        pickle.dump(obj,f)
    

def load_data(data_path, imageDimensions):
    X = load_pkl(data_path)
    scale(X,imageDimensions,imageDimensions)
    X = np.array([np.array(xi) for xi in X])
    
    X = np.stack(X)
    
    if(len(X.shape) < 4):
        X = np.expand_dims(X, axis = 1)

    return X

def test(model_path = 'model.pt',data_path = 'test_data.pkl', results_path = 'output_labels.npy', imageDimensions = 54):
    print("Initializing...")
    model = torch.load(model_path)
    print(model.eval())
    print("Processing data...")
    X = load_data(data_path, imageDimensions)

    inputs = torch.stack([torch.Tensor(i) for i in X])
    max_results = np.zeros(len(inputs))



    for i in range(0,len(inputs)):

        model_input = inputs[i].unsqueeze(0)
        model_input = Variable(model_input)

        expected_outputs = model(model_input)  

        max_results[i] = np.argmax(expected_outputs.detach().numpy()[0])  
        max_results[i] += 1

        if i % 100 == 0:
            print("{:d}% ".format(int(100 * (i) / len(inputs))))

        if max_results[i] == 9:
            max_results[i] = -1

    np.save(results_path, max_results)
      
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