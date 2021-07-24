# =============================================================================
# Pseudo-likelihood l1-reg prl
# =============================================================================

#%% import module
import numpy as np
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("r_start", type=int)
parser.add_argument("r_end", type=int)
args = parser.parse_args()
r_start = args.r_start
r_end = args.r_end

def print_log(content):
    log = open("log_{}_to_{}.txt".format(r_start, r_end), "a")
    print(content, file=log)
    log.close()  
    
    
#%% load data
beta = 1
lambda_l1 = 0.01
sigma = np.loadtxt("data_5w.txt")[:,1:]
M = sigma.shape[0]
P = sigma.shape[1]


#%% train function
def generate_data(sigma, index_r):
    # sigma [M,N]
    M = sigma.shape[0]
    b = (sigma[:,index_r]).reshape(M,1)
    x = np.delete(sigma, index_r, axis=1)
    return x, b
    
def write_xb(x, b, index_r):
    # write x
    x_line = x.reshape(-1, order="f")
    fx = open("train_x_{}".format(index_r), "w")
    fx.write("%%MatrixMarket matrix array real general\n")
    fx.write("{} {}\n".format(x.shape[0],x.shape[1]))
    np.savetxt(fx, x_line, delimiter="\n", newline="\n", fmt="%d")
    fx.close()
    # write b
    b_line = b.reshape(-1)
    fb = open("train_b_{}".format(index_r), "w")
    fb.write("%%MatrixMarket matrix array real general\n")
    fb.write("{} {}\n".format(b_line.size, 1))
    np.savetxt(fb, b_line, newline="\n", fmt="%d")
    fb.close()

def read_model(index_r):
    fm = open("model_{}".format(index_r), "r")
    lines = fm.readlines()
    h_r = lines[10]
    J_r = lines[11:]
    h_r = float(h_r.rstrip("\n"))
    for i in range(len(J_r)):
        J_r[i] = float(J_r[i].rstrip("\n"))
    J_r = np.array(J_r).reshape(-1)
    J_r = np.insert(J_r, index_r, 0)
    J_r = J_r/(2*beta)
    h_r = h_r/(2*beta)
    return J_r, h_r

def train(index_r):
    # generate data and write
    print_log("Generate data and write to file...")
    x, b = generate_data(sigma, index_r)
    write_xb(x, b, index_r)
    
    # trained by Boyd
    print_log("Training by boyd begin...")
    cmd = "l1_logreg_train -s -r train_x_{} train_b_{} {} model_{}"\
            .format(index_r, index_r, lambda_l1, index_r)
    os.system(cmd)
    print_log("Training finish!")
    

#%% train
time_start = time.time()
for index_r in np.arange(r_start, r_end, 1):
    print_log("Column {} is testing.".format(index_r))
    train(index_r)
time_end = time.time()
print("Time cost {:.2f}s".format(time_end-time_start))
