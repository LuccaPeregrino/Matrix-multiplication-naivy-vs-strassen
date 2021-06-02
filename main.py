import time
import random
import numpy as np
import matplotlib.pyplot as plt
from self import self

def matrix_generator(order,interval_num,result):
	self.order = order
	self.interval_num = interval_num
	shape = (self.order,self.order)
	M = np.zeros(shape)	

	if result == False:
		M = np.random.randint(self.interval_num[0], self.interval_num[1], size=(self.order, self.order))
	#print(f"{M}\n")
	
	return M

def basic_mult_matrix(A,B):
	result = matrix_generator(order,0,True)
	
	for i in range(len(A)):
	   for j in range(len(A)):
		   for k in range(len(B)):
			   result[i][j] += X[i][k] * Y[k][j]
			   
	return result
	
def strassen_mult_matrix(x, y):
	
    row, col = x.shape
    row_2, col_2 = row//2, col//2
    a, b, c, d = x[:row_2, :col_2], x[:row_2, col_2:], x[row_2:, :col_2], x[row_2:, col_2:]
    e, f, g, h = y[:row_2, :col_2], y[:row_2, col_2:], y[row_2:, :col_2], y[row_2:, col_2:]
  
    if len(x) == 1:
        return x * y

    p1 = strassen_mult_matrix(a, f - h)  
    p2 = strassen_mult_matrix(a + b, h)        
    p3 = strassen_mult_matrix(c + d, e)        
    p4 = strassen_mult_matrix(d, g - e)        
    p5 = strassen_mult_matrix(a + d, e + h)        
    p6 = strassen_mult_matrix(b - d, g + h)  
    p7 = strassen_mult_matrix(a - c, e + f)  
  
    c11 = p5 + p4 - p2 + p6  
    c12 = p1 + p2           
    c21 = p3 + p4            
    c22 = p1 + p5 - p3 - p7  
  
    aux1 = np.hstack((c11, c12))
    aux2 = np.hstack((c21, c22))
    result = np.vstack((aux1,aux2))
  
    return result

if __name__ == '__main__':

	f = open('dados .txt', 'r')
	max_order = int(f.readline())
	num_matrix = int(f.readline())
	aux = (f.readline())
	interval_num =  aux.split()
	orders = []
	orders2 = []
	times = []
	times2 = []
	time1_aux = 0
	time2_aux = 0
	
	for x in range(max_order):
		order = 2**(x+1)
		for y in range(int(num_matrix/2)):
			X = matrix_generator(order,interval_num, False)
			Y = matrix_generator(order, interval_num, False)
			
			inicio = time.thread_time()					
			result = basic_mult_matrix(X,Y)
			fim = time.thread_time()
			print(f"Naivy \n {result} \n")
			time1_aux += (fim-inicio)
			
			inicio2 = time.thread_time()
			result2 = strassen_mult_matrix(X,Y)
			fim2 = time.thread_time()
			print(f"Strassen \n {result2}\n")
			time2_aux += (fim2-inicio2)
		
		print(f"Ordem 2^{x+1} finalizada" )
		times.append(time1_aux/int(num_matrix/2))
		times2.append(time2_aux/int(num_matrix/2))
		orders.append(order)
		
	print("naivy")
	print(times)
	print("strassen")
	print(times2)	
	plt.plot(orders,times,label = "naivy")
	plt.plot(orders,times2,label = "strassen")
	plt.xlabel('Order')
	plt.ylabel('Time(seconds)')
	plt.legend()
	plt.show()
	
	
