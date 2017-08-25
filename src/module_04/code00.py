import xlrd, numpy as np, matplotlib.pyplot as plt

# opent the xls file for reading
xlsfile = xlrd.open_workbook('fire_theft.xls', encoding_override='utf-8')

# there can be many sheets in xls document
sheet = xlsfile.sheet_by_index(0)

# ask the sheet for each row of data explicitly
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])

# compute the number of samples
num_samples = data.shape[0]

# print out some metadata
print('data shape: {}'.format(data.shape))

# blab about the number of samples in the data
print('the number of samples in the data: {}'.format(num_samples))

# column one is incidence of fire
X = data[:,0]

# column two is incidence of theft
Y = data[:,1];

# create linear design matrix
A = np.vstack((X,np.ones_like(X))).T

# solve linear system
b = np.linalg.lstsq(A, Y)[0]

# blab about the solution
print('the model values are: {}'.format(b))

# plot the original data
plt.plot(X,Y,'o',label='data', markersize=3)

# add the estimated linear model
plt.plot(X, b[0]*X + b[1], 'r', label='model')

# configure the plot
plt.legend(); plt.grid(True)

# we're done
plt.show()

'''
    The numpy way ...
'''