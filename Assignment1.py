import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import sklearn.datasets

# Load do ficheiro
train_data = pd.read_csv(r'C:/Users/duart/OneDrive/Ambiente_de_Trabalho/Master_Analysis_Engineering_Big_Data/23-24/1st_semester/AA_ML/Kaggle_challenges/3_body_problem/3_body_problem/X_train.csv')
test_data = pd.read_csv(r'C:/Users/duart/OneDrive/Ambiente_de_Trabalho/Master_Analysis_Engineering_Big_Data/23-24/1st_semester/AA_ML/Kaggle_challenges/3_body_problem/3_body_problem/X_test.csv')

X = train_data.copy()
summary_stats = X.describe(include="all")
summary_stats.to_excel('summary_stats_nonprocessed.xlsx')

# Identify faulty rows based on the criterion (all values = 0.0 except for Id)
faulty_rows = train_data[(train_data.drop('Id', axis=1) == 0).all(axis=1)]

# Remove the faulty rows from the DataFrame
filtered_train_data = train_data[~train_data.index.isin(faulty_rows.index)]
filtered_train_data.to_csv('X_train_preprocessed.csv')

summary_stats_filtered = filtered_train_data.describe(include='all')
summary_stats_filtered.to_excel('summary_stats_processed.xlsx')




# test_data.describe(include='all')
# test_data.head()

# Definir as variáveis
t = pd.DataFrame(train_data, columns=['t'])
x1 = pd.DataFrame(train_data, columns=['x_1'])
x2 = pd.DataFrame(train_data, columns=['x_2'])
x3 = pd.DataFrame(train_data, columns=['x_3'])
y1 = pd.DataFrame(train_data, columns=['y_1'])
y2 = pd.DataFrame(train_data, columns=['y_2'])
y3 = pd.DataFrame(train_data, columns=['y_3'])
vx1 = pd.DataFrame(train_data, columns=['v_x_1'])
vx2 = pd.DataFrame(train_data, columns=['v_x_2'])
vx3 = pd.DataFrame(train_data, columns=['v_x_3'])
vy1 = pd.DataFrame(train_data, columns=['v_y_1'])
vy2 = pd.DataFrame(train_data, columns=['v_y_2'])
vy3 = pd.DataFrame(train_data, columns=['v_y_3'])

# Verificar shape das variaveis
# lista_var = [t, x1, x2, x3, y1, y2, y3, vx1, vx2, vx3, vy1, vy2, vy3]
# for i in lista_var:
#     print(np.shape(i))

# t_test = pd.DataFrame(test_data, columns=['t'])
# print(np.shape(t_test))

# quando fores correr o código pela segunda vez seleciona esta linha e faz
# ctrl + F9, assim ele corre a partir daqui e fica mais rapido porque não
# considera as linhas que eu escrevi para descrever as variavies (mas como o
# correste normalmente da primeira vez elas ja estao guardadas na memoria)


# X.head()

# y = train_data.target.values.ravel()
# train_data.target.head()


def trajetorias_3corpos(graph, row):
    rows = row
    x1_rows = x1.head(n=rows)
    y1_rows = y1.head(n=rows)
    x2_rows = x2.head(n=rows)
    y2_rows = y2.head(n=rows)
    x3_rows = x3.head(n=rows)
    y3_rows = y3.head(n=rows)
    # vx1_rows = y3.head(n=rows)
    # vx2_rows = y3.head(n=rows)
    # vx3_rows = y3.head(n=rows)
    # vy1_rows = y3.head(n=rows)
    # vy2_rows = y3.head(n=rows)
    # vy3_rows = y3.head(n=rows)

    if graph == 'scatter':
        plt.scatter(x1_rows, y1_rows)
        plt.scatter(x2_rows, y2_rows)
        plt.scatter(x3_rows, y3_rows)
    if graph == 'plot':
        plt.plot(x1_rows, y1_rows)
        plt.plot(x2_rows, y2_rows)
        plt.plot(x3_rows, y3_rows)
    else:
        print('wrong graph name')

    plt.show()


trajetorias_3corpos('plot', 150000)
