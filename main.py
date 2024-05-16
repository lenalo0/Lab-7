import numpy as np
from time import perf_counter

# Генерация данных
size = 10**6
list1 = np.random.rand(size).tolist()
list2 = np.random.rand(size).tolist()
array1 = np.array(list1)
array2 = np.array(list2)

# Поэлементное перемножение списков
start = perf_counter()
result_list = [a * b for a, b in zip(list1, list2)]
end = perf_counter()
print(f"Время выполнения для стандартных списков: {end - start:.6f} секунд")

# Поэлементное перемножение массивов NumPy
start = perf_counter()
result_array = np.multiply(array1, array2)
end = perf_counter()
print(f"Время выполнения для массивов NumPy: {end - start:.6f} секунд")

import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
filename = 'data2.csv'
data = pd.read_csv(filename)

# Извлечение данных из столбца 4
column_data = data.iloc[:, 3]

# Построение гистограммы
plt.figure(figsize=(10, 6))
plt.hist(column_data, bins=16, edgecolor='black')
plt.title('Гистограмма для столбца 4')
plt.xlabel('Значения')
plt.ylabel('Частота')
plt.show()

# Вычисление среднеквадратичного отклонения
std_dev = np.std(column_data)
print(f"Среднеквадратичное отклонение: {std_dev:.6f}")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Определение интервалов
x = np.linspace(-3*np.pi, 3*np.pi, 500)
y = np.cos(x)
z = x / np.sin(x)

# Построение 3D графика
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x, y, z, label='z = x / sin(x), y = cos(x)')
ax.set_title('Трёхмерный график функции')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()
