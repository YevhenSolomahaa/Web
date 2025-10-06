import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Створення синтетичних даних (100 точок)
np.random.seed(42)  # Для відтворюваності результатів
n = 1  # Номер варіанту (замініть на ваш варіант)

# Генерація даних
x = np.linspace(0, 10, 100)
y = n * x + np.sin(x / n) + np.random.normal(0, 0.5, 100)  # Додаємо шум

# 2. Розділення даних на навчальну (70%) і тестову (30%) вибірки
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Перетворення форми даних для scikit-learn
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

# 3. Побудова лінійної регресійної моделі
model = LinearRegression()
model.fit(x_train, y_train)

# Отримання коефіцієнтів
a = model.coef_[0]  # Коефіцієнт нахилу
b = model.intercept_  # Точка перетину

print(f"Коефіцієнт нахилу (a): {a:.4f}")
print(f"Точка перетину (b): {b:.4f}")
print(f"Рівняння регресії: y = {a:.4f}x + {b:.4f}")

# Прогнозування на тестовій вибірці
y_pred = model.predict(x_test)

# 4. Оцінка якості моделі
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nОцінка якості моделі:")
print(f"Середньоквадратична похибка (MSE): {mse:.4f}")
print(f"Середня абсолютна помилка (MAE): {mae:.4f}")
print(f"Коефіцієнт детермінації (R²): {r2:.4f}")

# 5. Побудова графіка
plt.figure(figsize=(12, 8))

# Графік навчальних даних
plt.subplot(2, 2, 1)
plt.scatter(x_train, y_train, alpha=0.7, label='Навчальні дані')
plt.plot(x_train, model.predict(x_train), color='red', linewidth=2, label='Лінія регресії')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Навчальна вибірка')
plt.legend()
plt.grid(True, alpha=0.3)

# Графік тестових даних
plt.subplot(2, 2, 2)
plt.scatter(x_test, y_test, alpha=0.7, label='Тестові дані')
plt.plot(x_test, y_pred, color='red', linewidth=2, label='Прогноз')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Тестова вибірка')
plt.legend()
plt.grid(True, alpha=0.3)

# Графік всіх даних
plt.subplot(2, 1, 2)
plt.scatter(x, y, alpha=0.7, label='Всі дані')
x_line = np.linspace(0, 10, 100).reshape(-1, 1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, color='red', linewidth=2, label='Лінія регресії')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Всі дані та лінія регресії')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6. Висновки
print("\nВИСНОВКИ:")
print("1. Лінійна регресійна модель була успішно побудована на основі синтетичних даних.")
print(f"2. Отримане рівняння: y = {a:.4f}x + {b:.4f}")
print(f"3. Коефіцієнт детермінації R² = {r2:.4f} показує, що модель пояснює приблизно {r2*100:.1f}% варіації даних.")
print("4. Значення MSE та MAE вказують на середню величину похибки прогнозу.")
print("5. Модель досить добре описує лінійну залежність, але може мати обмеження через наявність нелінійної складової (sin(x/n)).")
print("6. Для покращення точності можна розглянути використання поліноміальної регресії або інших методів.")