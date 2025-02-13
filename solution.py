  import pandas as pd
  from sklearn.model_selection import train_test_split
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.linear_model import PassiveAggressiveClassifier
  from sklearn.metrics import accuracy_score, confusion_matrix
  import matplotlib.pyplot as plt
  import seaborn as sns

   # Загрузка данных
df = pd.read_csv('/content/fake_news.csv')

# Разделение данных на признаки и метки
X = df['text']
y = df['label']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Преобразование текстовых данных в TF-IDF матрицу
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Инициализация и обучение модели
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Предсказание на тестовой выборке
y_pred = pac.predict(tfidf_test)

# Оценка точности
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {score * 100:.2f}%')


# Построение матрицы ошибок
cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])

# Визуализация матрицы ошибок
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
