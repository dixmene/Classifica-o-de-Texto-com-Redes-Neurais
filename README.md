
## Classificação de Sentimentos em Avaliações de Filmes com Redes Neurais Usando o Dataset IMDB

### 1. Carregamento e Preparação dos Dados

```python
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Carregar dataset IMDB (sentimentos positivos/negativos)
(vocab_size, maxlen) = (10000, 100)  # Limite de vocabulário e comprimento de sequência

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Padronizar o comprimento das sequências
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
```

**Contextualização:**
- **Carregamento dos Dados:** O dataset IMDB é utilizado, limitado às 10.000 palavras mais frequentes para reduzir a complexidade e garantir um bom desempenho. As avaliações são rotuladas como positivas ou negativas.
- **Padronização:** As sequências de texto são padronizadas para 100 palavras, o que facilita o processamento e treinamento da rede neural.

### 2. Definição da Arquitetura da Rede Neural

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Definir o modelo
model = Sequential()

# Camada de Embedding
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=maxlen))

# Camada LSTM
model.add(LSTM(128))

# Camada de saída - classificação binária (positivo/negativo)
model.add(Dense(1, activation='sigmoid'))

# Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Chamar o método build diretamente com o shape correto
model.build(input_shape=(None, maxlen))

# Exibir o resumo do modelo
model.summary()
```

**Contextualização:**
- **Camada de Embedding:** Converte palavras em vetores densos de 128 dimensões, permitindo que a rede compreenda melhor as relações semânticas entre palavras.
- **Camada LSTM:** Permite que o modelo capture dependências temporais no texto, crucial para entender o contexto das avaliações.
- **Camada Densa:** Produz uma saída binária (0 ou 1) usando uma função de ativação `sigmoid` para classificar as avaliações como positivas ou negativas.

### 3. Treinamento do Modelo

```python
from tensorflow.keras.callbacks import EarlyStopping

# Configurar Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Treinar o modelo
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])
```

**Contextualização:**
- **Early Stopping:** Evita overfitting interrompendo o treinamento se a perda de validação não melhorar após 2 épocas, restaurando os melhores pesos do modelo.
- **Treinamento:** O modelo é treinado por até 5 épocas, com um tamanho de lote de 64. A validação é realizada em cada época para monitorar o desempenho do modelo.

### 4. Avaliação do Modelo

```python
# Avaliar o modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc:.4f}')
```

**Contextualização:**
- **Avaliação:** Medimos a precisão do modelo no conjunto de teste para avaliar seu desempenho em dados não vistos durante o treinamento.

### 5. Resultados do Treinamento

**Resultados das Épocas:**
```plaintext
Epoch 1/5
391/391 ━━━━━━━━━━━━━━━━━━━━ 77s 181ms/step - accuracy: 0.7372 - loss: 0.5056 - val_accuracy: 0.8502 - val_loss: 0.3436
Epoch 2/5
391/391 ━━━━━━━━━━━━━━━━━━━━ 70s 180ms/step - accuracy: 0.8935 - loss: 0.2634 - val_accuracy: 0.8505 - val_loss: 0.3550
Epoch 3/5
391/391 ━━━━━━━━━━━━━━━━━━━━ 72s 185ms/step - accuracy: 0.9329 - loss: 0.1820 - val_accuracy: 0.8338 - val_loss: 0.4100
Epoch 4/5
391/391 ━━━━━━━━━━━━━━━━━━━━ 66s 168ms/step - accuracy: 0.9527 - loss: 0.1329 - val_accuracy: 0.8350 - val_loss: 0.4269
Epoch 5/5
391/391 ━━━━━━━━━━━━━━━━━━━━ 52s 132ms/step - accuracy: 0.9613 - loss: 0.1063 - val_accuracy: 0.8339 - val_loss: 0.5327
```

**Acurácia de Teste Final:**
```plaintext
Test Accuracy: 0.8339
```

### 6. Melhor Desempenho com Dropout

```python
from tensorflow.keras.layers import Dropout

# Adicionar Dropout
model.add(Dropout(0.5))  # Drop 50% of the neurons

# Treinar com Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])
```

**Resultados das Épocas com Dropout:**
```plaintext
Epoch 1/5
391/391 ━━━━━━━━━━━━━━━━━━━━ 59s 150ms/step - accuracy: 0.9733 - loss: 0.0774 - val_accuracy: 0.8265 - val_loss: 0.6068
Epoch 2/5
391/391 ━━━━━━━━━━━━━━━━━━━━ 85s 156ms/step - accuracy: 0.9706 - loss: 0.0823 - val_accuracy: 0.8293 - val_loss: 0.7413
Epoch 3/5
391/391 ━━━━━━━━━━━━━━━━━━━━ 60s 155ms/step - accuracy: 0.9869 - loss: 0.0440 - val_accuracy: 0.8224 - val_loss: 0.6469
```

### 7. Visualizações

**Gráfico de Acurácia:**
```python
import matplotlib.pyplot as plt

# Plotar a acurácia de treino e validação
plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.show()
```

**Gráfico de Perda:**
```python
# Plotar a perda de treino e validação
plt.plot(history.history['loss'], label='Perda de Treinamento')
plt.plot(history.history['val_loss'], label='Perda de Validação')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/153b27c6-39b2-416a-a7a8-f0efceac3760)
![image](https://github.com/user-attachments/assets/6ee7dd21-1721-4c28-8c14-0c6b4149e2ce)

### 8. Análise dos Resultados

**Histórico de Treinamento e Validação:**

```python
import pandas as pd

# Criar um DataFrame com o histórico de treinamento
history_df = pd.DataFrame(history.history)

# Exibir o histórico de treinamento e validação
print("Histórico de Treinamento e Validação:")
print(history_df)

# Exibir o melhor desempenho da validação
best_epoch = history_df['val_loss'].idxmin()
best_val_loss = history_df['val_loss'].min()
best_val_accuracy = history_df['val_accuracy'].iloc[best_epoch]

print("\nMelhor desempenho de validação:")
print(f'Época: {best_epoch + 1}')
print(f'Perda de Validação: {best_val_loss:.4f}')
print(f'Acurácia de Validação: {best_val_accuracy:.4f}')
```

**Melhor Desempenho de Validação:**
```plaintext
Época: 1
Perda de Validação: 0.6068
Acurácia de Validação: 0.8265
```

### Conclusão

Neste projeto de classificação de texto com redes neurais, nosso objetivo foi construir um modelo para analisar e classificar sentimentos em avaliações de filmes do dataset IMDB, distinguindo entre avaliações positivas e negativas. Utilizamos uma arquitetura de rede neural composta por camadas de `Embedding`, `LSTM` e `Dense` para realizar essa tarefa.

**Principais Resultados:**
- **Acurácia Final:** O modelo obteve uma acurácia de 83,39% no conjunto de teste. Embora essa taxa de acurácia indique um desempenho relativamente bom, há espaço para melhorias, especialmente considerando a complexidade da análise de sentimentos.
- **Impacto do Dropout:** A inclusão da camada de `Dropout` foi implementada para ajudar a prevenir overfitting. A análise dos resultados após a adição do dropout mostrou que a perda de validação aumentou, sugerindo que talvez a taxa de dropout escolhida não tenha sido a ideal e ajustes adicionais podem ser necessários.

**Análise do Treinamento:**
- **Épocas de Treinamento:** O treinamento mostrou um desempenho inicial robusto, com a acurácia de treinamento aumentando consistentemente. No entanto, a acurácia de validação apresentou flutuações, o que pode indicar que o modelo estava se ajustando ao conjunto de dados específico, e pode haver uma necessidade de mais regularização ou ajustes nos hiperparâmetros.
- **Early Stopping:** O uso de early stopping ajudou a prevenir overfitting ao interromper o treinamento quando a perda de validação começou a aumentar. Isso é crucial para garantir que o modelo generalize bem para novos dados.

**Implicações:**
O projeto demonstrou a eficácia das redes neurais na tarefa de análise de sentimentos, utilizando técnicas avançadas como embeddings e LSTM. A abordagem aplicada proporciona uma base sólida para futuros projetos e aprimoramentos, e as lições aprendidas são valiosas para a construção de modelos de PLN mais precisos e eficientes.
