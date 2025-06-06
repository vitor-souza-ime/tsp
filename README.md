Aqui está uma versão aprimorada e decorada do seu `README.md`, com **negritos**, ✅ *emojis*, e seções organizadas para melhorar a legibilidade no GitHub. Também inclui sugestões de onde inserir **figuras** (como imagens geradas ou capturas de tela) para torná-lo mais visualmente atraente.

---

# 🚀 **TSP Solver: CPU vs GPU Performance Comparison**

Este projeto contém dois programas em Python para resolver o **Problema do Caixeiro Viajante (TSP)** e comparar o desempenho entre implementações **em CPU e GPU (CUDA/CuPy)**.

---

## 📁 **Estrutura do Projeto**

```
📦 tsp-solver/
├── graph.py        # Geração do grafo e solução ótima
├── tsp.py          # Comparação de desempenho: CPU vs GPU
├── tsp_dados.json  # Dados exportados (grafo, coordenadas, etc.)
├── tsp_dados.pkl   # Backup dos dados em Pickle
└── resultados_cpu_gpu.json  # Resultados das execuções
```

---

## ✨ **Funcionalidades**

### 📌 `graph.py`

* 🔹 Gera grafo completo com **N=10 vértices** (configurável).
* 🔹 Calcula distâncias euclidianas com **variação aleatória de ±20%**.
* 🔹 Plota o grafo com **NetworkX** e **Matplotlib**.
* 🔹 Encontra a **solução ótima por força bruta**.
* 🔹 Salva dados em `tsp_dados.json` e `tsp_dados.pkl`.

📸 **Sugestão de Imagem:** Inserir aqui a imagem do grafo gerado e o caminho ótimo plotado.

---

### 📌 `tsp.py`

* 🔹 Carrega dados salvos por `graph.py`.
* 🔹 Resolve o TSP com força bruta:

  * ✅ **CPU**: via `itertools.permutations`
  * ⚡ **GPU**: via **CuPy**, processando em lotes
* 🔹 Compara tempos de execução e **calcula speedup** (aceleração).
* 🔹 Plota gráficos comparativos e salva em `resultados_cpu_gpu.json`.

📸 **Sugestão de Imagem:** Inserir gráfico de barras comparando tempos CPU vs GPU.

---

## ⚙️ **Requisitos**

* ✅ Python 3.6+
* 📦 Bibliotecas:

  * `numpy`
  * `matplotlib`
  * `networkx`
  * `cupy` (opcional, para GPU)

### 💻 Hardware (para execução em GPU)

* NVIDIA GPU com suporte a CUDA
* CUDA Toolkit e drivers instalados

---

## 🧪 **Instalação**

```bash
git clone https://github.com/seu-usuario/tsp-solver.git
cd tsp-solver

# Instalar dependências básicas
pip install numpy matplotlib networkx

# (Opcional) Para uso de GPU:
pip install cupy
```

---

## ▶️ **Como Usar**

### 1. 🧱 **Gerar o Grafo**

```bash
python graph.py
```

📦 Gera os arquivos `tsp_dados.json` e `tsp_dados.pkl`
📸 Mostra o grafo com pesos e caminho ótimo

---

### 2. ⚖️ **Comparar CPU vs GPU**

```bash
python tsp.py
```

📈 Mostra tempos de execução, aceleração da GPU e salva gráfico de comparação.

---

## 💡 **Exemplo de Saída**

### `graph.py`

```
=== TRAVELING SALESMAN PROBLEM GRAPH GENERATOR ===
Generating graph with N=10 vertices...
Graph generated successfully!
Optimal path found: [0, 2, 5, 7, 9, 1, 4, 8, 3, 6, 0]
Total distance: 123.45
Data saved to 'tsp_dados.json' and 'tsp_dados.pkl'
```

### `tsp.py`

```
=== TSP SOLVER: CPU vs GPU ===
Data loaded: 10 vertices
Known optimal solution: 123.45
CPU - Brute force completed in 0.1234 seconds
GPU - Brute force completed in 0.0987 seconds
Speedup: 1.25x
Results saved to 'resultados_cpu_gpu.json'
```

---

## ⚠️ **Notas Importantes**

* 🚨 Para $N > 10$, o tempo de execução cresce exponencialmente devido a $(N-1)!$
* ⚡ A versão GPU exige CuPy e uma GPU NVIDIA compatível
* 🔧 Para alterar o número de vértices, edite a variável `N` em `graph.py`
* 🔁 A semente aleatória fixa (`np.random.seed(42)`) garante reprodutibilidade

---

## 📊 **Limitações**

* A geração de permutações ainda ocorre na CPU, o que pode ser um gargalo para valores altos de $N$
* Não resolve TSP assimétrico nem variantes com restrições adicionais

---

## 🤝 **Contribuições**

Contribuições são bem-vindas!

1. Fork este repositório
2. Crie uma nova branch: `git checkout -b minha-melhoria`
3. Envie seu *pull request* com uma descrição clara

---

## 📄 **Licença**

Distribuído sob a **Licença MIT**. Veja o arquivo `LICENSE` para mais detalhes.

---

## 📬 **Contato**

Dúvidas ou sugestões? Entre em contato:
📧 **\[[seu\_email@example.com](mailto:seu_email@example.com)]**
Ou abra uma *issue* no repositório!

---

Se quiser, posso gerar imagens e gráficos ilustrativos para incluir nesse `README.md`. Deseja que eu faça isso?
