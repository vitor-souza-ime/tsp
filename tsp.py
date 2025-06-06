# Programa TSP GPU vs CPU - Comparação de Performance
# Este programa resolve o TSP usando GPU (CUDA) e CPU para comparar tempos

import json
import time
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU (CUDA) disponível")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy não encontrado. Apenas CPU será usado.")

class TSPSolverGPU:
    def __init__(self, matriz_distancias):
        self.matriz_distancias = np.array(matriz_distancias)
        self.n = len(matriz_distancias)

        if GPU_AVAILABLE:
            # Transferir dados para GPU
            self.matriz_distancias_gpu = cp.array(matriz_distancias)

    def forca_bruta_cpu(self):
        """Resolve TSP usando força bruta na CPU"""
        print("Resolvendo TSP usando Força Bruta (CPU)...")
        start_time = time.time()

        vertices = list(range(1, self.n))
        melhor_distancia = float('inf')
        melhor_caminho = None

        count = 0
        for perm in permutations(vertices):
            caminho = [0] + list(perm) + [0]
            distancia = self._calcular_distancia_cpu(caminho)

            if distancia < melhor_distancia:
                melhor_distancia = distancia
                melhor_caminho = caminho

            count += 1
            if count % 10000 == 0:
                print(f"CPU - Testadas {count} permutações...")

        tempo_execucao = time.time() - start_time
        print(f"CPU - Força bruta concluída em {tempo_execucao:.4f} segundos")
        print(f"CPU - Permutações testadas: {count}")

        return melhor_caminho, melhor_distancia, tempo_execucao

    def forca_bruta_gpu(self):
        """Resolve TSP usando força bruta na GPU"""
        if not GPU_AVAILABLE:
            print("GPU não disponível, usando CPU...")
            return self.forca_bruta_cpu()

        print("Resolvendo TSP usando Força Bruta (GPU)...")
        start_time = time.time()

        # Gerar todas as permutações
        vertices = list(range(1, self.n))
        all_perms = list(permutations(vertices))

        # Converter para formato adequado para GPU
        batch_size = min(50000, len(all_perms))  # Processar em lotes
        melhor_distancia = float('inf')
        melhor_caminho = None

        print(f"GPU - Total de permutações: {len(all_perms)}")

        for i in range(0, len(all_perms), batch_size):
            batch_perms = all_perms[i:i+batch_size]

            # Criar caminhos completos (com início e fim em 0)
            caminhos_batch = []
            for perm in batch_perms:
                caminho = [0] + list(perm) + [0]
                caminhos_batch.append(caminho)

            # Transferir para GPU
            caminhos_gpu = cp.array(caminhos_batch)

            # Calcular distâncias na GPU
            distancias = self._calcular_distancias_gpu_batch(caminhos_gpu)

            # Encontrar mínimo no lote
            min_idx = cp.argmin(distancias)
            min_dist = float(distancias[min_idx])

            if min_dist < melhor_distancia:
                melhor_distancia = min_dist
                melhor_caminho = caminhos_batch[int(min_idx)]

            if (i // batch_size + 1) % 10 == 0:
                print(f"GPU - Processados {i + len(batch_perms)} caminhos...")

        tempo_execucao = time.time() - start_time
        print(f"GPU - Força bruta concluída em {tempo_execucao:.4f} segundos")

        return melhor_caminho, melhor_distancia, tempo_execucao

    def _calcular_distancia_cpu(self, caminho):
        """Calcula distância de um caminho na CPU"""
        distancia = 0
        for i in range(len(caminho) - 1):
            distancia += self.matriz_distancias[caminho[i]][caminho[i+1]]
        return distancia

    def _calcular_distancias_gpu_batch(self, caminhos_batch):
        """Calcula distâncias de um lote de caminhos na GPU"""
        # caminhos_batch shape: (batch_size, n+1)
        batch_size, caminho_len = caminhos_batch.shape

        # Criar índices para acessar a matriz de distâncias
        origem_indices = caminhos_batch[:, :-1]  # Todos exceto o último
        destino_indices = caminhos_batch[:, 1:]  # Todos exceto o primeiro

        # Calcular distâncias usando indexação avançada
        distancias_segmentos = self.matriz_distancias_gpu[origem_indices, destino_indices]

        # Somar distâncias de cada caminho
        distancias_totais = cp.sum(distancias_segmentos, axis=1)

        return distancias_totais

def carregar_dados(arquivo='tsp_dados.json'):
    """Carrega os dados do arquivo"""
    try:
        with open(arquivo, 'r') as f:
            dados = json.load(f)
        print(f"Dados carregados de {arquivo}")
        return dados
    except FileNotFoundError:
        print(f"Arquivo {arquivo} não encontrado!")
        return None

def plotar_comparacao(resultado_cpu, resultado_gpu, coordenadas):
    """Plota a comparação entre CPU e GPU"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Comparação de tempos
    if resultado_gpu is not None:
        algoritmos = ['CPU', 'GPU']
        tempos = [resultado_cpu[2], resultado_gpu[2]]
        cores = ['blue', 'green']

        bars = ax1.bar(algoritmos, tempos, color=cores)
        ax1.set_ylabel('Tempo (segundos)')
        ax1.set_title('Comparação de Tempo: CPU vs GPU')

        # Adicionar valores nas barras
        for bar, tempo in zip(bars, tempos):
            height = bar.get_height()
            ax1.annotate(f'{tempo:.4f}s',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        # Speedup
        speedup = resultado_cpu[2] / resultado_gpu[2]
        ax2.bar(['Speedup'], [speedup], color='orange')
        ax2.set_ylabel('Speedup (vezes)')
        ax2.set_title(f'Aceleração GPU: {speedup:.2f}x')
        ax2.annotate(f'{speedup:.2f}x',
                    xy=(0, speedup),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    else:
        ax1.text(0.5, 0.5, 'GPU não disponível', ha='center', va='center', transform=ax1.transAxes)
        ax2.text(0.5, 0.5, 'GPU não disponível', ha='center', va='center', transform=ax2.transAxes)

    # Plotar melhor caminho
    melhor_caminho = resultado_cpu[0]

    ax3.scatter([coordenadas[i][0] for i in range(len(coordenadas))],
               [coordenadas[i][1] for i in range(len(coordenadas))],
               c='blue', s=100)

    for i in range(len(melhor_caminho) - 1):
        x1, y1 = coordenadas[melhor_caminho[i]]
        x2, y2 = coordenadas[melhor_caminho[i+1]]
        ax3.plot([x1, x2], [y1, y2], 'r-', linewidth=2)

    for i, (x, y) in enumerate(coordenadas):
        ax3.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')

    ax3.set_title(f'Melhor Caminho (Distância: {resultado_cpu[1]:.2f})')
    ax3.grid(True, alpha=0.3)

    # Tabela de resultados
    ax4.axis('tight')
    ax4.axis('off')

    if resultado_gpu is not None:
        tabela_dados = [
            ['CPU', f"{resultado_cpu[1]:.2f}", f"{resultado_cpu[2]:.4f}s"],
            ['GPU', f"{resultado_gpu[1]:.2f}", f"{resultado_gpu[2]:.4f}s"],
            ['Speedup', f"{speedup:.2f}x", f"{resultado_cpu[2]/resultado_gpu[2]:.2f}x"]
        ]
    else:
        tabela_dados = [
            ['CPU', f"{resultado_cpu[1]:.2f}", f"{resultado_cpu[2]:.4f}s"],
            ['GPU', 'N/A', 'N/A']
        ]

    tabela = ax4.table(cellText=tabela_dados,
                      colLabels=['Método', 'Distância', 'Tempo'],
                      cellLoc='center',
                      loc='center')
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10)
    tabela.scale(1.2, 1.5)
    ax4.set_title('Resumo da Comparação')

    plt.tight_layout()
    plt.show()

def main():
    """Função principal"""
    print("=== TSP SOLVER: CPU vs GPU ===")

    # Carregar dados
    dados = carregar_dados()
    if dados is None:
        print("Erro ao carregar dados. Execute primeiro o programa gerador.")
        return

    matriz_distancias = dados['matriz_distancias']
    coordenadas = dados['coordenadas']
    solucao_otima = dados['melhor_distancia']

    print(f"Dados carregados: {len(matriz_distancias)} vértices")
    print(f"Solução ótima conhecida: {solucao_otima:.2f}")

    if len(matriz_distancias) > 10:
        print("AVISO: Para mais de 10 vértices, o tempo pode ser muito longo!")
        resposta = input("Deseja continuar? (s/n): ")
        if resposta.lower() != 's':
            return

    # Criar solver
    solver = TSPSolverGPU(matriz_distancias)

    # Resolver usando CPU
    print("\n" + "="*50)
    resultado_cpu = solver.forca_bruta_cpu()

    # Resolver usando GPU
    print("\n" + "="*50)
    resultado_gpu = None
    if GPU_AVAILABLE:
        resultado_gpu = solver.forca_bruta_gpu()
    else:
        print("Pulando GPU - CuPy não está instalado")

    # Mostrar resultados
    print("\n" + "="*50)
    print("=== RESULTADOS FINAIS ===")
    print(f"CPU - Melhor distância: {resultado_cpu[1]:.2f}")
    print(f"CPU - Tempo: {resultado_cpu[2]:.4f} segundos")

    if resultado_gpu is not None:
        print(f"GPU - Melhor distância: {resultado_gpu[1]:.2f}")
        print(f"GPU - Tempo: {resultado_gpu[2]:.4f} segundos")
        speedup = resultado_cpu[2] / resultado_gpu[2]
        print(f"Speedup: {speedup:.2f}x")

        # Verificar se as soluções são iguais
        if abs(resultado_cpu[1] - resultado_gpu[1]) < 0.001:
            print("✅ CPU e GPU encontraram a mesma solução ótima!")
        else:
            print("⚠️  CPU e GPU encontraram soluções diferentes")

    # Plotar resultados
    plotar_comparacao(resultado_cpu, resultado_gpu, coordenadas)

    # Salvar resultados
    resultados = {
        'cpu': {
            'caminho': resultado_cpu[0],
            'distancia': resultado_cpu[1],
            'tempo': resultado_cpu[2]
        }
    }

    if resultado_gpu is not None:
        resultados['gpu'] = {
            'caminho': resultado_gpu[0],
            'distancia': resultado_gpu[1],
            'tempo': resultado_gpu[2]
        }
        resultados['speedup'] = speedup

    with open('resultados_cpu_gpu.json', 'w') as f:
        json.dump(resultados, f, indent=2)

    print(f"\nResultados salvos em 'resultados_cpu_gpu.json'")

if __name__ == "__main__":
    main()
