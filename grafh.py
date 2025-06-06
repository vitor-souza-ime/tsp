# Programa 1: Gerador de Grafo para Problema do Caixeiro Viajante
# Este programa gera um grafo completo com N=10 vértices, distâncias aleatórias,
# plota o grafo e salva os dados em arquivo

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import json
from itertools import permutations
import random

N=10

def gerar_grafo_tsp(n=30):
    """
    Gera um grafo completo com n vértices e distâncias aleatórias
    """
    # Gerar coordenadas aleatórias para os vértices
    np.random.seed(42)  # Para reprodutibilidade
    coordenadas = np.random.rand(n, 2) * 100  # Coordenadas entre 0 e 100

    # Calcular matriz de distâncias
    matriz_distancias = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                # Distância euclidiana + componente aleatória
                dist_euclidiana = np.sqrt((coordenadas[i][0] - coordenadas[j][0])**2 +
                                        (coordenadas[i][1] - coordenadas[j][1])**2)
                # Adicionar variação aleatória de ±20%
                variacao = random.uniform(0.8, 1.2)
                matriz_distancias[i][j] = dist_euclidiana * variacao
            else:
                matriz_distancias[i][j] = 0

    return coordenadas, matriz_distancias

def plotar_grafo(coordenadas, matriz_distancias):
    """
    Plota o grafo usando NetworkX e Matplotlib
    """
    n = len(coordenadas)

    # Criar grafo
    G = nx.Graph()

    # Adicionar nós com posições
    pos = {}
    for i in range(n):
        G.add_node(i)
        pos[i] = (coordenadas[i][0], coordenadas[i][1])

    # Adicionar arestas com pesos
    for i in range(n):
        for j in range(i+1, n):
            G.add_edge(i, j, weight=matriz_distancias[i][j])

    # Configurar o plot
    plt.figure(figsize=(12, 8))

    # Desenhar nós
    nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                          node_size=1000, alpha=0.9)

    # Desenhar arestas
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1)

    # Desenhar labels dos nós
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    # Desenhar pesos das arestas (apenas algumas para não poluir)
    edge_labels = {}
    edges = list(G.edges())
    # Mostrar apenas algumas arestas para clareza
    for i, (u, v) in enumerate(edges[:15]):  # Primeiras 15 arestas
        edge_labels[(u, v)] = f'{matriz_distancias[u][v]:.1f}'

    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

    plt.title('Grafo do Problema do Caixeiro Viajante (N=10)', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return G

def encontrar_melhor_caminho_brute_force(matriz_distancias):
    """
    Encontra o melhor caminho usando força bruta (para N pequeno)
    """
    n = len(matriz_distancias)
    vertices = list(range(1, n))  # Excluir vértice 0 (ponto de partida fixo)

    melhor_distancia = float('inf')
    melhor_caminho = None

    # Testar todas as permutações
    for perm in permutations(vertices):
        caminho = [0] + list(perm) + [0]  # Começar e terminar no vértice 0
        distancia_total = 0

        for i in range(len(caminho) - 1):
            distancia_total += matriz_distancias[caminho[i]][caminho[i+1]]

        if distancia_total < melhor_distancia:
            melhor_distancia = distancia_total
            melhor_caminho = caminho

    return melhor_caminho, melhor_distancia

def salvar_dados(coordenadas, matriz_distancias, melhor_caminho, melhor_distancia):
    """
    Salva os dados em arquivos para uso posterior
    """
    dados = {
        'coordenadas': coordenadas.tolist(),
        'matriz_distancias': matriz_distancias.tolist(),
        'melhor_caminho': melhor_caminho,
        'melhor_distancia': melhor_distancia,
        'n_vertices': len(coordenadas)
    }

    # Salvar em JSON
    with open('tsp_dados.json', 'w') as f:
        json.dump(dados, f, indent=2)

    # Salvar em pickle (mais eficiente para arrays NumPy)
    with open('tsp_dados.pkl', 'wb') as f:
        pickle.dump(dados, f)

    print("Dados salvos em 'tsp_dados.json' e 'tsp_dados.pkl'")

def main():
    """
    Função principal
    """
    print("=== GERADOR DE GRAFO PARA PROBLEMA DO CAIXEIRO VIAJANTE ===")
    print("Gerando grafo com N=10 vértices...")

    # Gerar grafo
    coordenadas, matriz_distancias = gerar_grafo_tsp(N)

    print("Grafo gerado com sucesso!")
    print(f"Coordenadas dos vértices:\n{coordenadas}")
    print(f"\nMatriz de distâncias (primeiras 5x5):\n{matriz_distancias[:5, :5]}")

    # Plotar grafo
    print("\nPlotando grafo...")
    G = plotar_grafo(coordenadas, matriz_distancias)

    # Encontrar solução ótima usando força bruta
    print("\nCalculando solução ótima usando força bruta...")
    melhor_caminho, melhor_distancia = encontrar_melhor_caminho_brute_force(matriz_distancias)

    print(f"Melhor caminho encontrado: {melhor_caminho}")
    print(f"Distância total: {melhor_distancia:.2f}")

    # Plotar melhor caminho
    plt.figure(figsize=(12, 8))
    pos = {i: (coordenadas[i][0], coordenadas[i][1]) for i in range(len(coordenadas))}

    # Desenhar todos os nós
    nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                          node_size=1000, alpha=0.9)

    # Desenhar todas as arestas em cinza claro
    nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.5, edge_color='gray')

    # Desenhar o melhor caminho em vermelho
    caminho_edges = [(melhor_caminho[i], melhor_caminho[i+1])
                     for i in range(len(melhor_caminho)-1)]
    nx.draw_networkx_edges(G, pos, edgelist=caminho_edges,
                          edge_color='red', width=3, alpha=0.8)

    # Labels dos nós
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    plt.title(f'Melhor Caminho TSP - Distância: {melhor_distancia:.2f}', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Salvar dados
    salvar_dados(coordenadas, matriz_distancias, melhor_caminho, melhor_distancia)

    print("\n=== RESUMO ===")
    print(f"Número de vértices: {len(coordenadas)}")
    print(f"Melhor caminho: {' -> '.join(map(str, melhor_caminho))}")
    print(f"Distância total: {melhor_distancia:.2f}")
    print("Dados salvos com sucesso!")

if __name__ == "__main__":
    main()
