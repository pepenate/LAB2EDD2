import pandas as pd
import math
import matplotlib.pyplot as plt

# Paso 1: Cargar los datos del archivo CSV
df = pd.read_csv('C:/Users/HP/Downloads/flights_final.csv')

# Paso 2: Definir la función para calcular la distancia entre dos coordenadas geográficas
def haversine(lat1, lon1, lat2, lon2):
    # Fórmula haversine para calcular la distancia entre dos puntos en una esfera
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    r = 6371  # Radio de la Tierra en kilómetros
    return r * c

# Paso 3: Crear el grafo sin usar librerías externas
class Grafo:
    def __init__(self):
        self.nodos = {}
        self.aristas = {}
    
    def agregar_nodo(self, nodo, **kwargs):
        self.nodos[nodo] = kwargs
    
    def agregar_arista(self, origen, destino, peso):
        self.aristas.setdefault(origen, []).append((destino, peso))
        self.aristas.setdefault(destino, []).append((origen, peso))  # Grafo no dirigido
    
    def obtener_vecinos(self, nodo):
        return self.aristas.get(nodo, [])
    
    def obtener_nodos(self):
        return self.nodos.keys()
    
    def obtener_aristas(self):
        edges = []
        seen = set()
        for origen in self.aristas:
            for destino, peso in self.aristas[origen]:
                if (origen, destino) not in seen and (destino, origen) not in seen:
                    edges.append((origen, destino, peso))
                    seen.add((origen, destino))
        return edges

G = Grafo()

for _, row in df.iterrows():
    source = row['Source Airport Code']
    destination = row['Destination Airport Code']
    lat1, lon1 = row['Source Airport Latitude'], row['Source Airport Longitude']
    lat2, lon2 = row['Destination Airport Latitude'], row['Destination Airport Longitude']
    distance = haversine(lat1, lon1, lat2, lon2)
    if source not in G.nodos:
        G.agregar_nodo(source, name=row['Source Airport Name'], city=row['Source Airport City'],
                       country=row['Source Airport Country'], lat=lat1, lon=lon1)
    if destination not in G.nodos:
        G.agregar_nodo(destination, name=row['Destination Airport Name'], city=row['Destination Airport City'],
                       country=row['Destination Airport Country'], lat=lat2, lon=lon2)
    G.agregar_arista(source, destination, distance)

# Función para mostrar la información completa de un aeropuerto
def mostrar_info_aeropuerto(codigo):
    if codigo in G.nodos:
        info = G.nodos[codigo]
        print(f"\nInformación del Aeropuerto '{codigo}':")
        print(f"Nombre: {info['name']}")
        print(f"Ciudad: {info['city']}, País: {info['country']}")
        print(f"Latitud: {info['lat']}, Longitud: {info['lon']}")
    else:
        print("Código de aeropuerto no encontrado.")

# Función para calcular los caminos mínimos desde un nodo usando Dijkstra
def dijkstra(G, inicio):
    distancias = {nodo: float('inf') for nodo in G.obtener_nodos()}
    distancias[inicio] = 0
    visitados = set()
    predecesores = {nodo: None for nodo in G.obtener_nodos()}
    
    while len(visitados) < len(G.nodos):
        # Nodo no visitado con la distancia más pequeña
        nodo_actual = None
        min_dist = float('inf')
        for nodo in G.nodos:
            if nodo not in visitados and distancias[nodo] < min_dist:
                min_dist = distancias[nodo]
                nodo_actual = nodo
        if nodo_actual is None:
            break  # No hay más nodos alcanzables
        visitados.add(nodo_actual)
        for vecino, peso in G.obtener_vecinos(nodo_actual):
            if vecino not in visitados:
                nueva_distancia = distancias[nodo_actual] + peso
                if nueva_distancia < distancias[vecino]:
                    distancias[vecino] = nueva_distancia
                    predecesores[vecino] = nodo_actual
    return distancias, predecesores

# Función para mostrar los 10 aeropuertos más lejanos desde un aeropuerto dado
def mostrar_10_mas_lejanos(G, inicio):
    distancias, _ = dijkstra(G, inicio)
    # Filtrar y ordenar los aeropuertos por distancia
    aeropuertos_ordenados = sorted(distancias.items(), key=lambda x: x[1], reverse=True)
    # Tomar los 10 con distancias finitas
    aeropuertos_ordenados = [item for item in aeropuertos_ordenados if item[1] != float('inf')][:10]
    
    print(f"\nLos 10 aeropuertos más lejanos desde '{inicio}' son:")
    for codigo, distancia in aeropuertos_ordenados:
        info = G.nodos[codigo]
        print(f"{codigo} - {info['name']} ({info['city']}, {info['country']})")
        print(f"Latitud: {info['lat']}, Longitud: {info['lon']} - Distancia: {distancia:.2f} km\n")

# Función para reconstruir el camino mínimo entre dos vértices
def reconstruir_camino(predecesores, inicio, fin):
    camino = []
    actual = fin
    while actual != inicio and actual is not None:
        camino.insert(0, actual)
        actual = predecesores.get(actual)
    if actual == inicio:
        camino.insert(0, inicio)
    else:
        camino = []
    return camino

# Función para mostrar el camino mínimo con visualización gráfica
def mostrar_camino_minimo(G, inicio, destino):
    distancias, predecesores = dijkstra(G, inicio)
    camino = reconstruir_camino(predecesores, inicio, destino)
    
    if not camino:
        print("No existe un camino entre los aeropuertos seleccionados.")
        return
    
    print(f"\nCamino mínimo desde {inicio} hasta {destino}:")
    for codigo in camino:
        info = G.nodos[codigo]
        print(f"{codigo} - {info['name']} ({info['city']}, {info['country']})")
        print(f"Latitud: {info['lat']}, Longitud: {info['lon']}")
    
    print(f"Distancia total del camino: {distancias[destino]:.2f} km")
    
    # Visualización del camino en el mapa
    plt.figure(figsize=(10, 6))
    # Dibujar todas las aristas del grafo para contexto
    for u, v, _ in G.obtener_aristas():
        plt.plot([G.nodos[u]['lon'], G.nodos[v]['lon']], [G.nodos[u]['lat'], G.nodos[v]['lat']], 'gray', alpha=0.5)
    # Resaltar el camino mínimo
    for i in range(len(camino) - 1):
        u, v = camino[i], camino[i + 1]
        plt.plot([G.nodos[u]['lon'], G.nodos[v]['lon']], [G.nodos[u]['lat'], G.nodos[v]['lat']], 'b-', linewidth=2)
    
    lons = [G.nodos[nodo]['lon'] for nodo in camino]
    lats = [G.nodos[nodo]['lat'] for nodo in camino]
    plt.scatter(lons, lats, c='red', s=50, zorder=5)
    
    for nodo in camino:
        info = G.nodos[nodo]
        plt.text(info['lon'], info['lat'], nodo, fontsize=8, ha='right', va='bottom', color='darkblue')
    
    plt.title("Camino Mínimo entre Aeropuertos")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.grid(True)
    plt.show()

# Función para mostrar el grafo completo con aristas y pesos
def mostrar_mapa_aeropuertos(G):
    plt.figure(figsize=(14, 8))
    lons = [G.nodos[nodo]['lon'] for nodo in G.nodos]
    lats = [G.nodos[nodo]['lat'] for nodo in G.nodos]
    
    # Dibujar todas las aristas
    for u, v, peso in G.obtener_aristas():
        x_values = [G.nodos[u]['lon'], G.nodos[v]['lon']]
        y_values = [G.nodos[u]['lat'], G.nodos[v]['lat']]
        plt.plot(x_values, y_values, 'gray', alpha=0.5)
        # Calcular posición media para colocar la etiqueta
        mid_lon = (G.nodos[u]['lon'] + G.nodos[v]['lon']) / 2
        mid_lat = (G.nodos[u]['lat'] + G.nodos[v]['lat']) / 2
        plt.text(mid_lon, mid_lat, f"{peso:.1f} km", fontsize=5, color='green', alpha=0.6)
    
    # Dibujar los aeropuertos en el mapa
    plt.scatter(lons, lats, c='blue', alpha=0.6, edgecolors='k', zorder=3, s=10)
    
    # Mostrar etiquetas de los aeropuertos
    for nodo in G.nodos:
        info = G.nodos[nodo]
        plt.text(info['lon'], info['lat'], nodo, fontsize=6, ha='right', va='bottom', color='darkblue')
    
    # Título y etiquetas del gráfico
    plt.title('Grafo de Aeropuertos y Rutas Aéreas')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.grid(True)
    plt.show()

# Función para determinar si el grafo es conexo y mostrar las componentes si no lo es
def obtener_componentes_conexas(G):
    visitados = set()
    componentes = []
    for nodo in G.obtener_nodos():
        if nodo not in visitados:
            componente = set()
            dfs(G, nodo, visitados, componente)
            componentes.append(componente)
    return componentes

def dfs(G, nodo, visitados, componente):
    visitados.add(nodo)
    componente.add(nodo)
    for vecino, _ in G.obtener_vecinos(nodo):
        if vecino not in visitados:
            dfs(G, vecino, visitados, componente)

def mostrar_conectividad(G):
    componentes = obtener_componentes_conexas(G)
    if len(componentes) == 1:
        print("El grafo es conexo.")
    else:
        print(f"El grafo no es conexo y tiene {len(componentes)} componentes.")
        for idx, componente in enumerate(componentes, start=1):
            print(f"Componente {idx} tiene {len(componente)} vértices.")

# Función para calcular el peso del árbol de expansión mínima (MST) usando Kruskal
def kruskal_mst(G, componente):
    parent = {}
    rank = {}
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root:
            return False
        if rank[u_root] < rank[v_root]:
            parent[u_root] = v_root
        else:
            parent[v_root] = u_root
            if rank[u_root] == rank[v_root]:
                rank[u_root] += 1
        return True
    # Inicializar conjuntos disjuntos
    for nodo in componente:
        parent[nodo] = nodo
        rank[nodo] = 0
    # Reunir las aristas dentro de la componente
    edges = []
    for u in componente:
        for v, peso in G.obtener_vecinos(u):
            if v in componente and (u, v, peso) not in edges and (v, u, peso) not in edges:
                edges.append((u, v, peso))
    # Ordenar aristas por peso
    edges.sort(key=lambda x: x[2])
    mst_weight = 0
    for u, v, peso in edges:
        if union(u, v):
            mst_weight += peso
    return mst_weight

def calcular_mst(G):
    componentes = obtener_componentes_conexas(G)
    if len(componentes) == 1:
        peso_total = kruskal_mst(G, componentes[0])
        print(f"Peso total del árbol de expansión mínima (MST) del grafo completo: {peso_total:.2f} km")
    else:
        print("El grafo no es conexo, calculando MST para cada componente:")
        for idx, componente in enumerate(componentes, start=1):
            peso_total = kruskal_mst(G, componente)
            print(f"Peso del MST de la componente {idx}: {peso_total:.2f} km")

# Función para seleccionar un aeropuerto mediante la interfaz gráfica con zoom
def seleccionar_aeropuerto_por_click(G):
    print("Haga clic en el aeropuerto deseado. Puede hacer zoom y pan para ajustar la vista.")
    fig, ax = plt.subplots(figsize=(14, 8))
    lons = [G.nodos[nodo]['lon'] for nodo in G.nodos]
    lats = [G.nodos[nodo]['lat'] for nodo in G.nodos]
    sc = ax.scatter(lons, lats, c='blue', alpha=0.6, edgecolors='k', zorder=3, s=10)
    for nodo in G.nodos:
        info = G.nodos[nodo]
        ax.text(info['lon'], info['lat'], nodo, fontsize=6, ha='right', va='bottom', color='darkblue')
    ax.set_title('Seleccione un aeropuerto haciendo clic (puede hacer zoom y pan)')
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    ax.grid(True)

    # Habilitar el zoom y pan
    plt.tight_layout()
    plt.draw()

    # Esperar al clic del usuario
    puntos = plt.ginput(1, timeout=0)
    plt.close()
    if puntos:
        lon_click, lat_click = puntos[0]
        # Encontrar el aeropuerto más cercano al punto clicado
        min_dist = float('inf')
        aeropuerto_cercano = None
        for nodo in G.nodos:
            lon = G.nodos[nodo]['lon']
            lat = G.nodos[nodo]['lat']
            dist = ((lon - lon_click)**2 + (lat - lat_click)**2)
            if dist < min_dist:
                min_dist = dist
                aeropuerto_cercano = nodo
        return aeropuerto_cercano
    else:
        return None

# Menú principal del programa
def main():
    while True:
        print("\n--- Menú de Opciones ---")
        print("1. Mostrar el grafo de aeropuertos y rutas con distancias")
        print("2. Determinar si el grafo es conexo")
        print("3. Calcular el peso del árbol de expansión mínima (MST)")
        print("4. Calcular y mostrar el camino mínimo entre dos aeropuertos")
        print("5. Salir")

        opcion = input("Seleccione una opción: ").strip()
        
        if opcion == '1':
            mostrar_mapa_aeropuertos(G)
        elif opcion == '2':
            mostrar_conectividad(G)
        elif opcion == '3':
            calcular_mst(G)
        elif opcion == '4':
            # Opción para ingresar código o seleccionar en el mapa
            metodo = input("Seleccione el método para elegir el aeropuerto inicial:\n1. Ingresar código\n2. Seleccionar en el mapa\nOpción: ").strip()
            if metodo == '1':
                inicio = input("Ingrese el código del aeropuerto inicial: ").strip().upper()
            elif metodo == '2':
                inicio = seleccionar_aeropuerto_por_click(G)
                if inicio is None:
                    print("No se seleccionó ningún aeropuerto.")
                    continue
                else:
                    print(f"Aeropuerto seleccionado: {inicio}")
            else:
                print("Opción no válida.")
                continue
            # Mostrar la información del aeropuerto inicial y los 10 más lejanos
            mostrar_info_aeropuerto(inicio)
            mostrar_10_mas_lejanos(G, inicio)

            # Opción para ingresar código o seleccionar en el mapa para el destino
            metodo = input("Seleccione el método para elegir el aeropuerto destino:\n1. Ingresar código\n2. Seleccionar en el mapa\nOpción: ").strip()
            if metodo == '1':
                destino = input("Ingrese el código del aeropuerto destino: ").strip().upper()
            elif metodo == '2':
                destino = seleccionar_aeropuerto_por_click(G)
                if destino is None:
                    print("No se seleccionó ningún aeropuerto.")
                    continue
                else:
                    print(f"Aeropuerto seleccionado: {destino}")
            else:
                print("Opción no válida.")
                continue
            # Mostrar el camino mínimo entre el aeropuerto inicial y el destino
            mostrar_camino_minimo(G, inicio, destino)
        elif opcion == '5':
            print("Saliendo del programa...")
            break
        else:
            print("Opción no válida. Intente nuevamente.")

if __name__ == "__main__":
    main()







