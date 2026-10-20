# layout.py
"""
Автоматическая раскладка вершин графа.

Используется, когда координаты не заданы вручную или в Excel.
Гарантирует, что:
    - исток (вершина 0) находится в крайней левой колонке,
    - сток (вершина count+1) — в крайней правой,
    - обычные вершины — между ними, слоями.
"""
import networkx as nx


def auto_layout(graph, count, mode="layered"):
    """
    Вернуть словарь {vertex: (x, y)} для всех вершин графа.

    Параметры
    ---------
    graph : nx.DiGraph
        Граф с вершинами 0..count+1.
    count : int
        Число «обычных» вершин (без источника и стока).
    mode : str
        'layered' — слоистая раскладка,
        'spring'  — пружинная (исток и сток разносим по краям вручную).

    Возвращает
    ----------
    dict[int, tuple[float, float]]
    """
    sink = count + 1

    if mode == "spring":
        pos = nx.spring_layout(graph, seed=42, k=1.6, iterations=120)
        pos = {v: (float(x), float(y)) for v, (x, y) in pos.items()}

        # Разносим исток и сток по краям
        xs = [x for v, (x, y) in pos.items() if v not in (0, sink)]
        if xs:
            min_x, max_x = min(xs), max(xs)
            span = max(max_x - min_x, 1.0)
        else:
            min_x, max_x, span = 0.0, 1.0, 1.0

        pos[0] = (min_x - span * 0.4, 0.0)
        pos[sink] = (max_x + span * 0.4, 0.0)

        return {v: (round(x, 2), round(y, 2)) for v, (x, y) in pos.items()}

    # ---------- Слоистая раскладка ----------
    # Уровни BFS от источника
    try:
        levels = nx.single_source_shortest_path_length(graph, 0)
    except nx.NetworkXError:
        levels = {}

    # Недостижимые — в отдельный правый слой (не в слой стока)
    max_reachable = max([lvl for v, lvl in levels.items() if v != sink], default=0)
    for v in graph.nodes():
        if v not in levels:
            max_reachable += 1
            levels[v] = max_reachable

    # Исток всегда уровень 0, сток — всегда максимальный
    levels[0] = 0
    ordinary_levels = [lvl for v, lvl in levels.items() if v != sink]
    max_lvl = max(ordinary_levels) if ordinary_levels else 0
    levels[sink] = max_lvl + 1

    # Группируем по слоям
    groups = {}
    for v, lvl in levels.items():
        groups.setdefault(lvl, []).append(v)

    # Раскладываем: x = уровень * 2, y = центрированный индекс
    pos = {}
    max_x = max(groups.keys()) * 2.0 if groups else 0.0

    for lvl in sorted(groups.keys()):
        nodes_sorted = sorted(groups[lvl])
        n = len(nodes_sorted)
        for i, v in enumerate(nodes_sorted):
            y = (n - 1) / 2 - i
            x = lvl * 2.0
            pos[v] = (round(x, 2), round(y, 2))

    # Явно ставим исток и сток по краям, по вертикали — центр
    pos[0] = (0.0, 0.0)
    pos[sink] = (round(max_x + 2.0, 2), 0.0)

    return pos