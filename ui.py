# ui.py
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

from core import CoreMixin
from algorithms import AlgorithmsMixin
from layout import auto_layout


class FlowNetworkApp(CoreMixin, AlgorithmsMixin):
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Начальное окно")
        self.root.geometry("1000x750")
        self.current_window = None
        self.current_input_window = None
        self.cache = {}
        self.elite_set = []
        self.elite_size = 5

        self.grasp_alpha = 0.3
        self.ls_iterations = 30
        self.step = 10

        # Последний ввод (для возврата из окна графа)
        self.last_input = None

    # =========================================================
    #                    Рисование графа
    # =========================================================
    def draw_manual_graph(self, graph, proportions, fixed_positions, loads, canvas, count):
        canvas.figure.clear()
        fig = canvas.figure
        ax = fig.add_subplot(111)
        colors = ['lightgray', 'gray', 'black']

        for node in graph.nodes():
            if node not in fixed_positions:
                continue
            x, y = fixed_positions[node]
            outgoing_proportions = list(proportions.get(node, [0, 0]))
            if proportions.get(node, [0, 0])[1] == 0 and proportions.get(node, [0, 0])[0] != 0:
                radius = 0.3
            else:
                radius = 0.2
            circle = Circle((x, y), radius, color='lightgray', ec='w')
            ax.add_patch(circle)
            if sum(outgoing_proportions) > 0:
                wedges, _ = ax.pie(outgoing_proportions, center=(x, y), radius=radius,
                                   startangle=90, counterclock=False,
                                   colors=colors[:len(outgoing_proportions)])
            fontsize = 9
            if node == 0:
                ax.text(x, y, 's', ha='center', va='center', fontsize=fontsize, color='white')
            elif node == count + 1:
                ax.text(x, y, 't', ha='center', va='center', fontsize=fontsize, color='white')
            else:
                ax.text(x, y, str(node), ha='center', va='center', fontsize=fontsize, color='black')

        for start, end in graph.edges():
            if start not in fixed_positions or end not in fixed_positions:
                continue
            start_pos = fixed_positions[start]
            end_pos = fixed_positions[end]
            direction = np.array(end_pos) - np.array(start_pos)
            norm = np.linalg.norm(direction)
            if norm > 0:
                norm_direction = direction / norm
            else:
                norm_direction = direction
            start_arrow_pos = start_pos + norm_direction * 0.2
            end_arrow_pos = end_pos - norm_direction * 0.2
            arrow = FancyArrowPatch(start_arrow_pos, end_arrow_pos,
                                    mutation_scale=8,
                                    arrowstyle='-|>', color='black')
            ax.add_patch(arrow)
        ax.axis('equal')
        plt.tight_layout()
        canvas.draw()

    def drawgraph(self, count, vertex_capacities, edges, fixed_positions):
        graph = nx.DiGraph()
        graph.add_edges_from(edges)
        for i in range(count + 2):
            if i not in graph.nodes():
                graph.add_node(i)

        flow_graph = self.create_flow_graph(graph, vertex_capacities)
        destination_vertex = f"{count+1}_in"

        start_time = time.time()
        flow_value, flow_dict = self.compute_max_flow(flow_graph, '0_in', destination_vertex, use_cache=False)
        elapsed_time = time.time() - start_time

        loads = {}
        for start in flow_dict:
            node_id_str = start.split('_')[0]
            try:
                node_id = int(node_id_str)
                loads[node_id] = sum(flow_dict[start].values())
                loads[node_id] = np.nan_to_num(loads[node_id])
            except (ValueError, Exception):
                pass

        proportions = {i: [loads.get(i, 0), vertex_capacities.get(i, 0) - loads.get(i, 0)]
                       for i in range(count + 2)}
        proportions[0] = [0, 0, 1]
        proportions[count + 1] = [0, 0, 1]

        root4 = tk.Toplevel(self.root)
        root4.title("Граф и распределение ресурсов")
        root4.geometry("1400x800")
        self.current_window = root4

        def on_closing():
            self.current_window = None
            root4.destroy()
        root4.protocol("WM_DELETE_WINDOW", on_closing)

        left_frame = tk.Frame(root4, width=1000)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_frame.pack_propagate(False)

        fig = plt.figure(figsize=(10, 7))
        canvas = FigureCanvasTkAgg(fig, master=left_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        right_frame = tk.Frame(root4, width=380, bg='lightgray')
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)

        control_frame = tk.Frame(right_frame, bg='lightgray')
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        time_label = tk.Label(control_frame, text=f"Время вычисления потока: {elapsed_time:.6f} сек",
                              font=('Arial', 10), bg='lightgray', fg='darkgreen')
        time_label.pack(pady=5)

        lbl1 = tk.Label(control_frame, text=f"Максимальный поток: {flow_value:.2f}",
                        font=('Arial', 14, 'bold'), bg='lightgray', fg='darkblue')
        lbl1.pack(pady=10)

        tk.Frame(control_frame, height=2, bg='gray').pack(fill=tk.X, pady=5)

        lbl2 = tk.Label(control_frame, text="Введите выделенное количество средств:",
                        font=('Arial', 12), bg='lightgray')
        lbl2.pack(pady=10)

        text_field = tk.Text(control_frame, width=15, height=1, font=('Arial', 12))
        text_field.pack(pady=5)
        text_field.insert(tk.END, "5000")

        mode_frame = tk.LabelFrame(control_frame, text="Выбор режима", bg='lightgray',
                                   font=('Arial', 10, 'bold'))
        mode_frame.pack(fill=tk.X, pady=5, padx=5)

        mode_var = tk.StringVar(value="grasp")

        row1 = tk.Frame(mode_frame, bg='lightgray')
        row1.pack(side=tk.TOP, pady=2)
        tk.Radiobutton(row1, text="Жадный", variable=mode_var, value="greedy",
                       bg='lightgray', font=('Arial', 9)).pack(side=tk.LEFT, padx=3)
        tk.Radiobutton(row1, text="Regret", variable=mode_var, value="regret",
                       bg='lightgray', font=('Arial', 9)).pack(side=tk.LEFT, padx=3)
        tk.Radiobutton(row1, text="LS", variable=mode_var, value="local_search",
                       bg='lightgray', font=('Arial', 9)).pack(side=tk.LEFT, padx=3)
        tk.Radiobutton(row1, text="SLS", variable=mode_var, value="sls",
                       bg='lightgray', font=('Arial', 9)).pack(side=tk.LEFT, padx=3)

        row2 = tk.Frame(mode_frame, bg='lightgray')
        row2.pack(side=tk.TOP, pady=2)
        tk.Radiobutton(row2, text="Regret+LS", variable=mode_var, value="regret_ls",
                       bg='lightgray', font=('Arial', 9)).pack(side=tk.LEFT, padx=3)
        tk.Radiobutton(row2, text="GRASP", variable=mode_var, value="grasp",
                       bg='lightgray', font=('Arial', 9)).pack(side=tk.LEFT, padx=3)
        tk.Radiobutton(row2, text="GRASP+LS", variable=mode_var, value="grasp_ls",
                       bg='lightgray', font=('Arial', 9)).pack(side=tk.LEFT, padx=3)

        row3 = tk.Frame(mode_frame, bg='lightgray')
        row3.pack(side=tk.TOP, pady=2)
        tk.Radiobutton(row3, text="GRASP+PR", variable=mode_var, value="grasp_pr",
                       bg='lightgray', font=('Arial', 9)).pack(side=tk.LEFT, padx=3)
        tk.Radiobutton(row3, text="GRASP+Regret+LS", variable=mode_var, value="combined",
                       bg='lightgray', font=('Arial', 9)).pack(side=tk.LEFT, padx=3)

        params_frame = tk.LabelFrame(control_frame, text="Параметры", bg='lightgray',
                                     font=('Arial', 10, 'bold'))
        params_frame.pack(fill=tk.X, pady=5, padx=5)

        alpha_frame = tk.Frame(params_frame, bg='lightgray')
        alpha_frame.pack(fill=tk.X, pady=2)
        tk.Label(alpha_frame, text="Alpha (GRASP):", font=('Arial', 9), bg='lightgray',
                 width=12, anchor='w').pack(side=tk.LEFT, padx=2)
        alpha_var = tk.StringVar(value=str(self.grasp_alpha))
        tk.Entry(alpha_frame, textvariable=alpha_var, width=8, font=('Arial', 9)).pack(side=tk.LEFT, padx=2)
        tk.Label(alpha_frame, text="(0-1)", font=('Arial', 8), bg='lightgray', fg='gray').pack(side=tk.LEFT)

        iter_frame = tk.Frame(params_frame, bg='lightgray')
        iter_frame.pack(fill=tk.X, pady=2)
        tk.Label(iter_frame, text="LS итераций:", font=('Arial', 9), bg='lightgray',
                 width=12, anchor='w').pack(side=tk.LEFT, padx=2)
        iter_var = tk.StringVar(value=str(self.ls_iterations))
        tk.Entry(iter_frame, textvariable=iter_var, width=8, font=('Arial', 9)).pack(side=tk.LEFT, padx=2)
        tk.Label(iter_frame, text="(10-100)", font=('Arial', 8), bg='lightgray', fg='gray').pack(side=tk.LEFT)

        step_frame = tk.Frame(params_frame, bg='lightgray')
        step_frame.pack(fill=tk.X, pady=2)
        tk.Label(step_frame, text="Шаг:", font=('Arial', 9), bg='lightgray',
                 width=12, anchor='w').pack(side=tk.LEFT, padx=2)
        step_var = tk.StringVar(value=str(self.step))
        tk.Entry(step_frame, textvariable=step_var, width=8, font=('Arial', 9)).pack(side=tk.LEFT, padx=2)
        tk.Label(step_frame, text="(5-50)", font=('Arial', 8), bg='lightgray', fg='gray').pack(side=tk.LEFT)

        tk.Button(params_frame, text="Применить параметры",
                  command=lambda: self.apply_params_silent(alpha_var, iter_var, step_var),
                  width=18, height=1, font=('Arial', 12), bg='lightblue').pack(pady=5)

        tk.Frame(control_frame, height=2, bg='gray').pack(fill=tk.X, pady=5)

        button_frame = tk.Frame(control_frame, bg='lightgray')
        button_frame.pack(pady=10)

        submit_button = tk.Button(button_frame, text="Распределить ресурсы",
                                  command=lambda: self.submit_text_with_mode(
                                      text_field, graph, proportions, fixed_positions,
                                      flow_value, vertex_capacities, canvas, lbl1, count,
                                      table_frame, time_label, mode_var, submit_button),
                                  width=19, height=1, font=('Arial', 12), bg='lightblue')
        submit_button.pack(side=tk.LEFT, padx=5)

        tk.Button(button_frame, text="Сброс кэша",
                  command=self.clear_cache,
                  width=11, height=1, font=('Arial', 12), bg='lightcoral').pack(side=tk.LEFT, padx=5)

        tk.Frame(control_frame, height=2, bg='gray').pack(fill=tk.X, pady=5)

        back_button = tk.Button(control_frame, text="← Назад к заданию графа",
                                command=lambda: self.back_to_input(root4),
                                width=24, height=1, font=('Arial', 11), bg='wheat')
        back_button.pack(pady=6)

        table_frame = tk.Frame(right_frame, bg='lightgray')
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(table_frame, text="Результаты распределения:",
                 font=('Arial', 12, 'bold'), bg='lightgray').pack(pady=5)
        tk.Label(table_frame, text="Нажмите 'Распределить ресурсы'",
                 font=('Arial', 10), bg='lightgray', fg='gray').pack(pady=10)

        self.draw_manual_graph(graph, proportions, fixed_positions, loads, canvas, count)

    # =========================================================
    #                    Возврат
    # =========================================================
    def back_to_input(self, graph_window):
        """
        Закрыть окно графа и вернуться туда, откуда пришли:
        - при ручном вводе — открыть окно ввода с сохранёнными данными,
        - при импорте из Excel — открыть начальное окно.
        """
        graph_window.destroy()
        self.current_window = None

        if self.last_input and self.last_input.get("source") == "manual":
            # Открыть окно ввода с предзаполнением
            self.on_button1_click(prefill=self.last_input)
        else:
            # Excel-путь или вообще ничего — на начальное окно
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()

    # =========================================================
    #                    Параметры
    # =========================================================
    def apply_params_silent(self, alpha_var, iter_var, step_var):
        try:
            alpha = float(alpha_var.get())
            if 0 <= alpha <= 1:
                self.grasp_alpha = alpha
        except ValueError:
            pass

        try:
            ls_iter = int(iter_var.get())
            if 10 <= ls_iter <= 100:
                self.ls_iterations = ls_iter
        except ValueError:
            pass

        try:
            step = int(step_var.get())
            if 5 <= step <= 50:
                self.step = step
        except ValueError:
            pass

    # =========================================================
    #                    Расчёт
    # =========================================================
    def submit_text_with_mode(self, text_field, graph, proportions, fixed_positions,
                              flow_value, vertex_capacities, canvas, lbl1, count,
                              table_frame, time_label, mode_var, submit_button):
        text = text_field.get("1.0", tk.END).strip()
        try:
            budget = int(text)
            if budget < 0:
                raise ValueError("Бюджет не может быть отрицательным")
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Введите корректное положительное число\n{str(e)}")
            return

        submit_button.config(state=tk.DISABLED, text="Идёт расчёт...",
                             bg='#d0d0d0', disabledforeground='#666666')
        self.root.update()

        try:
            total_start_time = time.time()
            mode = mode_var.get()
            destination_vertex = f"{count+1}_in"

            proportions_copy = {k: v.copy() for k, v in proportions.items()}

            if mode == "greedy":
                resources, current_capacities, flow_value, proportions_copy = self.greedy_algorithm(
                    graph, count, vertex_capacities, budget, flow_value, proportions_copy)
                mode_text = "Жадный"
            elif mode == "regret":
                resources, current_capacities, flow_value, proportions_copy = self.regret_algorithm(
                    graph, count, vertex_capacities, budget, flow_value, proportions_copy, regret_weight=0.5)
                mode_text = "Regret-выбор"
            elif mode == "local_search":
                resources, current_capacities, flow_value, proportions_copy = self.greedy_algorithm(
                    graph, count, vertex_capacities, budget, flow_value, proportions_copy)
                resources, current_capacities, flow_value, proportions_copy = self.local_search_only(
                    resources, current_capacities, flow_value, graph, count, vertex_capacities, proportions_copy)
                mode_text = "Локальный поиск"
            elif mode == "sls":
                resources, current_capacities, flow_value, proportions_copy = self.sls_algorithm(
                    graph, count, vertex_capacities, budget, flow_value, proportions_copy,
                    k_steps=100, k_neighbors=10, tabu_size=5, mu=0.1, improve_freq=20)
                mode_text = "SLS"
            elif mode == "regret_ls":
                resources, current_capacities, flow_value, proportions_copy = self.regret_with_local_search(
                    graph, count, vertex_capacities, budget, flow_value, proportions_copy)
                mode_text = "Regret+LS"
            elif mode == "grasp":
                resources, current_capacities, flow_value, proportions_copy = self.grasp_construction(
                    graph, count, vertex_capacities, budget, flow_value, proportions_copy)
                mode_text = f"GRASP (alpha={self.grasp_alpha})"
            elif mode == "grasp_ls":
                resources, current_capacities, flow_value, proportions_copy = self.grasp_with_local_search(
                    graph, count, vertex_capacities, budget, flow_value, proportions_copy)
                mode_text = f"GRASP+LS (alpha={self.grasp_alpha}, iter={self.ls_iterations})"
            elif mode == "grasp_pr":
                resources, current_capacities, flow_value, proportions_copy = self.grasp_with_path_relinking(
                    graph, count, vertex_capacities, budget, flow_value, proportions_copy, iterations=5)
                mode_text = "GRASP+PR"
            else:
                resources, current_capacities, flow_value, proportions_copy = self.combined_grasp_regret_ls(
                    graph, count, vertex_capacities, budget, flow_value, proportions_copy,
                    alpha=self.grasp_alpha, regret_prob=0.5, regret_weight=0.5)
                mode_text = f"GRASP+Regret+LS (alpha={self.grasp_alpha})"

            total_elapsed_time = time.time() - total_start_time

            flow_graph = self.create_flow_graph(graph, current_capacities)
            flow_value, flow_dict = self.compute_max_flow(flow_graph, '0_in',
                                                          destination_vertex, use_cache=False)

            loads1 = {}
            for start in flow_dict:
                node_id_str = start.split('_')[0]
                try:
                    node_id = int(node_id_str)
                    loads1[node_id] = sum(flow_dict[start].values())
                    loads1[node_id] = np.nan_to_num(loads1[node_id])
                except ValueError:
                    continue

            proportions = {i: [loads1.get(i, 0), current_capacities.get(i, 0) - loads1.get(i, 0)]
                           for i in range(count + 2)}
            proportions[0] = [0, 0, 1]
            proportions[count + 1] = [0, 0, 1]

            time_label.config(text=f"Время: {total_elapsed_time:.2f} сек | {mode_text}")

            lbl1.config(text=f"Максимальный поток: {flow_value:.2f}")
            self.draw_manual_graph(graph, proportions, fixed_positions, loads1, canvas, count)

            for widget in table_frame.winfo_children():
                widget.destroy()

            tk.Label(table_frame, text="Результаты распределения:",
                     font=('Arial', 12, 'bold'), bg='lightgray').pack(pady=5)

            table_container = tk.Frame(table_frame, bg='lightgray')
            table_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            columns = ('Агрегат', 'Начальная ПС', 'Конечная ПС', 'Выделено средств')
            tree = ttk.Treeview(table_container, columns=columns, show='headings', height=12)

            tree.heading('Агрегат', text='Агрегат')
            tree.heading('Начальная ПС', text='Начальная ПС')
            tree.heading('Конечная ПС', text='Конечная ПС')
            tree.heading('Выделено средств', text='Выделено средств')

            tree.column('Агрегат', width=70, anchor='center')
            tree.column('Начальная ПС', width=90, anchor='center')
            tree.column('Конечная ПС', width=90, anchor='center')
            tree.column('Выделено средств', width=100, anchor='center')

            for i in range(1, count + 1):
                if resources[i] > 0:
                    tree.insert('', tk.END, values=[i,
                                                   int(vertex_capacities.get(i, 0)),
                                                   int(current_capacities.get(i, 0)),
                                                   int(resources[i])])

            if tree.get_children() == ():
                tk.Label(table_container, text="Нет агрегатов, получивших средства",
                         font=('Arial', 10), bg='lightgray', fg='gray').pack(pady=20)
            else:
                scrollbar = ttk.Scrollbar(table_container, orient=tk.VERTICAL, command=tree.yview)
                tree.configure(yscrollcommand=scrollbar.set)
                tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        finally:
            submit_button.config(state=tk.NORMAL, text="Распределить ресурсы", bg='lightblue')
            self.root.update()

    # =========================================================
    #                    Ручной ввод графа
    # =========================================================
    def on_button1_click(self, prefill=None):
        # Если уже открыто окно ввода — закрыть старое
        if self.current_input_window is not None:
            try:
                self.current_input_window.destroy()
            except tk.TclError:
                pass
            self.current_input_window = None

        root2 = tk.Toplevel(self.root)
        root2.title("Ввод графа")
        root2.geometry("900x820")
        self.current_input_window = root2

        def on_input_closing():
            self.current_input_window = None
            root2.destroy()
        root2.protocol("WM_DELETE_WINDOW", on_input_closing)

        state = {
            "count": 0,
            "capacity_entries": [],
            "edges": [],
        }

        # --- Число вершин ---
        top = tk.LabelFrame(root2, text="Число вершин", font=('Arial', 12, 'bold'))
        top.pack(fill=tk.X, padx=10, pady=8)

        tk.Label(top, text="N =", font=('Arial', 12)).pack(side=tk.LEFT, padx=5)
        count_var = tk.StringVar(value="")
        tk.Entry(top, textvariable=count_var, width=8, font=('Arial', 12)).pack(side=tk.LEFT)
        tk.Label(top, text="(вершины 1..N; Исток и Сток добавляются автоматически)",
                 font=('Arial', 9), fg='gray').pack(side=tk.LEFT, padx=10)

        # --- Пропускные способности ---
        cap_frame_outer = tk.LabelFrame(root2, text="Пропускные способности",
                                        font=('Arial', 12, 'bold'))
        cap_frame_outer.pack(fill=tk.BOTH, expand=False, padx=10, pady=8)

        cap_canvas = tk.Canvas(cap_frame_outer, height=180, highlightthickness=0)
        cap_scroll = tk.Scrollbar(cap_frame_outer, orient=tk.VERTICAL, command=cap_canvas.yview)
        cap_inner = tk.Frame(cap_canvas)
        cap_inner.bind("<Configure>",
                       lambda e: cap_canvas.configure(scrollregion=cap_canvas.bbox("all")))
        cap_canvas.create_window((0, 0), window=cap_inner, anchor="nw")
        cap_canvas.configure(yscrollcommand=cap_scroll.set)
        cap_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cap_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Связи ---
        edge_frame = tk.LabelFrame(root2, text="Связи", font=('Arial', 12, 'bold'))
        edge_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        add_row = tk.Frame(edge_frame)
        add_row.pack(fill=tk.X, pady=4)

        tk.Label(add_row, text="Откуда:", font=('Arial', 10)).pack(side=tk.LEFT, padx=2)

        from_var = tk.StringVar(value="Исток")
        from_spin = tk.Spinbox(
            add_row,
            textvariable=from_var,
            width=12,
            font=('Arial', 10),
            values=["Исток"],
            wrap=True,
            justify='center',
        )
        from_spin.pack(side=tk.LEFT, padx=2)

        tk.Label(add_row, text="Куда:", font=('Arial', 10)).pack(side=tk.LEFT, padx=2)

        to_var = tk.StringVar(value="1")
        to_spin = tk.Spinbox(
            add_row,
            textvariable=to_var,
            width=12,
            font=('Arial', 10),
            values=["1"],
            wrap=True,
            justify='center',
        )
        to_spin.pack(side=tk.LEFT, padx=2)

        # Список связей
        list_frame = tk.Frame(edge_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=4)

        edge_listbox = tk.Listbox(list_frame, height=10, font=('Arial', 10))
        edge_scroll = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=edge_listbox.yview)
        edge_listbox.configure(yscrollcommand=edge_scroll.set)
        edge_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        edge_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # ---- Вспомогательные функции ----
        def label_of(vertex, n):
            if vertex == 0:
                return "Исток"
            if vertex == n + 1:
                return "Сток"
            return str(vertex)

        def value_of_label(label, n):
            s = label.strip()
            if s in ("Исток", "исток", "0"):
                return 0
            if s in ("Сток", "сток", str(n + 1)):
                return n + 1
            return int(s)

        def refresh_capacity_fields():
            for w in cap_inner.winfo_children():
                w.destroy()
            state["capacity_entries"] = []
            n = state["count"]
            for i in range(1, n + 1):
                row = tk.Frame(cap_inner)
                row.pack(fill=tk.X, pady=1)
                tk.Label(row, text=f"Вершина {i}:", font=('Arial', 10),
                         width=12, anchor='w').pack(side=tk.LEFT)
                e = tk.Entry(row, width=10, font=('Arial', 10))
                e.pack(side=tk.LEFT, padx=4)
                state["capacity_entries"].append(e)

        def refresh_edge_spinners():
            n = state["count"]

            from_values = ["Исток"] + [str(i) for i in range(1, n + 1)]
            to_values = [str(i) for i in range(1, n + 1)] + ["Сток"]

            from_spin.config(values=from_values)
            to_spin.config(values=to_values)

            if from_var.get() not in from_values:
                from_var.set("Исток")
            if to_var.get() not in to_values:
                to_var.set("1" if n >= 1 else "Сток")

            edge_listbox.delete(0, tk.END)
            for (a, b) in state["edges"]:
                edge_listbox.insert(tk.END, f"{label_of(a, n)} → {label_of(b, n)}")

        def apply_count():
            try:
                n = int(count_var.get())
                if n <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Ошибка", "Введите положительное целое число")
                return
            state["count"] = n
            state["edges"] = []
            edge_listbox.delete(0, tk.END)
            refresh_capacity_fields()
            refresh_edge_spinners()

        tk.Button(top, text="Применить", command=apply_count,
                  font=('Arial', 11), bg='lightblue').pack(side=tk.LEFT, padx=10)

        def add_edge():
            if state["count"] <= 0:
                messagebox.showwarning("Внимание", "Сначала задайте число вершин")
                return
            n = state["count"]
            try:
                a = value_of_label(from_var.get(), n)
                b = value_of_label(to_var.get(), n)
            except ValueError:
                messagebox.showerror("Ошибка", "Некорректный номер вершины")
                return
            if not (0 <= a <= n + 1) or not (0 <= b <= n + 1):
                messagebox.showerror("Ошибка", f"Вершина должна быть от 1 до {n}, "
                                                f"либо Исток / Сток")
                return
            if a == b:
                messagebox.showwarning("Внимание", "Исток и приёмник не могут совпадать")
                return
            if (a, b) in state["edges"]:
                messagebox.showinfo("Инфо", "Такая связь уже добавлена")
                return
            state["edges"].append((a, b))
            edge_listbox.insert(tk.END, f"{label_of(a, n)} → {label_of(b, n)}")

        tk.Button(add_row, text="+ Добавить", command=add_edge,
                  font=('Arial', 10), bg='lightgreen').pack(side=tk.LEFT, padx=10)

        def remove_selected():
            sel = edge_listbox.curselection()
            if not sel:
                return
            idx = sel[0]
            edge_listbox.delete(idx)
            del state["edges"][idx]

        tk.Button(add_row, text="Удалить выбранное", command=remove_selected,
                  font=('Arial', 10), bg='lightcoral').pack(side=tk.LEFT, padx=4)

        def build_graph():
            if state["count"] <= 0:
                messagebox.showerror("Ошибка", "Задайте число вершин")
                return
            if not state["edges"]:
                messagebox.showerror("Ошибка", "Добавьте хотя бы одну связь")
                return

            count = state["count"]

            capacities = []
            for i, e in enumerate(state["capacity_entries"], start=1):
                raw = e.get().strip()
                if not raw:
                    messagebox.showerror("Ошибка", f"Не задана ПС вершины {i}")
                    return
                try:
                    cap = int(raw)
                    if cap <= 0:
                        raise ValueError
                    capacities.append(cap)
                except ValueError:
                    messagebox.showerror("Ошибка", f"Некорректная ПС вершины {i}: {raw}")
                    return

            vertex_capacities = {0: float('inf')}
            for i, cap in enumerate(capacities, 1):
                vertex_capacities[i] = cap
            vertex_capacities[count + 1] = float('inf')

            edges = list(state["edges"])

            graph = nx.DiGraph()
            graph.add_edges_from(edges)
            for i in range(count + 2):
                if i not in graph.nodes():
                    graph.add_node(i)

            fixed_positions = auto_layout(graph, count, mode="layered")

            # Сохраняем последний ввод для возврата
            self.last_input = {
                "source": "manual",
                "count": count,
                "capacities": list(capacities),
                "edges": list(edges),
            }

            self.current_input_window = None
            root2.destroy()
            self.drawgraph(count, vertex_capacities, edges, fixed_positions)

        tk.Button(root2, text="Построить граф", command=build_graph,
                  font=('Arial', 13, 'bold'), bg='lightblue',
                  height=1, width=20).pack(pady=10)

        # --- Предзаполнение (если пришли из окна графа по «Назад») ---
        if prefill is not None and prefill.get("source") == "manual":
            count_var.set(str(prefill["count"]))
            apply_count()
            # Заполнить capacities
            caps = prefill.get("capacities", [])
            for i, e in enumerate(state["capacity_entries"]):
                if i < len(caps):
                    e.insert(0, str(caps[i]))
            # Заполнить связи
            n = prefill["count"]
            state["edges"] = list(prefill.get("edges", []))
            edge_listbox.delete(0, tk.END)
            for (a, b) in state["edges"]:
                edge_listbox.insert(tk.END, f"{label_of(a, n)} → {label_of(b, n)}")

    # =========================================================
    #                    Excel
    # =========================================================
    def on_button2_click(self):
        file_path = fd.askopenfilename(title="Выберите файл Excel",
                                       filetypes=[("Excel files", "*.xlsx;*.xls")])
        if not file_path:
            return
        try:
            df = pd.read_excel(file_path)

            if 'Число вершин' not in df.columns:
                raise ValueError("В файле отсутствует колонка 'Число вершин'")
            num_vertices = df['Число вершин'].dropna().to_numpy()
            if len(num_vertices) == 0:
                raise ValueError("Не найдено значение 'Число вершин'")
            count = int(num_vertices[0])

            if 'Пропускная способность' not in df.columns:
                raise ValueError("В файле отсутствует колонка 'Пропускная способность'")
            capacities_data = df['Пропускная способность'].dropna().to_numpy()
            if len(capacities_data) != count:
                raise ValueError(
                    f"Количество пропускных способностей ({len(capacities_data)}) "
                    f"не соответствует числу вершин ({count})"
                )
            capacities = [int(cap) for cap in capacities_data]

            if 'Связи' not in df.columns:
                raise ValueError("В файле отсутствует колонка 'Связи'")
            connections_data = df['Связи'].dropna().to_numpy()
            edges = []
            for conn in connections_data:
                try:
                    edge = tuple(map(int, str(conn).split('-')))
                    if len(edge) != 2:
                        raise ValueError
                    edges.append(edge)
                except ValueError:
                    raise ValueError(f"Некорректный формат связи: {conn}")

            graph = nx.DiGraph()
            graph.add_edges_from(edges)
            for i in range(count + 2):
                if i not in graph.nodes():
                    graph.add_node(i)

            coords_ok = False
            coordinates_data = None
            if 'Координаты' in df.columns:
                coordinates_data = df['Координаты'].dropna().to_numpy()
                if len(coordinates_data) == count:
                    coords_ok = True

            if coords_ok:
                fixed_positions = {0: (0, 0)}
                for i, pos in enumerate(coordinates_data, start=1):
                    try:
                        pos_str = str(pos).strip("() ")
                        x, y = map(int, pos_str.split(","))
                        fixed_positions[i] = (x, y)
                    except (ValueError, AttributeError):
                        raise ValueError(f"Некорректный формат координат: {pos}")
                max_x = max(x for x, _ in fixed_positions.values())
                fixed_positions[count + 1] = (max_x + 2, 0)
            else:
                fixed_positions = auto_layout(graph, count, mode="layered")

            vertex_capacities = {0: float('inf')}
            for i, cap in enumerate(capacities, 1):
                vertex_capacities[i] = int(cap)
            vertex_capacities[count + 1] = float('inf')

            # Источник — Excel. Возврат пойдёт на начальное окно.
            self.last_input = {"source": "excel"}

            self.drawgraph(count, vertex_capacities, edges, fixed_positions)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при чтении файла: {str(e)}")

    # =========================================================
    #                    Запуск
    # =========================================================
    def run(self):
        button1 = tk.Button(self.root, text="Задать данные для графа",
                            command=self.on_button1_click,
                            width=30, height=1, font=('Arial', 14))
        button1.place(relx=.35, rely=.4)

        button2 = tk.Button(self.root, text="Взять данные для графа из файла",
                            command=self.on_button2_click,
                            width=30, height=1, font=('Arial', 14))
        button2.place(relx=.35, rely=.6)

        self.root.mainloop()