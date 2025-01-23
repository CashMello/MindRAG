#1、load triplets of Graph
#2、embedding entities
#3、save faiss index
import os
import csv
import networkx as nx

class GraphManager:
    def __init__(self, graph_file, save_dir):
        self.graph_file = graph_file  # CSV 文件路径
        self.save_dir = save_dir      # 图保存的目录
        self.graph = None

        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)

    def load_graph(self):
        """
        从 CSV 文件加载图，或从保存的文件中加载图。
        """
        # 检查是否有已保存的图文件
        save_path = os.path.join(self.save_dir, "saved_graph.graphml")
        if os.path.exists(save_path):
            print("Loading graph from saved file...")
            self.graph = nx.read_graphml(save_path)
        else:
            print("Loading graph from CSV file...")
            self.graph = self._load_graph_from_csv()
            # 保存图以便下次直接加载
            self.save_graph(save_path)
        return self.graph

    def _load_graph_from_csv(self):
        """
        从 CSV 文件加载图。
        """
        graph = nx.DiGraph()
        with open(self.graph_file, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                subject = row['Subject']
                predicate = row['Predicate']
                obj = row['Object']
                graph.add_edge(subject, obj, predicate=predicate)
        return graph

    def save_graph(self, save_path):
        """
        将图保存到文件。
        """
        nx.write_graphml(self.graph, save_path)
        print(f"Graph saved to {save_path}")

    def check_graph_existence(self):
        """
        检查是否有已保存的图文件。
        """
        save_path = os.path.join(self.save_dir, "saved_graph.graphml")
        return os.path.exists(save_path)

    def get_all_entities(self):
        """
        返回图中所有实体组成的列表。
        """
        if self.graph is None:
            raise ValueError("Graph is not loaded. Please call `load_graph()` first.")

        entities = list(self.graph.nodes)
        return entities

    def get_connected_entities_and_relations(self, entities, k=1):
        """
        获取与给定实体通过关系连接的实体和关系（k-hop），并返回去重后的实体和关系列表。

        :param entities: 初始实体集合（list）
        :param k: k-hop 的跳数（默认 1）
        :return: 两个列表，分别是去重后的实体和关系
        """
        if self.graph is None:
            raise ValueError("Graph is not loaded. Please call `load_graph()` first.")

        # 初始化结果集合
        connected_entities = set(entities)  # 去重实体
        connected_relations = set()         # 去重关系

        # 遍历 k-hop
        for _ in range(k):
            new_entities = set()
            for entity in connected_entities:
                # 获取当前实体的所有邻居节点和边
                for neighbor, edge_data in self.graph[entity].items():
                    new_entities.add(neighbor)  # 添加邻居实体
                    connected_relations.add(edge_data['predicate'])  # 添加关系
            connected_entities.update(new_entities)  # 更新实体集合

        # 返回去重后的实体和关系列表
        return list(connected_entities), list(connected_relations)

