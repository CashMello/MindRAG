import os, sys
import pandas as pd
import numpy as np
import json
import ollama


class KGRewriter:
    def __init__(self):
        pass


    def Rewrite(self, entities_set, description_set):
        prompt = f"""
        You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
        Given a list of entities, and a list of descriptions, please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
        If the provided descriptions are contradictory, please resolve the contradictions and provide a single, refined, coherent summary.
        Make sure it is written in third person, and include the entity names so we the have full context.

        #######
        -Data-
        Entities: {entities_set}
        Description List: {description_set}
        
        #######
        Output:
        """
        response = ollama.generate(
            model="qwen2.5:72b",
            prompt=prompt,
            options={
                "temperature": 0.5,  # 控制生成文本的随机性
            }
        )

        generated_text = f'Summary:{response["response"]}'

        return generated_text

    def Keyword_Extract(self, query):
        prompt = f"""---Role---
        
        You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

        ---Goal---

        Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

        ---Instructions---

        - Output the keywords in JSON format.
        - The JSON should have two keys:
        - "high_level_keywords" for overarching concepts or themes.
        - "low_level_keywords" for specific entities or details.

        ######################
        -Examples-
        ######################
        Example 1:

        Query: "How does international trade influence global economic stability?"
        ################
        Output:
        {{
            "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
            "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
        }}
        #############################
        Example 2:

        Query: "What are the environmental consequences of deforestation on biodiversity?"
        ################
        Output:
        {{
            "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
            "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
        }}
        #############################
        Example 3:

        Query: "What is the role of education in reducing poverty?"
        ################
        Output:
        {{
            "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
            "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
        }}
        #############################
        -Real Data-
        ######################
        Query: {query}
        ######################
        Output:

        """
        response = ollama.generate(
            model="qwen2.5:72b",
            prompt=prompt,
            options={
                "temperature": 0.5,  # 控制生成文本的随机性
            }
        )
        # 提取关键词
        try:
            response_dict = json.loads(response.response)  # 解析 JSON 字符串为字典
        except json.JSONDecodeError as e:
            print("Failed to parse response as JSON:", e)
            return []

        high_level_keywords = response_dict["high_level_keywords"]
        low_level_keywords = response_dict["low_level_keywords"]
        
        print("High-level Keywords:", high_level_keywords)
        print("Low-level Keywords:", low_level_keywords)

        return high_level_keywords + low_level_keywords

