import os,sys
import pandas as pd
import numpy as np
import ollama

class KGRewriter:
    def __init__(self, LLM, triplet_dict):
        self.LLM = LLM
        self.triplet_dict = triplet_dict

    def Rewrite(self,entities_set, description_set):

        prompt = f"""
        You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
        Given a list of entities, and a list of descriptions, all related to the same entity or group of entities.
        Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
        If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
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
