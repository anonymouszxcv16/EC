import json, re
import gymnasium as gym
import os

import numpy as np
import cv2

# join gym environment
prefix = "../JoinGym/"

def make_env(seed, query_ids, disable_cartesian_product, enable_bushy):
    # read db schema and convert to dict
    schema_regex = r"(.*)\((.*)\)"
    db_schema = {}

    random_query_id = next(iter(query_ids))
    if isinstance(random_query_id, int):
        job_base_path = "imdb/job"
    else:
        job_base_path = "imdb/joingym"

    job_base_path = f"{prefix}{job_base_path}"

    with open(f"{prefix}imdb/schema.txt", "r") as f:
        schemas = f.readlines()
        for schema in schemas:
            match = re.match(schema_regex, schema.strip())
            table_name = match.group(1).strip()
            columns = match.group(2).strip().split(",")
            columns = [column.strip() for column in columns]
            db_schema[table_name] = columns

    join_contents = {}
    for id in query_ids:
        path = os.path.join(job_base_path, f"q{id}.json")
        with open(path, "r") as file:
            join_contents[id] = json.load(file)

    def thunk():
        env_name = "join_optimization_bushy-v0" if enable_bushy else "join_optimization_left-v0"
        env = gym.make(
            env_name, db_schema=db_schema, join_contents=join_contents,
            disable_cartesian_product=disable_cartesian_product,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

class PreprocessFrame:
    def __init__(self, width=16, height=16, grayscale=True):
        self.width = width
        self.height = height
        self.grayscale = grayscale

    def preprocess(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = frame / 255.0
        frame = (frame > 0.5).astype(np.float32)  # Binary 64x64x1
        return frame.flatten()  # Shape: (4096,)

    def reset(self, frame):
        return self.preprocess(frame)

    def step(self, frame):
        return self.preprocess(frame)