from typing import List, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class SingleDataExample:
    nodes: List[str]
    edges: List[List[int]]
    sentence: str


@dataclass(frozen=True)
class PairDataExample:
    data_id: int
    example_1: SingleDataExample
    example_2: SingleDataExample
    label: int


@dataclass(frozen=True)
class SimpleDataFeature:
    id: Optional[int]
    label: int
    input_ids: List[int]
    attention_masks: List[int]
    token_type_ids: Optional[List[int]]


@dataclass(frozen=True)
class DataFeature:
    id: Optional[int]
    label: int
    input_ids: List[int]
    attention_masks: List[int]
    token_type_ids: Optional[List[int]]
    amr_input_ids: List[int]
    amr_attention_masks: List[int]
    amr_token_type_ids: Optional[List[int]]
    # [graph1_num_nodes,graph2_num_nodes]
    nodes_nums: List[int]
    edges1: List[List[int]]
    edges2: List[List[int]]
    node_intervals: List[List[int]]
