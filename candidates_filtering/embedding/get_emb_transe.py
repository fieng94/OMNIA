import numpy as np
import pandas as pd
import pykeen
from pykeen.pipeline import pipeline
from pykeen.models import TransE
from pykeen.triples import TriplesFactory

def get_triplet_emb(model: pykeen.models.TransE, triple_factory: pykeen.triples.TriplesFactory,
                    head: str, relation: str, tail: str):
    """ Return the triplet embedding from model, dataset and triplet list """
    # get total entity embedding
    entities_embedding = model.entity_representations[0](
        indices=None).detach().numpy()
    # get total relation embedding
    relations_embedding = model.relation_representations[0](
        indices=None).detach().numpy()

    # get id of each element of the triplet
    head_id = triple_factory.entity_to_id[head]
    relation_id = triple_factory.relation_to_id[relation]
    tail_id = triple_factory.entity_to_id[tail]

    # get every element embedding
    head_emb = entities_embedding[head_id]
    relation_emb = relations_embedding[relation_id]
    tail_emb = entities_embedding[tail_id]
    return (head_emb, relation_emb, tail_emb)


def compute_dist_emb(head_emb: np.ndarray, relation_emb: np.ndarray, tail_emb: np.ndarray) -> np.float32:
    """ Return computed distance of embedding """
    # sum of head and relation
    sum_head_relation = head_emb + relation_emb
    # difference between sum of head and relation and tail
    distance = np.linalg.norm(sum_head_relation - tail_emb)
    return distance


def get_list_dist(df: pd.DataFrame, model: pykeen.models.TransE, triple_factory: pykeen.triples.TriplesFactory,) -> list[np.float32]:
    """ Return list of distance for every embedded triple """
    # list of distance
    list_dist = []
    # iterate over the DataFrame
    for index, row in df.iterrows():
        head = row['Head']
        relation = row['Relation']
        tail = row['Tail']
        # get embedding
        head_emb, rel_emb, tail_emb = get_triplet_emb(
            model, triple_factory, head, relation, tail)
        # get distance of every triplet
        dist = compute_dist_emb(head_emb, rel_emb, tail_emb)
        list_dist.append(dist)
    return list_dist