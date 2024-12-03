import os
import numpy as np
import csv

from evaluator.map import MAP
from evaluator.fmeasure import FMeasure
from evaluator.ndcg import NDCG
from classes.data import get_all_data, get_rank_triples, get_topk_triples

def evaluation(dataset, k, model_name):
    ndcg_class = NDCG()
    fmeasure = FMeasure()
    m = MAP()

    if dataset.ds_name == "dbpedia":
        IN_SUMM = os.path.join(os.getcwd(), f'outputs-{model_name}/dbpedia')

        # dbpedia instances have index 0 to 100 and 140 to 165
        start = [0, 140]
        end   = [100, 165]
    elif dataset.ds_name == "lmdb":
        IN_SUMM = os.path.join(os.getcwd(), f'outputs-{model_name}/lmdb')
        start = [100, 165]
        end   = [140, 175]
    elif dataset.ds_name == "faces":
        IN_SUMM = os.path.join(os.getcwd(), f'outputs-{model_name}/faces')
        start = [0, 25]
        end   = [25, 50]

    all_ndcg_scores = []
    all_fscore = []
    all_map_scores = []
    total_ndcg=0
    total_fscore=0
    total_map_score=0
    entity_ids = []

    for i in range(start[0], end[0]):
        t = i+1
        entity_ids.append(t)

        # gold_list_top: list of list where each list contains the top k triples for respective gold summary
        # triples_dict: dictionary of triples with triple itself as key and index as value
        # triple_tuples: list where every element contains a triple as a string
        gold_list_top, triples_dict, triple_tuples = get_all_data(dataset.db_path, t, k, dataset.file_n)

        # rank_triples: list of predicted triple ranks from the rank file
        # encoded_rank_triples: list of corresponding indexes
        rank_triples, encoded_rank_triples = get_rank_triples(IN_SUMM, t, k, triples_dict)

        # topk_triples: list of predicted top k triples from the top file
        # encoded_topk_triples: list of corresponding indexes
        topk_triples, encoded_topk_triples = get_topk_triples(IN_SUMM, t, k, triples_dict)


        ndcg_score = ndcg_class.get_score(gold_list_top, encoded_rank_triples)
        f_score = fmeasure.get_score(encoded_topk_triples, gold_list_top)
        map_score = m.get_map(encoded_rank_triples, gold_list_top)

        total_ndcg += ndcg_score
        all_ndcg_scores.append(ndcg_score)

        total_fscore += f_score
        all_fscore.append(f_score)

        all_map_scores.append(map_score)

    for i in range(start[1], end[1]):
        t = i+1
        entity_ids.append(t)

        gold_list_top, triples_dict, triple_tuples = get_all_data(dataset.db_path, t, k, dataset.file_n)
        rank_triples, encoded_rank_triples = get_rank_triples(IN_SUMM, t, k, triples_dict)
        topk_triples, encoded_topk_triples = get_topk_triples(IN_SUMM, t, k, triples_dict)
        ndcg_score = ndcg_class.get_score(gold_list_top, encoded_rank_triples)
        f_score = fmeasure.get_score(encoded_topk_triples, gold_list_top)
        map_score = m.get_map(encoded_rank_triples, gold_list_top)
        total_ndcg += ndcg_score
        all_ndcg_scores.append(ndcg_score)
        total_fscore += f_score
        all_fscore.append(f_score)
        all_map_scores.append(map_score)

    print(len(all_fscore))
    # Analyze all_fscore and write the eid for the 5% worst performing entities into a CSV file
    num_worst = max(1, int(len(all_fscore) * 0.05))  # Calculate the number of entities corresponding to the bottom 5%
    worst_indices = np.argsort(all_fscore)[:num_worst]  # Get the indices of the bottom 5% scores

    worst_entities = [entity_ids[i] for i in worst_indices]  # Map indices to entity IDs

    # Define the CSV file name
    csv_file_name = f'worst_entities_{dataset.ds_name}_top{k}.csv'

    # Write the worst performing entity IDs to a CSV file
    with open(csv_file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # csvwriter.writerow(['Dataset', 'Top K', 'Entity ID', 'F-Score'])
        for idx in worst_indices:
            csvwriter.writerow([dataset.ds_name, k, entity_ids[idx], all_fscore[idx]])

    print("{}@top{}: F-Measure={}, NDCG={}, MAP={}".format(dataset, k, np.average(all_fscore), np.average(all_ndcg_scores), np.average(all_map_scores)))



    ## Hast be done one level higher!