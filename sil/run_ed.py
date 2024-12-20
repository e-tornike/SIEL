import pandas as pd
import os
import urllib3
import jsonlines
import typer
import json
import time
from collections import defaultdict
import requests
import hashlib
from lightning.fabric.utilities.seed import seed_everything

from haystack.document_stores.memory import InMemoryDocumentStore
from haystack.document_stores.opensearch import OpenSearchDocumentStore
from qdrant_haystack import QdrantDocumentStore
from haystack.schema import Document
from haystack.nodes import PreProcessor
from haystack.nodes import EmbeddingRetriever, BM25Retriever
from haystack.nodes import JoinDocuments, SentenceTransformersRanker
from haystack.pipelines import Pipeline
from ranx import Qrels, Run, evaluate
from tqdm import tqdm
import datetime

urllib3.disable_warnings()


def load_json(path):
    with open(path, "r") as reader:
        data = json.load(reader)
    return data

def load_jsonl(path):
    data = []
    with jsonlines.open(path, "r") as reader:
        for obj in reader:
            data.append(obj)
    return data

def get_candidate_vars(ids, rd_metas):
    candidates = []
    for i in ids:
        rd_meta = rd_metas.get(i, {})
        variables = [v.split("-")[-1] for v in list(rd_meta.keys())]
        candidates.extend(variables)
    return candidates

def make_documents(metadata, doc_ids=None, metadata_values=[], use_prefix=False):
    print("Making documents...")
    documents = []
    for rd_id, variables in tqdm(metadata.items()):
        if doc_ids and rd_id not in doc_ids:
            continue
        else:
            for vid, vmeta in variables.items():
                year = vmeta.get("year", "")

                values = []

                for item in metadata_values:
                    if item in ["title", "variable_label", "question_text", "sub_question", "item_category", "topic", "item_categories", "answer_categories"]:
                        item_val = vmeta.get(item, "")
                        if isinstance(item_val, list):
                            item_val = " ".join(item_val)
                        values.append(item_val.lower())
                    elif item in vmeta:
                        print(f"Item '{item}' is not in the set of predefined item categories but is contained in the available metadata: {vmeta.keys()}")
                    else:
                        print(f"Item '{item}' is neither in the set of predefined item categories nor in the available metadata. Make sure that you are selecting the correct item.")

                if values:
                    doc = Document(
                        content="data: "+" ".join(values) if use_prefix else " ".join(values),
                        meta={
                            "rd_id": rd_id,
                            "id": vid.split("-")[-1],
                            "year": year,
                        }
                    )
                    documents.append(doc)
    print("Done.")
    return documents

def make_document_index(
        metadata,
        doc_ids,
        clean_whitespace=True,
        split_by="word",
        split_length=512,
        metadata_values=[],
        use_prefix=False,
    ):
    preprocessor = PreProcessor(
        clean_whitespace=clean_whitespace,
        split_by=split_by,
        split_length=split_length,
    )

    documents = make_documents(metadata, doc_ids, metadata_values=metadata_values, use_prefix=use_prefix)
    print("Preprocessing documents...")
    docs_to_index = preprocessor.process(documents)
    print("Done.")

    return docs_to_index

def make_inmemory_document_index_pipeline(
        metadata,
        doc_ids,
        metadata_values,
        do_sparse=False,
        use_prefix=False,
        top_k_sparse=10,
    ):
    document_store = InMemoryDocumentStore(use_bm25=do_sparse)
    document_store.delete_documents()

    docs_to_index = make_document_index(metadata, doc_ids, metadata_values=metadata_values, use_prefix=use_prefix)
    document_store.write_documents(docs_to_index)

    print(f"Documents in index: {document_store.get_document_count()}")

    print("Initializing retrievers...")
    params = {}
    pipeline = Pipeline()
    inputs = []
    if do_sparse:
        print("Sparse retriever...")
        sparse_retriever = BM25Retriever(document_store=document_store)
        pipeline.add_node(component=sparse_retriever, name="SparseRetriever", inputs=["Query"])
        params["SparseRetriever"] = {"top_k": top_k_sparse}
        inputs.append("SparseRetriever")

    return pipeline, params, document_store

def make_pipeline(
        metadata,
        doc_ids,
        model_name,
        reranker_name,
        embedding_dim,
        index_dir,
        scale_score=False, 
        join_documents="concatenate",
        do_sparse=True,
        do_dense=True,
        do_join=True,
        do_rerank=True,
        top_k_sparse=10,
        top_k_dense=10,
        top_k_join=20,
        top_k_reranker=10,
        recreate_index=False,
        doc_store_type="qdrant",
        metadata_values=[],
        model_format="sentence_transformers",
        use_prefix=False,
    ):
    index_name = f"{os.path.basename(model_name).replace('/', '==')}"
    if doc_ids:
        hex = hashlib.md5("".join(sorted(doc_ids)+sorted(metadata_values)).encode()).hexdigest()
        index_name += f"_hex={hex}"
    else:
        hex = hashlib.md5("".join(sorted(metadata_values)).encode()).hexdigest()
        index_name += f"_hex={hex}"
    index_path = os.path.join(index_dir, "collection", index_name)

    index_exists = True if os.path.isdir(index_path) else False
    if index_exists and doc_store_type not in ["inmemory"]:
        print(f"Index already exists: {index_path}")

    if doc_store_type == "inmemory":
        document_store = InMemoryDocumentStore(use_bm25=do_sparse)
    elif doc_store_type == "opensearch":
        document_store = OpenSearchDocumentStore(index=index_name, embedding_dim=embedding_dim, batch_size=1000)
    elif doc_store_type == "qdrant":
        document_store = QdrantDocumentStore(
            path=index_dir,
            index=index_name,
            embedding_dim=embedding_dim,
            recreate_index=recreate_index,
            progress_bar=False,
        )

    if recreate_index or not index_exists or doc_store_type in ["inmemory"]:
        print("Initializing index...")
        docs_to_index = make_document_index(metadata, doc_ids, metadata_values=metadata_values, use_prefix=use_prefix)

        if isinstance(document_store, InMemoryDocumentStore):
            document_store.delete_documents()
        print("Writing documents to document store...")
        document_store.write_documents(docs_to_index)
        print("Done.")

    print(f"Documents in index: {document_store.get_document_count()}")

    print("Initializing retrievers...")
    params = {}
    pipeline = Pipeline()
    inputs = []
    if do_sparse:
        print("Sparse retriever...")
        sparse_retriever = BM25Retriever(document_store=document_store)
        pipeline.add_node(component=sparse_retriever, name="SparseRetriever", inputs=["Query"])
        params["SparseRetriever"] = {"top_k": top_k_sparse}
        inputs.append("SparseRetriever")
    if do_dense:
        print("Dense retriever...")
        dense_retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model=model_name,
            model_format=model_format,
            use_gpu=True,
            scale_score=scale_score,
            progress_bar=False,
        )
        if recreate_index or not index_exists or doc_store_type in ["inmemory"]:
            document_store.update_embeddings(retriever=dense_retriever, update_existing_embeddings=False)
        pipeline.add_node(component=dense_retriever, name="DenseRetriever", inputs=["Query"])
        params["DenseRetriever"] = {"top_k": top_k_dense}
        inputs.append("DenseRetriever")
    if do_join and do_sparse and do_dense:
        print("Joiner...")
        join_documents = JoinDocuments(join_mode=join_documents)
        pipeline.add_node(component=join_documents, name="JoinDocuments", inputs=inputs)
        params["JoinDocuments"] = {"top_k_join": top_k_join}
        inputs = ["JoinDocuments"]
    if do_rerank:
        print("Reranker...")
        rerank = SentenceTransformersRanker(model_name_or_path=reranker_name, use_gpu=True)
        pipeline.add_node(component=rerank, name="ReRanker", inputs=inputs)
        params["ReRanker"] = {"top_k": top_k_reranker}
    print("Done.")

    return pipeline, params, document_store


def parse_results(preds):
    results = {}
    for d in preds:
        results[d.meta["id"]] = d.score
    return results


def get_clean_rdids(variables_list):
    return sorted(list(set([_v.split("_")[0] for vs in variables_list for v in vs.split(";") for _v in v.split(",") if _v and "ZA" in _v])))


def run_retrieval(
        df, 
        meta_json, 
        model_name, 
        reranker_name, 
        emb_dim,
        index_dir,
        do_sparse=True,
        do_dense=True,
        do_join=True,
        do_rerank=True,
        top_k_sparse=10,
        top_k_dense=10,
        top_k_join=20,
        top_k_reranker=10,
        recreate_index=False,
        do_filter=True,
        filters=["research_data"],
        doc_store_type="qdrant",
        do_subset_datastore=False,
        metadata_values=[],
        model_format="sentence_transformers",
        use_prefix=False,
    ):
    queries = {}
    qrels_dict = {}
    run_dict = {}

    all_rd_ids = []
    if do_subset_datastore:
        all_rd_ids = get_clean_rdids(df[df["is_variable"].isin(["1", 1])]["variable"].tolist())
        print(f"Length of all research data: {len(all_rd_ids)}")

    if doc_store_type != "inmemory":
        pipeline, params, _document_store = make_pipeline(
            metadata=meta_json,
            doc_ids=all_rd_ids, 
            model_name=model_name, 
            reranker_name=reranker_name, 
            embedding_dim=emb_dim,
            index_dir=index_dir,
            do_sparse=do_sparse,
            do_dense=do_dense,
            do_join=do_join,
            do_rerank=do_rerank,
            top_k_sparse=top_k_sparse,
            top_k_dense=top_k_dense,
            top_k_join=top_k_join,
            top_k_reranker=top_k_reranker,
            recreate_index=recreate_index,
            doc_store_type=doc_store_type,
            metadata_values=metadata_values,
            model_format=model_format,
            use_prefix=use_prefix,
        )

    doc_groups = df.groupby(by="doc_id")
    for i, (doc_id, doc_group) in enumerate(tqdm(doc_groups)):
        print(f"Running retrieval for document group: {doc_id} ({i+1}/{len(doc_groups)}))...")
        
        year = 2024  
        rd_ids = get_clean_rdids(doc_group[doc_group["is_variable"].isin(["1", 1])]["variable"].tolist())
        if rd_ids == []:
            print(f"Skipping document '{rd_ids}' due to missing research data IDs.")
            continue
        
        if doc_store_type == "inmemory":
            pipeline, params, _document_store = make_inmemory_document_index_pipeline(
                metadata=meta_json,
                doc_ids=rd_ids,
                metadata_values=metadata_values,
                do_sparse=do_sparse,
                use_prefix=use_prefix,
                top_k_sparse=top_k_sparse,
            )

        if do_filter and doc_store_type != "inmemory":
            params["filters"] = defaultdict()

            if "research_data" in filters and rd_ids:
                params["filters"]["rd_id"] = rd_ids
            if "year" in filters and rd_ids:
                params["filters"]["$and"]["year"] = {"$lt": year}

        print("RD IDs:", rd_ids)

        candidate_vars = get_candidate_vars(rd_ids, meta_json)
        
        doc_queries = {}
        for j in range(doc_group.shape[0]):
            row = doc_group.iloc[j].fillna("")

            uid = row["uuid"]
            sent = "data: "+row["sentence"] if use_prefix else row["sentence"]
            vars = row.get("variable", "").split(";")
            vtype = row.get("type", "").split(";")
            vsubtype = row.get("subtype", "").split(";")
            type_short = row.get("type_short", "")
            subtype_short = row.get("subtype_short", "")
            vars = [v for vs in vars for v in vs.split(",") if v and "ZA" in v]

            query_meta = []
            for vs,t,s in zip(vars, vtype, vsubtype):
                for v in vs.split(","):
                    if v and v not in ["Unk"] and "http" not in v:
                        if v in candidate_vars:
                            query_meta.append({"variable": v, "type": t, "subtype": s})
                        else:
                            print(f"Variable '{v}' is not in the set of candidate variables.")
            
            if sent and rd_ids:
                qrels_dict[uid] = {v["variable"]: 1 for v in query_meta}

                doc_queries[uid] = {
                    "sentence": sent, 
                    "variables": query_meta,
                    "type_short": type_short,
                    "subtype_short": subtype_short,
                    "rd_ids": rd_ids,
                    }
        
        print(f"Running batch search for {len(doc_queries)} queries...")
        queries_str = [q["sentence"] for _,q in doc_queries.items()]
        predictions = pipeline.run_batch(
            queries=queries_str,
            params=params,
        )

        print("Parsing results...")
        for uid,d,q in zip(doc_queries.keys(), predictions["documents"], predictions["queries"]):
            assert q == doc_queries[uid]["sentence"]
            results = parse_results(d)
            run_dict[uid] = results
            if results == {}:
                print(f"Empty results for query: '{q}' ({uid})")

        for k,v in doc_queries.items():
            queries[k] = v

        assert len(queries) == len(qrels_dict) == len(run_dict)
        print(f"Total queries: {len(queries)}")
    print("Done.")
    return queries, qrels_dict, run_dict


def combine_data(paths):
    df = pd.DataFrame()
    for path in paths:
        _df = pd.read_csv(path, sep="\t")
        df = pd.concat([df, _df])
    df.reset_index(inplace=True)
    # keep only those documents which have duplicates
    non_dup_df = df.drop_duplicates()
    return df, df.loc[list(set(df.index)-set(non_dup_df.index))]


def load_dataframes(paths):
    df = pd.DataFrame()

    for path in paths:
        _df = pd.read_csv(path, sep="\t")
        df = pd.concat([df, _df])

    df.reset_index(inplace=True)
    return df


def fine_grained_results(df, qrels_dict, run_dict, cols, metrics):
    results = {}

    for col in cols:
        col_results = {}
        for col_name, col_group in df.groupby(by=col):
            col_uuids = col_group["uuid"].tolist()

            col_qrels_dict = {k:v for k,v in qrels_dict.items() if k in col_uuids and v != {}}
            col_run_dict = {k:v for k,v in run_dict.items() if k in col_uuids and k in col_qrels_dict}

            assert len(col_qrels_dict) == len(col_run_dict)

            if len(col_qrels_dict) > 0:
                if len([l for l in list(col_run_dict.values()) if l]) > 0:  # make sure that col_run_dict does not only contain empty runs
                    col_results[col_name] = evaluate(Qrels(col_qrels_dict), Run(col_run_dict), metrics)
            else:
                pass
        results[col] = col_results
    return results


def main(
    variables_path: str = "./data/gsim/survey_items.jsonl",
    index_dir: str = "./data/qdrant_index",
    data_dir: str = "./data/sild",
    model_name: str = "intfloat/multilingual-e5-base",
    reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    do_sparse: bool = True,
    do_dense: bool = True,
    do_join: bool = True,
    do_rerank: bool = False,
    top_k_sparse: int = 50,
    top_k_dense: int = 50,
    top_k_join: int = 100,
    top_k_reranker: int = 50,
    recreate_index: bool = False,
    do_subset_datastore: bool = False,
    do_filter: bool = True,
    filters: str = "research_data",
    metrics: str = "map@1,map@2,map@3,map@4,map@5,map@10,map@20,map@50,mrr,r-precision,recall@1,recall@2,recall@3,recall@4,recall@5,recall@10,recall@20,recall@50,ndcg@1,ndcg@2,ndcg@3,ndcg@4,ndcg@5,ndcg@10,ndcg@20,ndcg@50",
    eval_attribute_cols: str = "lang,doc_id,type_short,subtype_short",
    output_dir: str = "",
    doc_store_type: str = "qdrant",
    seed: int = 1234,
    metadata_values: str = "title,variable_label,question_text,sub_question,item_category,topic,item_categories,answer_categories",
    model_format: str = "sentence_transformers",
    use_validation_data: bool = True,
    test_set: str = "test",
    use_prefix: bool = False,
    eval_on_gold: bool = True,
    ):
    seed_everything(seed)
    os.makedirs(index_dir, exist_ok=True)

    config = locals()
    metrics = metrics.split(",")
    eval_attribute_cols = eval_attribute_cols.split(",")
    filters = filters.split(",")
    metadata_values = metadata_values.split(",")

    meta = load_jsonl(variables_path)
    meta_json = {m["url"].split(":")[-1]: m["variables"] for m in meta}

    sv4_train_path = f"{data_dir}/diff_train.tsv"
    sv4_rand_train_path = f"{data_dir}/rand_train.tsv"
    sv4_val_path = f"{data_dir}/diff_val.tsv"
    sv4_rand_val_path = f"{data_dir}/rand_val.tsv"
    sv4_test_path = f"{data_dir}/diff_test.tsv"
    sv4_rand_test_path = f"{data_dir}/rand_test.tsv"

    set_mapping = {
        "train": sv4_train_path,
        "train_rand": sv4_rand_train_path,
        "val": sv4_val_path,
        "val_rand": sv4_rand_val_path,
        "test": sv4_test_path,
        "test_rand": sv4_rand_test_path,
    }

    if use_validation_data:  # evaluate on data that is not used as test data in Diff or Rand
        test_df = load_dataframes([sv4_test_path, sv4_rand_test_path])
        train_df = load_dataframes([sv4_train_path, sv4_val_path, sv4_rand_train_path, sv4_rand_val_path])
        train_df = train_df[~train_df["doc_id"].isin(test_df["doc_id"].unique())]
        train_df = train_df.drop_duplicates(subset=["sentence"])

        for did in train_df["doc_id"].unique():
            assert did not in test_df["doc_id"].unique()

        if eval_on_gold:
            var_rows = train_df[(train_df["is_variable"] == 1) & (train_df["variable"].str.contains("ZA"))]  # evaluate on the set of documents that are not in the test set in either sv4 or sv4_rand
        else:
            var_rows = train_df
    else:
        if test_set in set_mapping:
            test_df = load_dataframes([set_mapping[test_set]])
            test_df = test_df.drop_duplicates(subset=["sentence"])

            if eval_on_gold:
                var_rows = test_df[(test_df["is_variable"] == 1) & (test_df["variable"].str.contains("ZA"))]
            else:
                var_rows = test_df
        else:
            raise Exception(f"Unkown test_set chosen. Choose one of {list(set_mapping.keys())}")
        
    print("Test set size:", var_rows.shape[0])

    if model_name == "bm25":
        emb_dim = 0
    elif os.path.exists(model_name):
        head_path = os.path.join(model_name, "config.json")
        if not os.path.exists(head_path):
            head_path = os.path.join(model_name, "servable", "head", "config.json")
            head_json = load_json(head_path)
            emb_dim = head_json.get("input_embedding_size", "")
        else:
            head_json = load_json(head_path)
            emb_dim = head_json.get("hidden_size", "")
    else:
        try:
            emb_dim = requests.get(f"https://huggingface.co/{model_name}/raw/main/config.json").json().get("hidden_size", "")
            if not emb_dim:
                emb_dim = requests.get(f"https://huggingface.co/{model_name}/raw/main/config.json").json().get("max_position_embeddings", "")
            if not emb_dim:
                emb_dim = requests.get(f"https://huggingface.co/{model_name}/raw/main/config.json").json().get("d_model", "")
        except Exception:
            raise Exception(f"Could not find the embedding dimension for the model: {model_name}")

    print(f"Modle name: {model_name}")
    print(f"Model embeddings size: {emb_dim}")

    _queries, qrels_dict, run_dict = run_retrieval(
        var_rows, 
        meta_json, 
        model_name, 
        reranker_name, 
        emb_dim,
        index_dir=index_dir,
        do_sparse=do_sparse,
        do_dense=do_dense,
        do_join=do_join,
        do_rerank=do_rerank,
        top_k_sparse=top_k_sparse,
        top_k_dense=top_k_dense,
        top_k_join=top_k_join,
        top_k_reranker=top_k_reranker,
        recreate_index=recreate_index,
        do_filter=do_filter,
        filters=filters,
        doc_store_type=doc_store_type,
        do_subset_datastore=do_subset_datastore,
        metadata_values=metadata_values,
        model_format=model_format,
        use_prefix=use_prefix,
    )

    if len([l for l in list(run_dict.values()) if l]) > 0: # make sure that run_dict does not only contain empty runs
        qrels = Qrels(qrels_dict)
        run = Run(run_dict)

        results = evaluate(qrels, run, metrics)
        fg_results = fine_grained_results(var_rows, qrels_dict, run_dict, eval_attribute_cols, metrics)

        if output_dir:
            timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
            output_dir = os.path.join(output_dir, timestamp)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "results.json")

            output_json = {"config": config, "results": results, "fg_results": fg_results}

            with open(output_path, "w") as outf:
                json.dump(output_json, outf)

            qrels_path = os.path.join(output_dir, "qrels.json")
            with open(qrels_path, "w") as outf:
                json.dump(qrels_dict, outf)

            run_path = os.path.join(output_dir, "run.json")
            with open(run_path, "w") as outf:
                json.dump(run_dict, outf)

            print(f"Results saved to {output_dir}")
            

if __name__ == "__main__":
    typer.run(main)