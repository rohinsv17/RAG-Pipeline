"""
Rohin Vijay
Evaluation file for RAG
--------------------------------
Runs evaluation Golden Dataset against RAG and scores metrics based on output
"""

import json
import os
import asyncio
from openai import AsyncOpenAI
from ragas.metrics.collections import (
    AnswerRelevancy,
    Faithfulness,
    ContextRecall,
)
from ragas.metrics import DiscreteMetric
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from config import (
    OLLAMA_BASE_URL,
    OLLAMA_API_KEY,
    EMBEDDING_MODEL,
    LLM_MODEL,
    GOLDEN_DATASET_PATH,
    EVAL_RECORDS_PATH,
)

# Evaluation class scores Pipeline against golden dataset and updates json file record
class Evaluator:

    # Constructor sets up Ragas with Ollama
    def __init__(self):
        print("[Evaluator] Initializing...")

        # Initializes the Ollama client for Ragas
        ollama_client = AsyncOpenAI(api_key = OLLAMA_API_KEY, base_url = OLLAMA_BASE_URL)

        # Makes the LLM compatible with Ragas
        self.ragas_llm = llm_factory(LLM_MODEL, provider = "openai", client = ollama_client)

        # Makes the embedding model compatible with Ragas
        self.ragas_embeddings = embedding_factory("openai", model = EMBEDDING_MODEL, client = ollama_client)

        # Initializes Ragas Retrieval Scorer
        self.retrieval_scorer = ContextRecall(llm = self.ragas_llm)

        # Initializes Ragas Relevancy Scorer
        self.relevancy_scorer = AnswerRelevancy(llm = self.ragas_llm, embeddings = self.ragas_embeddings)

        # Initializes Ragas Groundedness Scorer
        self.groundedness_scorer = Faithfulness(llm = self.ragas_llm)

        # Initializes Ragas Completeness Scorer
        self.completeness_scorer = DiscreteMetric(name = "Completeness", allowed_values = ["pass", "fail"], prompt = "Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'.\nResponse: {response} Grading Notes: {grading_notes}")


    # Loads Golden Dataset file
    def loadGoldenDataset(self) -> list[dict]:

        # Checks the golden dataset file exists before trying to open it
        if(not os.path.exists(GOLDEN_DATASET_PATH)):
            raise FileNotFoundError(f"Golden dataset not found at '{GOLDEN_DATASET_PATH}'.")
 
        # Opens json file and assigns to variable
        with open(GOLDEN_DATASET_PATH, "r", encoding = "utf-8") as file:
            dataset = json.load(file)

        # Validates that each entry has the three required fields
        for i, entry in enumerate(dataset):
            for field in ["question", "expected_chunks", "expected_response"]:
                if(field not in entry):
                    raise ValueError(f"Entry {i} is missing '{field}'")
 
        print(f"[Evaluator] Loaded {len(dataset)} entries from golden dataset\n")
        return dataset
    


    # == RETRIEVAL SCORE ==
 
    # Asynchronously scores how well the retrieved chunks match the expected chunks using Ragas ContextRecall
    async def scoreRetrieval(self, question: str, retrieved_contexts: list[str], expected_chunks: list[str]) -> float:

        # Concatenates expected chunks into a single reference string
        expected_reference = " ".join(expected_chunks)

        # Uses Ragas to score the retrieval of the chunks to the expected chunks
        result = await self.retrieval_scorer.ascore(
            user_input = question,
            retrieved_contexts = retrieved_contexts,
            reference = expected_reference,
        )

        return round(result.value, 4)


    # == RELEVANCY SCORE ==
 
    # Asynchronously scores how relevant the answer is to the question using Ragas AnswerRelevancy
    async def scoreRelevancy(self, question: str, actual_response: str) -> float:

        # Uses Ragas to score the relevancy of the answer to the question
        result = await self.relevancy_scorer.ascore(
            user_input = question,
            response = actual_response,
        )
 
        return round(result.value, 4)
    

    # == GROUNDEDNESS SCORE ==
 
    # Asynchronously scores how faithful the answer is to the retrieved chunks using Ragas Faithfullness
    async def scoreGroundedness(self, question: str, actual_response: str, retrieved_contexts: list[str]) -> float:

        # Uses Ragas to score the groundedness of the answer to the question
        result = await self.groundedness_scorer.ascore(
            user_input = question,
            response = actual_response,
            retrieved_contexts = retrieved_contexts,
        )

        return round(result.value, 4)
    

    # == COMPLETENESS SCORE ==
 
    # Sycnronously scores how complete the LLM is to the question
    def scoreCompleteness(self, actual_response: str, expected_response: str) -> str:

        # Uses Ragas to score the completeness of the answer to the question
        result = self.completeness_scorer.score(
            llm = self.ragas_llm,
            response = actual_response,
            grading_notes = expected_response,
        )

        return result.value
     
    # Loads existing eval records from disk or returns empty structure if none exist
    def loadRecords(self) -> dict:
 
        # Returns empty records structure if no records file exists yet
        if(not os.path.exists(EVAL_RECORDS_PATH)):
            return { "runs": [] }

        # Returns empty records structure if file exists but is empty
        if(os.path.getsize(EVAL_RECORDS_PATH) == 0):
            return { "runs": [] }
 
        # Reads and returns existing records from json file
        with open(EVAL_RECORDS_PATH, "r", encoding = "utf-8") as file:
            return json.load(file)
 
    # Appends a new evaluation run to the persistent records file
    def saveRecords(self, records: dict) -> None:
 
        # Adds updated records to json file
        with open(EVAL_RECORDS_PATH, "w", encoding = "utf-8") as file:
            json.dump(records, file, indent = 2, ensure_ascii = False)
 
        print(f"[Evaluator] Records saved to '{EVAL_RECORDS_PATH}'")
     
    # Runs full evaluation against golden dataset
    def evaluate(self) -> None:

        # Instantiates Retriever class
        from query import Retriever
        retriever = Retriever()

        # Calls dataset loading method
        dataset = self.loadGoldenDataset()

        # Initializes list to hold run results for evaluation gating
        run_results = []

        # Iterates through dataset to find an entry matching the question
        for i, entry in enumerate(dataset):
            question = entry["question"]
            expected_chunks = entry["expected_chunks"]
            expected_response = entry["expected_response"]
 
            print(f"[{i + 1}/{len(dataset)}]")
 
            # Runs the RAG pipeline to get the actual answer for this question
            actual_response  = retriever.ragOrchestrator(question, verbose = True)
 
            # Retrieves the actual chunks the pipeline fetched for scoring context
            query_embedding = retriever.embedQuery(question)
            retrieved_chunks = retriever.retrieveChunks(query_embedding, question)
            retrieved_chunks = retriever.rerankChunks(question, retrieved_chunks)
            retrieved_contexts = [chunk["text"] for chunk in retrieved_chunks]

            # Runs asyncio to score retrieval with a temporary event loop
            retrieval_score = asyncio.run(self.scoreRetrieval(question, retrieved_contexts, expected_chunks))

            # Runs asyncio to score answer relevancy with a temporary event loop
            relevancy_score  = asyncio.run(self.scoreRelevancy(question, actual_response))

            # Runs asyncio to score groundedness to retrieved chunks with a temporary event loop
            groundedness_score = asyncio.run(self.scoreGroundedness(question, actual_response, retrieved_contexts))

            # Scores completeness as either pass or fail
            completeness_score = self.scoreCompleteness(actual_response, expected_response)
    
            # Initializes a list to hold reranked chunks
            reranked_chunks = []

            for chunk in retrieved_chunks:
                reranked_chunks.append({
                    "source":chunk["source"],
                    "page":chunk["page"],
                    "score":chunk["score"],
                    "text":chunk["text"]
                })


            # Packages all inputs, outputs and scores for this entry
            result = {
                "question":question,
                "actual_response":actual_response,
                "metrics": {
                    "vector_search": {
                        "retrieval_score":retrieval_score,
                    },
                    "llm_generation": {
                        "relevancy_score":relevancy_score,
                        "groundedness_score":groundedness_score,
                    },
                    "rag_system": {
                        "completeness_score":completeness_score,
                    },
                },
                "retrieved_chunks":reranked_chunks
            }

            # Appends iteration result to the list of results
            run_results.append(result)

            # Prints per-entry scores to terminal as they complete
            print(f"\nVector Search Evaluation:")
            print(f"Retrieval Score: {retrieval_score:.4f}")

            print(f"\nLLM Generation Evaluation:")
            print(f"Relevancy Score: {relevancy_score:.4f}")
            print(f"Groundedness Score: {groundedness_score:.4f}")

            print(f"\nRAG System Evaluation:")
            print(f"Completeness Score: {completeness_score}\n")
            
 
        # Computes averages scores across all numeric metrics
        avg_retrieval = round(sum(r["metrics"]["vector_search"]["retrieval_score"]      for r in run_results) / len(run_results), 4)
        avg_relevancy = round(sum(r["metrics"]["llm_generation"]["relevancy_score"]    for r in run_results) / len(run_results), 4)
        avg_groundedness = round(sum(r["metrics"]["llm_generation"]["groundedness_score"] for r in run_results) / len(run_results), 4)

        # Converts completeness pass/fail to 1/0 for each entry then averages
        completeness_values = []
        for r in run_results:
            if(r["metrics"]["rag_system"]["completeness_score"] == "pass"):
                completeness_values.append(1)
            else:
                completeness_values.append(0)
        avg_completeness = round(sum(completeness_values) / len(completeness_values), 4)

        # Initializes a dictionary to hold to expected thresholds
        expected_thresholds = {}

        # Designateds scores based on thresholds
        if(avg_retrieval > 0.8):
            expected_thresholds["retrieval_score"] = "pass"
        else:
            expected_thresholds["retrieval_score"] = "fail"

        if(avg_relevancy > 0.9):
            expected_thresholds["relevancy_score"] = "pass"
        else:
            expected_thresholds["relevancy_score"] = "fail"

        if(avg_groundedness > 0.9):
            expected_thresholds["groundedness_score"] = "pass"
        else:
            expected_thresholds["groundedness_score"] = "fail"

        if(avg_completeness > 0.9):
            expected_thresholds["completeness_score"] = "pass"
        else:
            expected_thresholds["completeness_score"] = "fail"

 
        # Wraps this run with metadata and aggregate scores
        run_record = {
            "results": run_results,
            "expected_thresholds": expected_thresholds,
            "avg_metrics": {
                "vector_search": {
                    "avg_retrieval_score":avg_retrieval,
                },
                "llm_generation": {
                    "avg_relevancy_score":avg_relevancy,
                    "avg_groundedness_score":avg_groundedness,
                },
                "rag_system": {
                    "avg_completeness_score":avg_completeness,
                },
            },
        }
 
        # Loads existing records and appends this run
        records = self.loadRecords()
        records["runs"].append(run_record)
        self.saveRecords(records)
        
 # Entry point runs program 
if(__name__ == "__main__"):

    # Initializes tracerprovider before Evaluator class is instantiated
    from tracing import initTracing
    initTracing()

    # Initializes Evaluator and runs evaluation
    evaluator = Evaluator()
    evaluator.evaluate()