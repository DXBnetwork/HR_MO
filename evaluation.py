# import os
# import json
# from dotenv import load_dotenv
# from ragas.metrics import Faithfulness, AnswerRelevancy
# from ragas import evaluate, EvaluationDataset, SingleTurnSample
# from langchain_groq import ChatGroq
# from braintrust import init_logger

# load_dotenv()

# logger = init_logger(project="My Project")

# ragas_llm = ChatGroq(
#     model="llama-3.3-70b-versatile",
#     api_key=os.environ["GROQ_API_KEY"]
# )

# def evaluate_saved_queries():
#     # Load all queries
#     queries = []
#     with open("queries_to_evaluate.jsonl", "r") as f:
#         for line in f:
#             queries.append(json.loads(line))
    
#     print(f"Evaluating {len(queries)} queries...")
    
#     # Create RAGAS samples
#     samples = []
#     for q in queries:
#         sample = SingleTurnSample(
#             user_input=q["query"],
#             response=q["answer"],
#             retrieved_contexts=q["contexts"]
#         )
#         samples.append(sample)
    
#     # Evaluate all at once
#     dataset = EvaluationDataset(samples=samples)
    
#     results = evaluate(
#         dataset=dataset,
#         metrics=[Faithfulness(), AnswerRelevancy()],
#         llm=ragas_llm
#     )
    
#     # Convert to dict
#     results_df = results.to_pandas()
    
#     # Log each result to Braintrust with scores
#     for idx, row in results_df.iterrows():
#         query_data = queries[idx]
        
#         faithfulness = float(row.get('faithfulness', 0.0))
#         relevancy = float(row.get('answer_relevancy', 0.0))
        
#         logger.log(
#             input={"query": query_data["query"]},
#             output={"response": query_data["answer"]},
#             scores={
#                 "faithfulness": faithfulness,
#                 "answer_relevancy": relevancy
#             },
#             metadata={
#                 "timestamp": query_data["timestamp"],
#                 "num_contexts": len(query_data["contexts"]),
#                 "status": "success",
#                 "evaluated": True
#             }
#         )
        
#         print(f"✅ {query_data['query'][:50]}... - F:{faithfulness:.2f} R:{relevancy:.2f}")
    
#     # Archive the file
#     os.rename("queries_to_evaluate.jsonl", f"queries_evaluated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    
#     print(f"\n✅ Evaluated {len(queries)} queries")

# if __name__ == "__main__":
#     from datetime import datetime
#     evaluate_saved_queries()