from langchain.chains import GraphCypherQAChain
# tag::import-prompt-template[]
from langchain.prompts.prompt import PromptTemplate
# end::import-prompt-template[]

from llm import llm
from graph import graph

# tag::prompt[]
CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about ABC company's products and its sales.
Convert the user's question based on the schema.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Fine Tuning:

Asin and name of the product are the same 

Schema:
{schema}

Examples:
What are the features of product B09ZZZZZZ:
MATCH (c:Company)-[a:ACQUIRED]->(s:Sellers)-[r:SELLS]->(p:Products)
where p.asin="B09ZZZZZZ"
RETURN p

Tell me about asin B09ZZZZZZ:
MATCH (c:Company)-[a:ACQUIRED]->(s:Sellers)-[r:SELLS]->(p:Products)
where p.asin="B09ZZZZZZ"
RETURN p


List some sellers acquired by ABC:
MATCH (c:Company)-[a:ACQUIRED]->(s:Sellers)-[r:SELLS]->(p:Products)
RETURN DISTINCT s.seller_name

Tell me about ABC Company:
MATCH (c:Company)-[a:ACQUIRED]->(s)
RETURN c, s.seller_name

Question:
{question}

Cypher Query:
"""
# end::prompt[]

# tag::template[]
cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)
# end::template[]


# tag::cypher-qa[]
cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_prompt
)
# tag::cypher-qa[]