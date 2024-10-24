import streamlit as st
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains import RetrievalQA


from llm import llm, embeddings
"""
# tag::vector[]
neo4jvector = Neo4jVector.from_existing_index(
    embeddings,                              # <1>
    url=st.secrets["NEO4J_URI"],             # <2>
    username=st.secrets["NEO4J_USERNAME"],   # <3>
    password=st.secrets["NEO4J_PASSWORD"],   # <4>
    index_name="moviePlots",                 # <5>
    node_label="Movie",                      # <6>
    text_node_property="plot",               # <7>
    embedding_node_property="plotEmbedding", # <8>
    retrieval_query="""
RETURN
    node.plot AS text,
    score,
    {
        title: node.title,
        directors: [ (person)-[:DIRECTED]->(node) | person.name ],
        actors: [ (person)-[r:ACTED_IN]->(node) | [person.name, r.role] ],
        tmdbId: node.tmdbId,
        source: 'https://www.themoviedb.org/movie/'+ node.tmdbId
    } AS metadata
"""
)
# end::vector[]

# tag::retriever[]
retriever = neo4jvector.as_retriever()
# end::retriever[]

# tag::qa[]
kg_qa = RetrievalQA.from_chain_type(
    llm,
    chain_type="stuff",
    retriever=retriever,
)
# end::qa[]

# tag::generate-response[]
def generate_response(prompt):
    """
    Use the Neo4j Vector Search Index
    to augment the response from the LLM
    """

    # Handle the response
    response = kg_qa({"question": prompt})

    return response['answer']
# end::generate-response[]
"""

"""
The `kg_qa` can now be registered as a tool within the agent.

# tag::importtool[]
from langchain.tools import Tool
# end::importtool[]

# tag::importkgqa[]
from tools.vector import kg_qa
# end::importkgqa[]

# tag::tool[]
tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat not covered by other tools",
        func=llm.invoke,
        return_direct=True
        ),
    Tool.from_function(
        name="Vector Search Index",  # <1>
        description="Provides information about movie plots using Vector Search", # <2>
        func = kg_qa, # <3>
        return_direct=True
    )
]
# end::tool[]
"""