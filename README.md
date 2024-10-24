# LLM Project
# chatbot_llm_rag_neo4j_mistralai
We create a chatbot using LLM mistalai or OPENAI, Neo4j and Streamlit
The openAI could be replaced here with mistral ai

## Create Conda or Python Environment

### 1. Create Conda Environment 
- [Install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html)

```
conda create --name llm_env python=3.9
conda activate llm_env
```

### Create Python Environment 
```
python3 -m venv llm_env
source llm_env/bin/activate

```

### 2. Install Requirements.txt

```
pip install -r requirement.txt
```

Add the streamlit credentials inside the .streamlit->secrets.toml

### 3. Add the credentials for the OpenAI or MistralAI or any other LLM 

### 4. To run the streamlit app run the below 

steamlit run bot.py