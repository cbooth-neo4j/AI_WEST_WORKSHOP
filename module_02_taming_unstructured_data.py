
# %% [markdown]
# # Module 2: Taming Unstructured Data
# This module focuses on processing and analyzing unstructured data using Neo4j and AI capabilities.
# We'll explore techniques for:
# - Extracting structured information from text
# - Building knowledge graphs from unstructured sources
# - Combining graph data with language & embedding models

# %% [markdown] 
# ## Setup again
import os
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase, RoutingControl
from pydantic import BaseModel, Field
from openai import OpenAI
import json
from typing import List

# ### Load environment variables
load_dotenv()

## Neo4j
HOST = os.environ['NEO4J_URI']
USERNAME = os.environ['NEO4J_USERNAME'] 
PASSWORD = os.environ['NEO4J_PASSWORD']
DATABASE = os.environ['NEO4J_DATABASE']

## AI
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
LLM = os.environ['LLM']
EMBEDDINGS_MODEL = os.environ['EMBEDDINGS_MODEL']

driver = GraphDatabase.driver(
    HOST,
    auth=(USERNAME, PASSWORD)
)

driver.execute_query(
    """
    MATCH (n) RETURN COUNT(n) as Count
    """,
    database_=DATABASE,
    routing_=RoutingControl.READ,
    result_transformer_= lambda r: r.to_df()
)

# %% [markdown]
# ## Unstructured Data
# Let's define some unstructured data from our hosts

list_of_bio = [
    """ 
    Chris Booth, Solution Engineer at Neo4j. Chris excels at Natural Language Processing, Chatbots and GraphRAG. Want to know more,
    drop him an email at chris.booth@neo4j.com
    """,
    """ 
    Abe Mauleon, MD at AI West. His interest in deep learning started with computer vision, which is now complemented by AI Agents and LLMs. 
    If you want to hear more, drop him an email at info@aiwest.co.uk 
    """,
]

# %% [markdown]
# ### Define the Domain Model
# Pydantic Models are simply classes which inherit from BaseModel and define fields as annotated attributes.

class Skill(BaseModel):
    """
    Represents a professional skill or knowledge of a person.
    """
    name: str = Field(..., description="Sortened name of the skill")
    
class Person(BaseModel):
    """
    Represents a person with a name.
    """
    name: str = Field(..., description="Full name of person")
    email: str = Field(..., description="A persons email address")
    skills: List[Skill] = Field(..., description="List of skills known by the person"
    )
    
class PersonList(BaseModel):
    persons:List[Person]

system_message = """
    You are an expert in extracting structured information from person resumes.
    Identify key details such as:
    - Name of the person
    - Email address of the person
    - Skills known by the person
    
    Present the extracted information in a clear, structured format. Be concise, focusing on:
    - Key skills
    - Full name of person
    Ignore nick names, titles or roles and company information be short and consise with skills
"""

client = OpenAI()

# %% [markdown]
# ## Convert unstructured data to structured data

def extract(document, model=LLM, temperature=1):
    response = client.beta.chat.completions.parse(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": document},
        ],
        response_format=Person,
    )
    return json.loads(response.choices[0].message.content)

# %%
rows = []
for text in list_of_bio:
    data = extract(text)
    rows.append(data)
rows

structured_data = PersonList.model_validate({'persons':rows})

# %%
for k, details_list in structured_data.model_dump().items():
    print(f"{k}")
    for details in details_list:
        for key, value in details.items():
            print(f"  {key}: {value}")
        print()

# %% [markdown]
# ## Graph Creation
# Now that data is structured and validated, we can save it to neo4j.
records, summary, keys = driver.execute_query(
    """
        UNWIND $rows AS row
        MERGE (p:Person{email:row.email})
        SET p.name = row.name
        WITH p, row
        FOREACH (skill IN row.skills | MERGE (s:Skill{name:skill.name}) MERGE (p)-[:KNOWS]->(s) )
        RETURN COUNT (*) AS rows_processed
    """,
    database_=DATABASE,
    routing_=RoutingControl.WRITE,
    rows = rows
)

# %% [markdown]
# Check the browser with the following:
# * ```MATCH p=(n:Person {name: "Chris Booth"})-[:KNOWS]->(:Skill) RETURN p```
# * ```MATCH p=(n:Person {name: "Abe Mauleon"})-[:KNOWS]->(:Skill) RETURN p```

# %%
