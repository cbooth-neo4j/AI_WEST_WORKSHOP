# %% [markdown]
# # Module 3 - GraphRAG and Agents
# This module has the following objectives:
#
# * Experiment with queries for an Agent
# * Define Tooling
# * Create agents with the available tools
# * Chatbot for an Agent
# * Text2Cypher (if we have time)

# %% [markdown]
# As before, let's setup the environment
import os
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase, RoutingControl
from langchain.schema import HumanMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import create_react_agent
from openai import OpenAI
from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import gradio as gr
import time

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
# ## Agent Thinking
'''
Let's say we want to build an Agent with multiple tools. Let's try to provide the following functionality:

### 1. Retrieve the skills of a person
- **Input**: Person
- **Output**: Skills
- **Example**: What skills does Kristof Neys have?

### 2. Retrieve similar skills to other skills
- **Input**: Skills  
- **Output**: Skills
- **Example**: What skills are similar to PowerBI and Data Visualization?

### 3. Retrieve similar persons to a person specified
- **Input**: Person
- **Output**: Person  
- **Example**: *"Which persons have similar skills as Kristof Neys?"*

### 4. Retrieve Persons based on a set of skills
- **Input**: Skills
- **Output**: Person
- **Example**: Which persons have Python and AWS experience?
'''

# %% [markdown]
# Set up an embedding model to vectorize user queries
embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)

# %% [markdown]
# ## Retrieve the skills of a person

# Find the connected skills given a person name.
person_name = "Lucy Turner"

person_skills_df = driver.execute_query(
    """
    MATCH (p:Person{name: $person_name})-[:KNOWS]->(s:Skill)
    RETURN p.name as name, COLLECT(s.name) as skills
    """,
    database_=DATABASE,
    routing_=RoutingControl.READ,
    result_transformer_= lambda r: r.to_df(),
    person_name = person_name
)

person_skills_df

# %% [markdown]
# ## Retrieve similar skills to other skills
# Retrieve skills based on a list of skills

skills = ['Contineous Delivery', 'Cloud Native', 'Security']
skills_vectors = embeddings.embed_documents(skills)
search_persons_with_skills_df = driver.execute_query(
    """
        UNWIND $skills_vectors AS v
        CALL db.index.vector.queryNodes('skill-embeddings', 3, TOFLOATLIST(v)) YIELD node, score
        WHERE score > 0.89
        OPTIONAL MATCH (node)-[:SIMILAR_SEMANTIC]-(s:Skill)
        WITH COLLECT(node) AS nodes, COLLECT(DISTINCT s) AS skills
        WITH nodes + skills AS all_skills
        UNWIND all_skills AS skill
        RETURN DISTINCT skill.name as skill_name
    """,
    database_=DATABASE,
    routing_=RoutingControl.READ,
    result_transformer_= lambda r: r.to_df(),
    skills_vectors = skills_vectors
)

search_persons_with_skills_df

# %% [markdown]
# ## Person Similarity

person_name_1 = "John Garcia"

person_similarity_community_df = driver.execute_query(
    """
    MATCH (p1:Person {name: $person_name_1})-[:KNOWS]->(s:Skill)
    WITH p1, COLLECT(s.name) as s1
    MATCH (p2:Person {leiden_community: p1.leiden_community})-[:KNOWS]->(s2:Skill)
    RETURN p1.name AS person_1, s1 AS skills_1, p1.leiden_community AS community, p2.name AS person_2, COLLECT(s2.name) AS skills_2
    """,
    database_=DATABASE,
    routing_=RoutingControl.READ,
    result_transformer_= lambda r: r.to_df(),
    person_name_1 = person_name_1
)
person_similarity_community_df

# %% [markdown]

# You can find all Skills in the community in the browser:
#
# ```MATCH p=(:Person{leiden_community: 88})-[:KNOWS]->(s:Skill) RETURN p```

# %% [markdown]
# ## Strategy - Similar Skillsets
# We can use the similar skillsets to find similar persons.
person_name_1 = "John Garcia"

person_similar_skillset_df = driver.execute_query(
    """
    MATCH (p1:Person {name: $person_name_1})-[:KNOWS]->(s:Skill)
    WITH p1, COLLECT(s.name) as s1
    MATCH (p1)-[r:SIMILAR_SKILLSET]-(p2:Person)-[:KNOWS]->(s2:Skill)
    WHERE r.overlap > 1
    RETURN p1.name AS person_1, s1 AS skills_1, r.overlap AS score, p2.name AS person_2, COLLECT(DISTINCT s2.name) AS skills_2
    ORDER BY score DESC
    """,
    database_=DATABASE,
    routing_=RoutingControl.READ,
    result_transformer_= lambda r: r.to_df(),
    person_name_1 = person_name_1
)
person_similar_skillset_df

# %% [markdown]
# ## Strategy - Similar Skillsets and Semantic Meaning
# WE cam use the Semantic Meaning and Skill overlap to find people with similar skills

person_name_1 = "John Garcia"

person_similarity_df = driver.execute_query(
    """
    MATCH (p1:Person {name: $person_name_1})-[:KNOWS]->(s:Skill)
    WITH p1, COLLECT(s.name) as skills_1
    CALL (p1){
      MATCH (p1)-[:KNOWS]->(s1:Skill)-[r:SIMILAR_SEMANTIC]-(s2:Skill)<-[:KNOWS]-(p2:Person)
      
      RETURN p1 as person_1, p2 as person_2, SUM(r.score) AS score
      UNION
      MATCH (p1)-[r:SIMILAR_SKILLSET]-(p2:Person)
      RETURN p1 as person_1, p2 AS person_2, SUM(r.overlap) AS score
    }
    WITH person_1.name as person_1, skills_1, person_2, SUM(score) as score
    WHERE score >= 1
    MATCH (person_2)-[:KNOWS]->(s:Skill)
    RETURN person_1, skills_1,  person_2.name as person_2, COLLECT(s.name) as skills_2, score
    ORDER BY score DESC LIMIT 5
    """,
    database_=DATABASE,
    routing_=RoutingControl.READ,
    result_transformer_= lambda r: r.to_df(),
    person_name_1 = person_name_1
)
person_similarity_df

# %% [markdown]
# ## Recommendation of Person given on set of skills
# ### Vector Index Search
skills = ['AWS', 'Security']
skills_vectors = embeddings.embed_documents(skills)

# %% [markdown]
# We get the approximate top 10 nearest nodes to the search vector v and take the 3 first returned. Then put them together in a list (skill_list) and does same ranking as before (number of skills)

nn_df = driver.execute_query(
    """UNWIND $skills_vectors AS v
    CALL db.index.vector.queryNodes('skill-embeddings', 3, TOFLOATLIST(v)) YIELD node, score
    WHERE score > 0.85
    WITH v as embedding, COALESCE(COLLECT(node.name), []) AS top
    RETURN *
    """,
    database_=DATABASE,
    routing_=RoutingControl.READ,
    result_transformer_= lambda r: r.to_df(),
    skills_vectors = skills_vectors
)
nn_df['skills'] = skills
cols = list(nn_df.columns)[-1:] + list(nn_df.columns)[:-1]
nn_df = nn_df[cols]

nn_df

find_persons_given_skills_df = driver.execute_query(
    """
    UNWIND $skills_vectors AS v
    CALL db.index.vector.queryNodes('skill-embeddings', 3, TOFLOATLIST(v)) YIELD node, score
    WHERE score > 0.85
    OPTIONAL MATCH (node)-[:SIMILAR_SEMANTIC]-(s:Skill)
    WITH COLLECT(node) AS nodes, COLLECT(DISTINCT s) AS skills
    WITH nodes + skills AS all_skills
    UNWIND all_skills AS skill
    MATCH (p:Person)-[:KNOWS]->(skill)
    RETURN p.name AS person, COUNT(DISTINCT(skill)) AS skill_count, COLLECT(DISTINCT(skill.name)) as similar_skills
    ORDER BY skill_count DESC LIMIT 10
    """,
    database_=DATABASE,
    routing_=RoutingControl.READ,
    result_transformer_= lambda r: r.to_df(),
    skills_vectors = skills_vectors
)

find_persons_given_skills_df
# %% [markdown]
# Nearly there! Agents with GraphRAG
# Let's create a Retrieval agent

class Skill(BaseModel):
    """
    Represents a professional skill or knowledge of a person.
    """
    name: str = Field(..., description="Sortened name of the skill")


# %% [markdown]
# ## Tool 1 - Skills of a Person
def retrieve_skills_of_person(person_name: str) -> pd.DataFrame:
    """Retrieve the skills of a person. Person is provided with it's name"""
    return driver.execute_query(
        """
        MATCH (p:Person{name: $person_name})-[:KNOWS]->(s:Skill)
        RETURN p.name as name, COLLECT(s.name) as skills
        """,
        database_=DATABASE,
        routing_=RoutingControl.READ,
        result_transformer_= lambda r: r.to_df(),
        person_name = person_name
    )
retrieve_skills_of_person('Mia Nelson') 

# %% [markdown]
# ## Tool 2 - Similar Skills
def find_similar_skills(skills: List[Skill]) -> pd.DataFrame:
    """Find similar skills to list of skills specified. Skills are specified by a list of their names"""
    skills = [s.name for s in skills]
    skills_vectors = embeddings.embed_documents(skills)
    return driver.execute_query(
    """
        UNWIND $skills_vectors AS v
        CALL db.index.vector.queryNodes('skill-embeddings', 3, TOFLOATLIST(v)) YIELD node, score
        WHERE score > 0.89
        OPTIONAL MATCH (node)-[:SIMILAR_SEMANTIC]-(s:Skill)
        WITH COLLECT(node) AS nodes, COLLECT(DISTINCT s) AS skills
        WITH nodes + skills AS all_skills
        UNWIND all_skills AS skill
        RETURN DISTINCT skill.name as skill_name
    """,
    database_=DATABASE,
    routing_=RoutingControl.READ,
    result_transformer_= lambda r: r.to_df(),
    skills_vectors = skills_vectors
)
find_similar_skills([Skill(name='Python')])

# %% [markdown]
# ## Tool 3 - Similar Persons
def person_similarity(person_name: str) -> pd.DataFrame:
    """Find a similar person to the one specified based on their skill similarity. Persons are provided with their name"""
    
    return driver.execute_query(
        """
        MATCH (p1:Person {name: $person_name})-[:KNOWS]->(s:Skill)
        WITH p1, COLLECT(s.name) as skills_1
        CALL (p1){
          MATCH (p1)-[:KNOWS]->(s1:Skill)-[r:SIMILAR_SEMANTIC]-(s2:Skill)<-[:KNOWS]-(p2:Person)
          RETURN p1 as person_1, p2 as person_2, SUM(r.score) AS score
          UNION 
          MATCH (p1)-[r:SIMILAR_SKILLSET]-(p2:Person)
          RETURN p1 as person_1, p2 AS person_2, SUM(r.overlap) AS score
        }
        WITH person_1.name as person_1, skills_1, person_2, SUM(score) as score
        WHERE score >= 1
        MATCH (person_2)-[:KNOWS]->(s:Skill)
        RETURN person_1, skills_1,  person_2.name as person_2, COLLECT(s.name) as skills_2, score
        ORDER BY score DESC LIMIT 5
        """,
        database_=DATABASE,
        routing_=RoutingControl.READ,
        result_transformer_= lambda r: r.to_df(),
        person_name = person_name
    )

person_similarity("Christopher Jackson")

# %% [markdown]
# ## Tool 4 - Persons based on skills
def find_person_based_on_skills(skills: List[Skill]) -> pd.DataFrame:
    """
    Find persons based on skills they have. Skills are specified by their names. 
    Note that similar skills can be found. These are considered similar. 
    """
    skills = [s.name for s in skills]
    skills_vectors = embeddings.embed_documents(skills)
    return driver.execute_query(
        """
        UNWIND $skills_vectors AS v
        CALL db.index.vector.queryNodes('skill-embeddings', 3, TOFLOATLIST(v)) YIELD node, score
        WHERE score > 0.89
        OPTIONAL MATCH (node)-[:SIMILAR_SEMANTIC]-(s:Skill)
        WITH COLLECT(node) AS nodes, COLLECT(DISTINCT s) AS skills
        WITH nodes + skills AS all_skills
        UNWIND all_skills AS skill
        MATCH (p:Person)-[:KNOWS]->(skill)
        RETURN p.name AS person, COUNT(DISTINCT(skill)) AS score, COLLECT(DISTINCT(skill.name)) as similar_skills
        ORDER BY score DESC LIMIT 10
        """,
        database_=DATABASE,
        routing_=RoutingControl.READ,
        result_transformer_= lambda r: r.to_df(),
        skills_vectors = skills_vectors
)
find_person_based_on_skills([Skill(name='Security'), Skill(name='Pandas')])

# %% [markdown]
# ## Setting up the Agent
llm = ChatOpenAI(model_name=LLM, temperature=1)
response = llm.invoke([HumanMessage(content="hi!")])
response.content

tools = [
    retrieve_skills_of_person, 
    find_similar_skills,
    person_similarity,
    find_person_based_on_skills,
]

llm_with_tools = llm.bind_tools(tools)

response = llm_with_tools.invoke([HumanMessage(content="What skills does Kristof Neys have?")])

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")

response = llm_with_tools.invoke([HumanMessage(content="What skills are similar to PowerBI and Data Visualization?")])

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")

response = llm_with_tools.invoke([HumanMessage(content="Which persons have similar skills as Kristof Neys?")])

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")

response = llm_with_tools.invoke([HumanMessage(content="Which persons have Python and AWS experience?")])

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")

# %% [markdown]
# ## Running Agents with LangGraph
agent_executor = create_react_agent(llm, tools)

response = agent_executor.invoke({"messages": [HumanMessage(content="hi!")]})

response["messages"]

def ask_to_agent(question):
    for step in agent_executor.stream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

# %% [markdown]
# Run some examples!
# * What skills are similar to PowerBI and Data Visualization?
# * Which persons have similar skills as Daniel Hill?
# * Which persons have Python and AWS experience?"
# * Who would you recommend i use for a AWS project and Python project?

question = "What skills does Daniel Hill have?"
ask_to_agent(question)

# %% [markdown]
# ## Chatbot interface
def user(user_message, history):
    if history is None:
        history = []
    history.append({"role": "user", "content": user_message})
    return "", history

def get_answer(history):
    steps = []
    full_prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
    
    for step in agent_executor.stream(
            {"messages": [HumanMessage(content=full_prompt)]},
            stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
        steps.append(step["messages"][-1].content)
    
    return steps[-1]

def bot(history):
    bot_message = get_answer(history)
    history.append({"role": "assistant", "content": ""})

    for character in bot_message:
        history[-1]["content"] += character
        time.sleep(0.01)
        yield history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        label="Chatbot on a Graph",
        avatar_images=[
            "https://png.pngtree.com/png-vector/20220525/ourmid/pngtree-concept-of-facial-animal-avatar-chatbot-dog-chat-machine-illustration-vector-png-image_46652864.jpg",
            "https://d-cb.jc-cdn.com/sites/crackberry.com/files/styles/larger/public/article_images/2023/08/openai-logo.jpg"
        ],
        type="messages", 
    )
    msg = gr.Textbox(label="Message")
    clear = gr.Button("Clear")

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot], chatbot
    )

    clear.click(lambda: [], None, chatbot, queue=False)

demo.queue()
demo.launch(share=True)

# %% [markdown]
# ## BONUS ROUND - Text2Cypher

text2cypher_prompt =  PromptTemplate.from_template(
    """
    Task: Generate a Cypher statement for querying a Neo4j graph database from a user input. 
    - Do not include triple backticks ``` or ```cypher or any additional text except the generated Cypher statement in your response.
    - Do not use any properties or relationships not included in the schema.
    
    Schema:
    {schema}
    
    #User Input
    {question}
    
    Cypher query:
    """
)

annotated_schema = """
    Nodes:
      Person:
        description: "A person in our talent pool."
        properties:
          name:
            type: "string"
            description: "The full name of the person. serves as a unique identifier."
          email:
            type: "string"
            description: "The email address of the person."
          leiden_community:
            type: "integer"
            description: "The talent community for the person.  People in the same talent segment share similar skills."
      Skill:
        description: "A professional skill."
        properties:
          name:
            type: "string"
            description: "The unique name of the skill."
    Relationships:
        KNOWS:
            description: "A person knowing a skill."
            query_pattern: "(:Person)-[:KNOWS]->(:Skill)"
    """

text2cypher_llm = ChatOpenAI(model=LLM, temperature=1)

@tool
def perform_aggregation_query(question: str) -> pd.DataFrame:
    """
    perform an aggregation query on the Neo4j graph database and obtain the results.
    """
    prompt = text2cypher_prompt.invoke({'schema': annotated_schema, 'question': question})
    query = text2cypher_llm.invoke(prompt).content
    print(f"executing Cypher query:\n{query}")
    return driver.execute_query(
        query,
        database_=DATABASE,
        routing_=RoutingControl.READ,
        result_transformer_= lambda r: r.to_df()
    )    

# %% [markdown]
perform_aggregation_query('describe communities by skills') 

# %% [markdown]
perform_aggregation_query('how many people share skills with Isabella Allen, and what are the skills')

# %% [markdown]
perform_aggregation_query('Can you list me a 5 random person name from the database?')