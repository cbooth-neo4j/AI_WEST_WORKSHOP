# %% [markdown]
# ## Run this in your terminal to install the necessary libraries
# pip install graphdatascience neo4j dotenv langchain langchain_openai langgraph matplotlib seaborn pydantic openai gradio

# %% [markdown] 
# ### Importing the necessary libraries
import os
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase, RoutingControl
from langchain_openai import OpenAIEmbeddings
from graphdatascience import GraphDataScience
import matplotlib.pyplot as plt
import seaborn as sns


# %% [markdown] 
# ## Setup
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

# %% [markdown] 
# ### Read Data
# #### Load Synthetic Skills dataset
url = "https://raw.githubusercontent.com/neo4j-product-examples/genai-workshop/refs/heads/main/talent/data/expanded_skills.csv"
skills_df = pd.read_csv(url)

# %% [markdown] 
# #### Describe the dataset
skills_df.describe()

# %% [markdown] 
# #### Show the first 10 rows of the dataset
skills_df.head(10)

# %% [markdown] 
# #### Split the skills column into a list
skills_df['skills'] = skills_df['skills'].str.split(', ')
skills_df.head()

# %% [markdown] 
# ## Create the Graph
# #### Connect to Neo4j database
driver = GraphDatabase.driver(
    HOST,
    auth=(USERNAME, PASSWORD)
)
# %% [markdown]
# #### In case we want to split large files
def split_dataframe(df, chunk_size = 50_000):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

# %% [markdown]
# #### Test the connection
result = driver.execute_query(
    """
    MATCH (n) RETURN COUNT(n) as Count
    """,
    database_=DATABASE,
    routing_=RoutingControl.READ,
    result_transformer_= lambda r: r.to_df()
)
if result.iloc[0]['Count'] == 0:
    print("Connection successful but database is empty")
result

# %% [markdown] 
# ### Set Constraints
# We know what we will be loading. Set some constraints to make sure the data is loaded correctly.

# Set the Constraints on Person node
driver.execute_query(
    'CREATE CONSTRAINT IF NOT EXISTS FOR (n:Person) REQUIRE (n.email) IS NODE KEY',
    database_=DATABASE,
    routing_=RoutingControl.WRITE
)
# Set the Constraints on Skill node
driver.execute_query(
    'CREATE CONSTRAINT IF NOT EXISTS FOR (n:Skill) REQUIRE (n.name) IS NODE KEY',
    database_=DATABASE,
    routing_=RoutingControl.WRITE
)
# %% [markdown]
# Confirm Constraints 
schema_result_df  = driver.execute_query(
    'SHOW CONSTRAINTS',
    database_=DATABASE,
    routing_=RoutingControl.READ,
    result_transformer_= lambda r: r.to_df()
)
schema_result_df.head()

# %% [markdown]
# ### Load (:Person)-[:KNOWS]->(:Skill)
# Create a person and skills node and create a relationship between them
for chunk in split_dataframe(skills_df):
    records, summary, keys = driver.execute_query(
        """
        UNWIND $rows AS row
        MERGE (p:Person{email:row.email})
        SET p.name = row.name
        WITH p, row
        FOREACH(skill IN row.skills | MERGE (s:Skill{name:skill}) MERGE (p)-[:KNOWS]->(s) )
        RETURN COUNT(*) AS rows_processed
        """,
        database_=DATABASE,
        routing_=RoutingControl.WRITE,
        rows = chunk.to_dict('records')
    )

# %% [markdown]
# ## Explroe the Graph
# Show the whole graph with MATCH p=()-[]-() RETURN p LIMIT 25.
# * MATCH (n:Person) RETURN n LIMIT 25;
# * MATCH (n:Skill) RETURN n LIMIT 25;
# * MATCH p=()-[:KNOWS]->() RETURN p LIMIT 25;
# Any querys written via the python driver can be run in Aura!

# ### What skills does each person know?
person_skills_df = driver.execute_query(
    """
    MATCH (p:Person)-[:KNOWS]->(s:Skill)
    RETURN p.email AS email, p.name AS person_name, collect(s.name) AS skills
    """,
    database_=DATABASE,
    routing_=RoutingControl.READ,
    result_transformer_= lambda r: r.to_df()
)
person_skills_df


# %% [markdown]
# What are the most frequent skills?
skill_count_df = driver.execute_query(
    """
    MATCH (p:Person)-[:KNOWS]->(s:Skill)
    RETURN s.name, COUNT(DISTINCT p) AS knownByCount ORDER BY knownByCount DESC LIMIT 10
    """,
    database_=DATABASE,
    routing_=RoutingControl.READ,
    result_transformer_= lambda r: r.to_df()
)
skill_count_df

# %% [markdown]
# ### Finding similarty using the power of graph
# We can define the similarity of persons based on the number of skills that are overlapping.
person_name_1 = "Thomas Brown"

similar_skills_df = driver.execute_query(
    """
    MATCH path_1=(p1:Person{name: $person_name_1})-[:KNOWS]->(s1:Skill)
    MATCH path_2=(s1)<-[:KNOWS]-(p2:Person)
    WITH p1.name as person_1, p2.name as person_2, COLLECT(DISTINCT s1.name) as skill_list, COUNT(DISTINCT(s1)) as skill_count
    WHERE skill_count > 1 AND person_1 <> person_2
    RETURN * ORDER BY skill_count DESC
    """,
    database_=DATABASE,
    routing_=RoutingControl.READ,
    result_transformer_= lambda r: r.to_df(),
    person_name_1 = person_name_1
)
similar_skills_df

# %% [markdown]
# ## Do the same for all people
similar_skills_all_df = driver.execute_query(
    """
    MATCH path_1=(p1:Person)-[:KNOWS]->(s1:Skill)<-[:KNOWS]-(p2:Person)
    WHERE p1.name < p2.name
    WITH p1.name as person_1, p2.name as person_2, COLLECT(DISTINCT s1.name) as skill_list, COUNT(DISTINCT(s1)) as skill_count
    WHERE skill_count >= 1
    RETURN * ORDER BY skill_count DESC
    """,
    database_=DATABASE,
    routing_=RoutingControl.READ,
    result_transformer_= lambda r: r.to_df()
)
similar_skills_all_df

#%%[markdown]
# load the skill count to the database in a new relationship
for chunk in split_dataframe(similar_skills_all_df):
    records, summary, keys = driver.execute_query(
        """
        UNWIND $rows AS row
        MERGE (p1:Person{name:row.person_1})
        MERGE (p2:Person{name:row.person_2})
        MERGE (p1)-[s:SIMILAR_SKILLSET]->(p2)
        SET s.overlap = row.skill_count
        RETURN COUNT(*) AS rows_processed
        """,
        database_=DATABASE,
        routing_=RoutingControl.WRITE,
        rows = chunk.to_dict('records')
    )

# %% [markdown]
# Bring back all people with similar skills, sorting by skill_count
similar_skills_all_df = driver.execute_query(
    """
    MATCH path_1=(p1:Person)-[:KNOWS]->(s1:Skill)<-[:KNOWS]-(p2:Person)
    WHERE p1.name < p2.name
    WITH p1.name as person_1, p2.name as person_2, COLLECT(DISTINCT s1.name) as skill_list, COUNT(DISTINCT(s1)) as skill_count
    WHERE skill_count >= 1
    RETURN * ORDER BY skill_count DESC
    """,
    database_=DATABASE,
    routing_=RoutingControl.READ,
    result_transformer_= lambda r: r.to_df()
)
similar_skills_all_df

# %% [markdown]
# Load the skill count to the database in a new relationship
for chunk in split_dataframe(similar_skills_all_df):
    records, summary, keys = driver.execute_query(
        """
        UNWIND $rows AS row
        MERGE (p1:Person{name:row.person_1})
        MERGE (p2:Person{name:row.person_2})
        MERGE (p1)-[s:SIMILAR_SKILLSET]->(p2)
        SET s.overlap = row.skill_count
        RETURN COUNT(*) AS rows_processed
        """,
        database_=DATABASE,
        routing_=RoutingControl.WRITE,
        rows = chunk.to_dict('records')
    )

# %% [markdown]
# Take a minute to explore the SIMILAR_SKILLSET network in the database.

# * ```MATCH p=()-[:SIMILAR_SKILLSET]->() RETURN p LIMIT 50```
# * ```MATCH p=()-[s:SIMILAR_SKILLSET]->() WHERE s.overlap >= 2 RETURN p LIMIT 50```

# %% [markdown]
# ## Take a breather and let the instructor drive for a bit. 
# We'll briefly show some other data science capabilities of Neo4j.

# %% [markdown]
# ## Communities
# Let's run some Graph Data Science based on Persons and Skills.
gds = GraphDataScience.from_neo4j_driver(driver=driver)
gds.set_database(DATABASE)
gds.version()

# %% [markdown]
graph_name = "person_similarity_projection"
node_projection = ["Person"]
rel_projection = {"SIMILAR_SKILLSET": {"orientation": 'UNDIRECTED', "properties": "overlap"}, }
G, res = gds.graph.project(graph_name, node_projection, rel_projection)

# %% [markdown]
# Run the leiden algorithm for community detection
gds.leiden.write(
    G,
    writeProperty='leiden_community',
    relationshipWeightProperty='overlap',
    maxLevels=100,
    gamma=1.5,
    theta=0.001,
    concurrency = 1,
    randomSeed = 42
)
# %% [markdown]
communities_df = driver.execute_query(
    """
    MATCH (p:Person)
    RETURN p.leiden_community AS Community, COUNT(*) as MemberCount
    """,
    database_=DATABASE,
    routing_=RoutingControl.READ,
    result_transformer_= lambda r: r.to_df()
)
communities_df

# %% [markdown]
# Check communities based on people with high overlap
community_check_df = driver.execute_query(
    """
    MATCH (p1:Person)-[s:SIMILAR_SKILLSET]->(p2:Person)
    WHERE s.overlap > 2
    RETURN s.overlap AS Overlap, p1.name AS Person1, p1.leiden_community AS Community1, p2.name AS Person2, p2.leiden_community AS Community2
    """,
    database_=DATABASE,
    routing_=RoutingControl.READ,
    result_transformer_= lambda r: r.to_df()
)
community_check_df

# %% [markdown]
# Check communities based on skills
communities_skills_df = gds.run_cypher('''
    MATCH (p:Person)-[:KNOWS]->(s) WHERE (p.leiden_community) IS NOT NULL
    WITH p.leiden_community AS leiden_community, s.name as skill, count(*) as cnt
    WHERE cnt > 5
    RETURN *
    ORDER BY leiden_community, cnt DESC
''')
communities_skills_df

# %% [markdown]
# Plot communities with their skill count
df = gds.run_cypher("""
MATCH (p:Person)-[:KNOWS]->(s) WHERE (p.leiden_community) IS NOT NULL
RETURN p.leiden_community AS leiden_community, s.name as skill, count(*) as cnt
""")
pivot_table = df.pivot(index="skill", columns="leiden_community", values="cnt").fillna(0)
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table, cmap="Blues", linewidths=0.5)
plt.xlabel("Community")
plt.ylabel("Skill")
plt.title("Skill Distribution Heatmap per Community")
plt.show()


# %% [markdown]
# Drop the projection to free up resources
G.drop()

# Let's run some Graph Data Science based on Persons and Skills.


# %% [markdown]
# ## Vector time! 
# Since the communities don't really make sense (due to the randomness of the skills for persons) we can try the similarity based on the semantic meaning.

# Let's find similar skills semantically
url = 'https://raw.githubusercontent.com/neo4j-product-examples/genai-workshop/refs/heads/main/talent/data/skills_embeddings.csv'
skills_embeddings_df = pd.read_csv(url)
skills_embeddings_df.head()
type(skills_embeddings_df['Embedding'].iloc[0][0])
skills_embeddings_df['Embedding'] = skills_embeddings_df['Embedding'].apply( lambda x: [ float(i) for i in x.strip("[]").split(", ")] )
type(skills_embeddings_df['Embedding'].iloc[0][0])
skills_embeddings_df.head()

# %% length of the embedding
len(skills_embeddings_df['Embedding'].iloc[0])

# %% [markdown]
# Add embeddings to the database and take a look in the browser!
for chunk in split_dataframe(skills_embeddings_df):
    records, summary, keys = driver.execute_query(
        """
        UNWIND $rows AS row
        MATCH (s:Skill{name: row.Skill})
        SET s.embedding = row.Embedding
        SET s.description = row.Description
        WITH s
        CALL db.create.setNodeVectorProperty(s, "embedding", s.embedding)
        RETURN COUNT(*) AS rows_processed
        """,
        database_=DATABASE,
        routing_=RoutingControl.WRITE,
        rows = chunk.to_dict('records')
    )

# %% [markdown]
# Vectors for Semantic Meaning (index the vectors for faster performance)
driver.execute_query(
    """
    CREATE VECTOR INDEX `skill-embeddings` IF NOT EXISTS
    FOR (s:Skill) ON (s.embedding)
    OPTIONS {
        indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        } 
    }
    """,
    database_=DATABASE,
    routing_=RoutingControl.WRITE
)    

indexes_result_df  = driver.execute_query(
    'SHOW INDEXES',
    database_=DATABASE,
    routing_=RoutingControl.READ,
    result_transformer_= lambda r: r.to_df()
)
indexes_result_df

# %% [markdown]
# ## Semantic Search
# Take some Skill and find relevant other Skills: "Python", "Java", "Git", "CI/CD", "AWS", "Data Visualization", "Power BI", "R"".
skill_search = "Power BI"

similar_skills_df  = driver.execute_query(
    """
    MATCH (s:Skill{name: $skill_search})
    CALL db.index.vector.queryNodes("skill-embeddings", 10, s.embedding) YIELD node, score
    WITH node as skill, score ORDER BY score DESC
    WHERE node.name <> s.name AND score > 0.9
    RETURN skill.name, score
    """,
    database_=DATABASE,
    routing_=RoutingControl.READ,
    result_transformer_= lambda r: r.to_df(),
    skill_search = skill_search
)
similar_skills_df

# %% [markdown]
# Create relationships for similar semantic skills
driver.execute_query(
    """
    CALL apoc.periodic.iterate(
        "MATCH (skill1:Skill) RETURN skill1",
        "WITH skill1 
        CALL db.index.vector.queryNodes('skill-embeddings', 10, skill1.embedding) YIELD node, score
        WITH skill1, node as skill2, score ORDER BY score DESC
        WHERE skill1.name < skill2.name AND score > 0.92
        MERGE (skill1)-[s:SIMILAR_SEMANTIC]->(skill2)
        SET s.score = score   
        ",
        {batchSize: 1000}
    )
    """,
    database_=DATABASE,
    routing_=RoutingControl.WRITE,
    result_transformer_= lambda r: r.to_df()
)
# %% [markdown]
# Let's look in Sandbox to see how these relationships look like.

# * ```MATCH p=()-[:SIMILAR_SEMANTIC]->() RETURN p```

# %% [markdown]
# What are similar skills in the database now?
similar_skills_df  = driver.execute_query(
    """
    MATCH (s1:Skill)-[r:SIMILAR_SEMANTIC]-(s2:Skill)
    WHERE s1.name < s2.name
    RETURN s1.name AS skill1, r.score AS score, s2.name AS skill2
    ORDER BY score DESC
    """,
    database_=DATABASE,
    routing_=RoutingControl.READ,
    result_transformer_= lambda r: r.to_df()
)
similar_skills_df

# %% [markdown]
# Now we can find similar people based on semantic similarity
similar_persons_df  = driver.execute_query(
    """
    MATCH (p1:Person)-[:KNOWS]->(s:Skill)
    WITH p1, COLLECT(s.name) as skills_1
    CALL (p1, p1){
      MATCH (p1)-[:KNOWS]->(s1:Skill)-[r:SIMILAR_SEMANTIC]-(s2:Skill)<-[:KNOWS]-(p2:Person)
      RETURN p1 as person_1, p2 as person_2, SUM(r.score) AS score
      UNION 
      MATCH (p1)-[r:SIMILAR_SKILLSET]-(p2:Person)
      RETURN p1 as person_1, p2 AS person_2, SUM(r.overlap) AS score
    }
    WITH person_1.name as person_1, skills_1, person_2, SUM(score) as score
    WHERE score > 3
    MATCH (person_2)-[:KNOWS]->(s:Skill)
    RETURN person_1, skills_1,  person_2.name as person_2, COLLECT(s.name) as skills_2, score
    ORDER BY score DESC
    """,
    database_=DATABASE,
    routing_=RoutingControl.READ,
    result_transformer_= lambda r: r.to_df()
)
similar_persons_df
# %% [markdown]
# # Calculate for all of them wuith a score of >3
similar_persons_df  = driver.execute_query(
    """
    MATCH (p1:Person)-[:KNOWS]->(s:Skill)
    WITH p1, COLLECT(s.name) as skills_1
    CALL (p1, p1){
      MATCH (p1)-[:KNOWS]->(s1:Skill)-[r:SIMILAR_SEMANTIC]-(s2:Skill)<-[:KNOWS]-(p2:Person)
      RETURN p1 as person_1, p2 as person_2, SUM(r.score) AS score
      UNION 
      MATCH (p1)-[r:SIMILAR_SKILLSET]-(p2:Person)
      RETURN p1 as person_1, p2 AS person_2, SUM(r.overlap) AS score
    }
    WITH person_1.name as person_1, skills_1, person_2, SUM(score) as score
    WHERE score > 3
    MATCH (person_2)-[:KNOWS]->(s:Skill)
    RETURN person_1, skills_1,  person_2.name as person_2, COLLECT(s.name) as skills_2, score
    ORDER BY score DESC
    """,
    database_=DATABASE,
    routing_=RoutingControl.READ,
    result_transformer_= lambda r: r.to_df()
)
similar_persons_df


# %%
