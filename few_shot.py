from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatOllama
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate,FewShotPromptTemplate
import langchain
import streamlit as sl



langchain.verbose = False

# Set up database connection
uri = "mssql+pyodbc://" # Ensure to modify this connection string according to your database server and authentication details
db = SQLDatabase.from_uri(uri, schema="SYNTHETIC", include_tables=['Hackathon_EDDC', 'REF_ETHNICITY', 'REF_GENDER', 'REF_TRIAGE_CATEGORY'])

# Set up models
sql_llm = ChatOllama(model="codellama")
gen_llm = ChatOllama(model="mistral")

# Few-Shot Examples
examples = [
    {
        "input": "How many patients are in the database?",
        "query": "SELECT COUNT(DISTINCT person_ID) AS total FROM [SYNTHETIC].[Hackathon_EDDC];"
    },
    {
        "input": "How many female patients are there?",
        "query": "SELECT COUNT(DISTINCT person_ID) FROM [SYNTHETIC].[Hackathon_EDDC] where sex = 2;"
    },
    {
        "input": "List all distinct modes of arrival recorded in the database.",
        "query": "SELECT DISTINCT mode_of_arrival FROM [SYNTHETIC].[Hackathon_EDDC];"
    },
    {
        "input": "Which hospital has the highest number of visits?",
        "query": "SELECT TOP 1 establishment_code, COUNT(*) AS visit_count FROM [SYNTHETIC].[Hackathon_EDDC] GROUP BY establishment_code ORDER BY visit_count DESC;"
    },
    {
        "input": "What is the average age of patients by sex?",
        "query": "SELECT r.gender, AVG(p.age) AS average_age FROM [SYNTHETIC].[Hackathon_EDDC] p JOIN [SYNTHETIC].[REF_GENDER] r ON p.sex = r.sex GROUP BY r.gender;"
    },
    {
        "input": "What percentage of patients have a triage category of 'Resuscitation'?",
        "query": "SELECT (COUNT(CASE WHEN t.triage_category = 1 THEN 1 END) * 100.0) / COUNT(*) AS resuscitation_percentage FROM [SYNTHETIC].[Hackathon_EDDC] p JOIN [SYNTHETIC].[REF_TRIAGE_CATEGORY] t ON p.triage_category = t.triage_category;"
    },
    {
        "input": "Which hospital has the highest number of mental health admissions?",
        "query": "SELECT TOP 1 establishment_code, COUNT(*) AS mental_health_admissions FROM [SYNTHETIC].[Hackathon_EDDC] WHERE mental_health_admission = 1 GROUP BY establishment_code ORDER BY mental_health_admissions DESC;"
    },
    {
        "input": "How many patients were admitted on weekends?",
        "query": "SELECT COUNT(*) AS weekend_admissions FROM [SYNTHETIC].[Hackathon_EDDC] WHERE DATEPART(weekday, presentation_datetime) IN (1, 7);"
    },
    {
        "input": "What is the least common mode of arrival for urgent cases?",
        "query": "SELECT TOP 1 mode_of_arrival, COUNT(*) AS arrival_count FROM [SYNTHETIC].[Hackathon_EDDC] WHERE triage_category = 3 GROUP BY mode_of_arrival ORDER BY arrival_count ASC;"
    }
]

example_prompt = PromptTemplate.from_template("User input: {input}\nSQL Query: {query}")

prefix = """
    You are a MS SQL expert. Given an input question, 
    create one syntactically correct MS SQL query to run,
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. 
    Wrap each column name in square brackets ([]) to denote them as delimited identifiers. 
    Pay attention to use only the column names you can see in the tables below. 
    Be careful to not query for columns that do not exisl. 
    Also, pay attention to which column is in which table. 
    Pay attention to use CAST(GETDATE() as date) function to get the current date, 
    if the question involves "today". Do not return more than {top_k} row.

    Here is the relevant table info:
    {table_info}

    Below are a number of examples of questions and their corresponding SQL queries.
"""

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix="User input: {input}\nSQL query: ",
    input_variables=["input", "top_k", "table_info"],
)

# Chain creation
generate_query = create_sql_query_chain(sql_llm, db, prompt)
execute_query = QuerySQLDataBaseTool(db=db)
answer_prompt = PromptTemplate.from_template(
    """[INST]
    Given the following user question, corresponding SQL query, and SQL result, answer the user question.
 
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer:
    [/INST]"""
)
rephrase_answer = answer_prompt | gen_llm | StrOutputParser()

# Streamlit UI
sl.title('Hospital Database Chatbot')
 
user_question = sl.text_input('Enter your question:', '')
if sl.button('Generate Query and Fetch Results'):

    query = generate_query.invoke({"question": user_question})
    sl.markdown(f"**SQL Query Generated:**\n```sql\n{query}\n```")
    result = execute_query.invoke(query)
    sl.markdown(f"**SQL Result:**\n {result}")
    answer = rephrase_answer.invoke({"question": user_question, "query": query, "result": result})
    sl.success(f"**Answer:**\n {answer}")