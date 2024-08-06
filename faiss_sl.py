from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from operator import itemgetter
import streamlit as sl



def main():
    sl.title("Chatbot")

    embeddings = OllamaEmbeddings(model='nomic')

    uri = "mssql+pyodbc://" # Ensure to modify this connection string according to your database server and authentication details
    db = SQLDatabase.from_uri(uri, schema="SYNTHETIC", include_tables=['Hackathon_EDDC', 'REF_ETHNICITY', 'REF_GENDER', 'REF_TRIAGE_CATEGORY', 'REF_PRIMARY_DIAGNOSIS'])

    sql_llm = ChatOllama(model="codellama")
    gen_llm = ChatOllama(model="llama2")

    vectorstore = FAISS.load_local('./db/faiss_index', embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key ="question",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )

    rag_template = """
        Please provide an answer to the current question exclusively, 
        when you receive a new question, focus on it and do not reply to previous question any more.
        Treat each question as an independent inquiry, without drawing upon prior conversations 
        or external data beyond the context.
        Context: {context}
        Please ensure your response is meticulously informed by the given context. 
        In instances where the latest question is unclear or ambiguous, 
        you may refer to Chat History for clarification purposes only. 
        Otherwise, prior conversations should not influence your response.
        If it is not feasible to furnish an accurate answer based solely on the available context, 
        refrain from speculative or unsupported assertions, and explicitly acknowledge any constraints in addressing the query.
        Question: {question}
    """

    rag_prompt = PromptTemplate(input_variables=["context", "question"], template=rag_template)

    rag_chain = ConversationalRetrievalChain.from_llm(
            llm=gen_llm,
            chain_type="stuff",
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": rag_prompt}
        )

    generate_query = create_sql_query_chain(sql_llm, db)
    execute_query = QuerySQLDataBaseTool(db=db)
    answer_prompt = PromptTemplate.from_template(
        """
        Given the following user question, corresponding SQL query, and SQL result, answer the user question.
    
        Question: {question}
        SQL Query: {query}
        SQL Result: {result}
        Answer:
        """
    )

    rephrase_answer = answer_prompt | gen_llm | StrOutputParser()

    def capture_results(query_result):
        query, execution_result = query_result['query'], query_result['result']
        final_answer = rephrase_answer.invoke({
            "question": query_result['question'],
            "query": query,
            "result": execution_result
        })
        return {
            "query": query,
            "execution_result": execution_result,
            "final_answer": final_answer
        }

    sql_chain = (
            RunnablePassthrough.assign(query=generate_query).assign(
                result=itemgetter("query") | execute_query
            )
            | RunnableLambda(capture_results)
        )

    classification_template = PromptTemplate.from_template(
        """You are good at classifying a question.
        Given the user question below, classify it as either being about `Database`, `Chat`.

        <If the question is about numbers in database classify the question as 'Database'>
        <If the question is about reports and documents, classify it as 'Chat'>

        <question>
        {question}
        </question>

        Classification:"""
    )

    classification_chain = classification_template | gen_llm | StrOutputParser()

    def route(info):
        if "database" in info["topic"].lower():
            return sql_chain
        elif "chat" in info["topic"].lower():
            return rag_chain
        else:
            return "I am sorry, I am not allowed to answer about this topic."
        
    question = sl.text_input("Enter your question here:")

    if sl.button('Ask'):
        full_chain = {
            "topic": classification_chain,
            "question": lambda x: x["question"],
        } | RunnableLambda(route)

        result = full_chain.invoke({"question": question})

        # Handle response format for "chat" topic
        if isinstance(result, dict) and 'topic' in result:
            answer = result["answer"]
            source_documents = result["source_documents"]
            if source_documents:
                sl.text_area("Answer:", value=answer, height=200)
                sl.markdown("---")
                for source_idx, source_doc in enumerate(source_documents):
                    with sl.expander(f"Source {source_idx + 1}"):
                        sl.text(source_doc.page_content)
            else:
                sl.text_area("Answer:", value=answer)

        # Handle response format for "database" topic
        elif isinstance(result, dict) and 'final_answer' in result:

            # Display the SQL query
            sl.subheader("Generated SQL Query:")
            sl.code(result["query"])

            # Display the SQL execution result
            sl.subheader("Execution Result")
            sl.caption(result["execution_result"])

            # Display the final rephrased answer
            sl.subheader("Final Answer")
            sl.success(result["final_answer"])
        else:
            sl.text_area("Result:", value=result)
 
if __name__ == "__main__":
    main()
