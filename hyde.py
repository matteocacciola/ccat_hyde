from langchain_classic.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

from cat import log, hook
from cat.services.memory.models import RecallSettings

# Keys
HYDE_ANSWER       = "hyde_answer"
AVERAGE_EMBEDDING = "average_embedding"


@hook(priority=1)
def before_cat_reads_message(user_message, cat):
    # Acquire settings
    settings = cat.mad_hatter.get_plugin().load_settings()
    log.debug(f" --------- ACQUIRE SETTINGS ---------")
    log.debug(f"settings: {settings}")

    # Make a prompt from template
    hypothesis_prompt = PromptTemplate(
        input_variables=["input"],
        template=settings["hyde_prompt"]
    )

    # Run a LLM chain with the user message as input
    hypothesis_chain = LLMChain(prompt=hypothesis_prompt, llm=cat.large_language_model)
    answer = hypothesis_chain(user_message)
    
    # Save HyDE answer in working memory
    cat.working_memory[HYDE_ANSWER] = answer["text"]
    
    log.debug("------------- HYDE -------------")
    log.debug(f"user message: {user_message}")
    log.debug(f"hyde answer: {answer['text']}")
    
    return user_message


# Calculates the average between the user's message embedding and the Hyde response embedding
def _calculate_vector_average(config: RecallSettings, cat):
    # If hyde answer exists, calculate and set average embedding
    if HYDE_ANSWER in cat.working_memory.keys():
        
       # Get user message embedding
        user_embedding = config.embedding
        
        # Calculate hyde embedding from hyde answer
        hyde_answer = cat.working_memory[HYDE_ANSWER]
        hyde_embedding = cat.embedder.embed_query(hyde_answer)

        # Calculate average embedding and stores it into a working memory
        average_embedding = [(x + y)/2 for x, y in zip(user_embedding, hyde_embedding)]
        cat.working_memory[AVERAGE_EMBEDDING] = average_embedding

        log.debug(f" --------- CALCULATE AVERAGE ---------")
        log.debug(f"hyde answer:       {hyde_answer}")
        log.debug(f"user_embedding:    {user_embedding}")
        log.debug(f"hyde_embedding:    {hyde_embedding}")
        log.debug(f"average_embedding: {average_embedding}")

        # Delete Hyde Answer from working memory
        del cat.working_memory[HYDE_ANSWER]

    # If average embedding exists, set the embedding
    if AVERAGE_EMBEDDING in cat.working_memory.keys():
        average_embedding = cat.working_memory[AVERAGE_EMBEDDING]
        config.embedding = average_embedding
        
        log.debug(f" --------- SET EMBEDDING ---------")
        log.debug(f"average_embedding: {average_embedding}")
        

@hook(priority=1)
def before_cat_recalls_memories(config: RecallSettings, cat):
    _calculate_vector_average(config, cat)
