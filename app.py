import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Set page configuration with dark theme
st.set_page_config(page_title="AI Chat Assistant", page_icon="💬", layout="centered")

# Apply dark theme styling
st.markdown("""
<style>
    .stApp {background-color: #1E1E1E; color: #FFFFFF;}
    .stTextInput, .stTextArea {background-color: #2D2D2D; color: #FFFFFF; border-radius: 5px;}
    .stButton>button {background-color: #4CAF50; color: white; border: none; border-radius: 5px; padding: 10px 24px; font-size: 16px;}
    .stButton>button:hover {background-color: #45a049;}
    .end-chat-btn>button {background-color: #f44336;}
    .end-chat-btn>button:hover {background-color: #d32f2f;}
    .chat-message {padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex; flex-direction: column;}
    .user-message {background-color: #2D2D2D; border-left: 5px solid #4CAF50;}
    .assistant-message {background-color: #383838; border-left: 5px solid #2196F3;}
    .summary-box {background-color: #2D2D2D; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #9C27B0; margin-top: 1rem;}
    .sentiment-box {background-color: #2D2D2D; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #FF9800; margin-top: 1rem;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_ended' not in st.session_state:
    st.session_state.chat_ended = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'sentiment' not in st.session_state:
    st.session_state.sentiment = ""
if 'sentiment_score' not in st.session_state:
    st.session_state.sentiment_score = 0
if 'api_validated' not in st.session_state:
    st.session_state.api_validated = False
if 'gemini_chain' not in st.session_state:
    st.session_state.gemini_chain = None
if 'openai_summary_chain' not in st.session_state:
    st.session_state.openai_summary_chain = None
if 'openai_sentiment_chain' not in st.session_state:
    st.session_state.openai_sentiment_chain = None
if 'gemini_sentiment_score_chain' not in st.session_state:
    st.session_state.gemini_sentiment_score_chain = None

# Title and description
st.title("AI Chat Assistant")
st.markdown("Chat with an AI assistant powered by Gemini and get summaries with OpenAI")

# Function to handle sending a message
def send_message():
    user_input = st.session_state.user_input
    if user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        try:
            # Get response from Gemini using LangChain
            response = st.session_state.gemini_chain.run(human_input=user_input)
            
            # If chain response is empty or fails, use a fallback
            if not response or response.strip() == "":
                response = f"I received your message. Let me think about how to respond to that."
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            # Add a fallback response if the AI fails
            fallback = f"I'm having trouble generating a response right now. Could you try rephrasing your message?"
            st.session_state.messages.append({"role": "assistant", "content": fallback})
        
        # Clear the input
        st.session_state.user_input = ""

# API Key validation
if not st.session_state.api_validated:
    with st.form("api_form"):
        st.subheader("API Configuration")
        gemini_api_key = st.text_input("Enter your Gemini API Key", type="password")
        openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
        
        submit_button = st.form_submit_button("Validate API Keys")
        
        if submit_button:
            if gemini_api_key and openai_api_key:
                try:
                    # Set environment variables
                    os.environ["GOOGLE_API_KEY"] = gemini_api_key
                    os.environ["OPENAI_API_KEY"] = openai_api_key
                    
                    # Initialize Gemini chat model
                    gemini_model = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash",
                        google_api_key=gemini_api_key,
                        temperature=0.7,
                        convert_system_message_to_human=True
                    )
                    
                    # Initialize OpenAI chat model
                    openai_model = ChatOpenAI(
                        model="gpt-3.5-turbo",
                        api_key=openai_api_key,
                        temperature=0.5
                    )
                    
                    # Create conversation memory
                    memory = ConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True
                    )
                    
                    # Create Langchain chains
                    st.session_state.gemini_chain = LLMChain(
                        llm=gemini_model,
                        prompt=PromptTemplate(
                            input_variables=["chat_history", "human_input"],
                            template="""
                            You are a helpful and friendly AI assistant. Respond conversationally but keep your responses brief (under 100 words).
                            
                            Chat History:
                            {chat_history}
                            
                            Human: {human_input}
                            AI Assistant:"""
                        ),
                        memory=memory,
                        verbose=True
                    )
                    
                    st.session_state.openai_summary_chain = LLMChain(
                        llm=openai_model,
                        prompt=PromptTemplate(
                            input_variables=["conversation"],
                            template="Summarize the following conversation in about 100 words using everyday language that anyone can understand:\n\n{conversation}\n\nSummary:"
                        ),
                        verbose=True
                    )
                    
                    st.session_state.openai_sentiment_chain = LLMChain(
                        llm=openai_model,
                        prompt=PromptTemplate(
                            input_variables=["conversation"],
                            template="Analyze the sentiment of the following conversation. Provide a brief 2-3 sentence analysis in simple language that captures the overall tone and emotions.\n\nConversation:\n{conversation}\n\nSentiment Analysis:"
                        ),
                        verbose=True
                    )
                    
                    st.session_state.gemini_sentiment_score_chain = LLMChain(
                        llm=gemini_model,
                        prompt=PromptTemplate(
                            input_variables=["sentiment_analysis"],
                            template="Based on this sentiment analysis: \"{sentiment_analysis}\", provide a single sentiment score from -1.0 (very negative) to 1.0 (very positive). Return ONLY the numeric score without explanation."
                        ),
                        verbose=True
                    )
                    
                    st.session_state.api_validated = True
                    st.success("API Keys validated successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error initializing models: {str(e)}")
            else:
                st.error("Both API keys are required")
else:
    # Display chat interface if chat hasn't ended
    if not st.session_state.chat_ended:
        # Display chat messages
        for message in st.session_state.messages:
            with st.container():
                st.markdown(f"""
                <div class="chat-message {'user-message' if message['role'] == 'user' else 'assistant-message'}">
                    <div><strong>{'You' if message['role'] == 'user' else 'Assistant'}</strong></div>
                    <div>{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input and send button
        with st.container():
            # Use a callback to handle sending messages
            st.text_input("Type your message:", 
                          key="user_input", 
                          on_change=send_message,
                          value="")
            
            # End chat button
            st.markdown("<div class='end-chat-btn'>", unsafe_allow_html=True)
            end_chat = st.button("End Chat")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Handle end chat button
        if end_chat and len(st.session_state.messages) > 0:
            st.session_state.chat_ended = True
            
            # Create conversation text
            conversation_text = ""
            for message in st.session_state.messages:
                prefix = "User: " if message["role"] == "user" else "Assistant: "
                conversation_text += prefix + message["content"] + "\n\n"
            
            try:
                with st.spinner("Generating summary..."):
                    st.session_state.summary = st.session_state.openai_summary_chain.run(conversation=conversation_text)
            except:
                st.session_state.summary = "A chat occurred between you and the AI assistant."
            
            try:
                with st.spinner("Analyzing sentiment..."):
                    st.session_state.sentiment = st.session_state.openai_sentiment_chain.run(conversation=conversation_text)
                    score_text = st.session_state.gemini_sentiment_score_chain.run(sentiment_analysis=st.session_state.sentiment)
                    
                    try:
                        st.session_state.sentiment_score = float(score_text.strip())
                        st.session_state.sentiment_score = max(-1.0, min(1.0, st.session_state.sentiment_score))
                    except:
                        st.session_state.sentiment_score = 0.0
            except:
                st.session_state.sentiment = "The conversation had a neutral tone overall."
                st.session_state.sentiment_score = 0.0
            
            st.rerun()
    else:
        # Display chat summary and sentiment analysis
        st.subheader("Chat Ended")
        
        # Display summary
        st.markdown("### Summary")
        st.markdown(f"""
        <div class="summary-box">
            {st.session_state.summary}
        </div>
        """, unsafe_allow_html=True)
        
        # Display sentiment analysis
        st.markdown("### Sentiment Analysis")
        st.markdown(f"""
        <div class="sentiment-box">
            {st.session_state.sentiment}
        </div>
        """, unsafe_allow_html=True)
        
        # Display sentiment meter
        st.markdown("### Sentiment Score")
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2:
            # Convert score from -1,1 range to 0,100 for the progress bar
            sentiment_percentage = int((st.session_state.sentiment_score + 1) * 50)
            
            # Create a more detailed sentiment scale
            if st.session_state.sentiment_score < -0.75:
                sentiment_label = "Very Negative"
                emoji = "😡"
            elif st.session_state.sentiment_score < -0.25:
                sentiment_label = "Negative"
                emoji = "😞"
            elif st.session_state.sentiment_score < 0.25:
                sentiment_label = "Neutral"
                emoji = "😐"
            elif st.session_state.sentiment_score < 0.75:
                sentiment_label = "Positive"
                emoji = "😊"
            else:
                sentiment_label = "Very Positive"
                emoji = "😄"
                
            st.markdown(f"<p style='text-align: center; font-size: 24px;'>{emoji} {sentiment_label}</p>", unsafe_allow_html=True)
            st.progress(sentiment_percentage, text=f"Score: {st.session_state.sentiment_score:.2f}")
            
            # Create a more detailed legend
            st.markdown("#### Sentiment Scale Legend:")
            
            # Create a table for the sentiment scale
            legend_data = [
                {"Range": "-1.0 to -0.75", "Category": "Very Negative", "Meaning": "Strong negative emotions, frustration, anger", "Emoji": "😡"},
                {"Range": "-0.75 to -0.25", "Category": "Negative", "Meaning": "Dissatisfaction, disappointment, concern", "Emoji": "😞"},
                {"Range": "-0.25 to 0.25", "Category": "Neutral", "Meaning": "Balanced, factual, neither positive nor negative", "Emoji": "😐"},
                {"Range": "0.25 to 0.75", "Category": "Positive", "Meaning": "Satisfaction, approval, contentment", "Emoji": "😊"},
                {"Range": "0.75 to 1.0", "Category": "Very Positive", "Meaning": "Enthusiasm, delight, strong appreciation", "Emoji": "😄"}
            ]
            
            # Display the legend as a table
            st.table(legend_data)
        
        # New chat button
        if st.button("Start New Chat"):
            st.session_state.chat_ended = False
            st.session_state.messages = []
            st.session_state.summary = ""
            st.session_state.sentiment = ""
            st.session_state.sentiment_score = 0.0
            st.rerun()