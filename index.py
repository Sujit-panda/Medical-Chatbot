from flask import Flask, request, jsonify, render_template, session
from langchain_core.runnables import RunnableWithMessageHistory, Runnable
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai
from flask_cors import CORS
from PIL import Image
import io
import base64
import os
from werkzeug.utils import secure_filename
import tempfile
from google.api_core import exceptions as google_exceptions

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "super_secret_key"
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True
CORS(app, supports_credentials=True)

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set Google Gemini and Pinecone API keys
genai.configure(api_key="AIzaSyC4k04O-KsUt9h5gNZ_GnbyCZlMV0LidDE")
# pinecone_api_key = 'your-pinecone-api-key'

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_2vPmPG_L5w7sUx9zEUCgomPFVvNu71gADGJKrSXESZ8XxXvpgXbTjehnTbPwn5HSBPNNHD")
index = pc.Index('medbot')

# Initialize SentenceTransformer model for encoding user input
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize conversation memory buffer
buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# Message templates
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

def get_system_message_template():
    """Dynamically create system message template based on session"""
    image_context = ""
    if 'image_analysis' in session:
        image_context = f"Consider this image analysis: {session['image_analysis']}\n"
    
    return SystemMessagePromptTemplate.from_template(
        template=f"""You are a medical assistant chatbot. {image_context}
        Answer questions truthfully using the provided context. 
        If the answer isn't in the context, say 'I DON'T KNOW'. Be empathetic and professional."""
    )

def get_prompt_template():
    """Create prompt template dynamically within request context"""
    return ChatPromptTemplate.from_messages([
        get_system_message_template(),
        MessagesPlaceholder(variable_name="history"),
        human_msg_template
    ])

# Custom Runnable for wrapping Google Gemini LLM
class GeminiRunnable(Runnable):
    def __init__(self, gemini_model):
        self.gemini_model = gemini_model

    def invoke(self, input_text):
        response = self.gemini_model.generate_content(input_text)
        return response.text.strip()

# Instantiate the Gemini models
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
vision_model = genai.GenerativeModel('gemini-1.5-pro')

# Wrap the Gemini model in the custom runnable
gemini_runnable = GeminiRunnable(gemini_model)

def create_conversation_chain():
    """Create conversation chain with dynamic prompt template"""
    return RunnableWithMessageHistory(
        runnable=gemini_runnable,
        get_session_history=lambda: buffer_memory.load_memory_variables({}),
        memory=buffer_memory,
        input_messages_key="input",
        history_messages_key="history",
        prompt=get_prompt_template(),
        verbose=True
    )

# Helper functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_image(image_data):
    """Analyze medical image and extract key information"""
    try:
        # Convert image data to PIL Image
        img = Image.open(io.BytesIO(image_data))
        
        # Prepare the prompt for medical image analysis
        prompt = [
            "As a medical imaging expert, analyze this image and describe:",
            "1. Visible anatomical structures",
            "2. Any abnormalities or notable features",
            "3. Potential clinical correlations",
            "4. Recommendations for further evaluation if needed",
            "Provide a structured report suitable for medical professionals.",
            img
        ]
        
        try:
            response = vision_model.generate_content(prompt)

            return response.text.strip()
        except google_exceptions.InvalidArgument as e:
            print(f"Gemini safety error: {e}")
            return "Image analysis blocked by safety filters"
        except Exception as e:
            print(f"Gemini API error: {e}")
            return "Error analyzing image - please try again"
            
    except Exception as e:
        print(f"Image processing error: {e}")
        return None

def find_match(input_text):
    """Find matching context from Pinecone vector database"""
    try:
        if 'image_analysis' in session:
            input_text += "\nImage Context: " + session['image_analysis']
        
        input_em = model.encode(input_text).tolist()
        result = index.query(vector=input_em, top_k=2, includeMetadata=True)
        
        if 'matches' in result and len(result['matches']) >= 2:
            return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']
        else:
            return "No sufficient matches found in the index."
    except Exception as e:
        print(f"Error in find_match: {e}")
        return "An error occurred while querying the index."

def query_refiner(conversation, query):
    """Refine user query based on conversation context"""
    response = gemini_model.generate_content(
        f"Refine the following user query based on the conversation:\n\nUser Query: {query}\nConversation: {conversation}"
    )
    return response.text.strip()

def get_conversation_string(responses, requests):
    """Convert conversation history to string"""
    conversation_string = ""
    for i in range(len(responses) - 1):
        conversation_string += "Human: " + requests[i] + "\n"
        conversation_string += "Bot: " + responses[i + 1] + "\n"
    return conversation_string

def generate_followup_questions(user_input):
    """Generate relevant follow-up questions"""
    pinecone_results = find_match(user_input)
    
    prompt = f"""Based on the following user input about their health concern and related medical information, 
    generate 3-4 relevant follow-up questions to better understand their condition. The questions should be 
    simple, clear, and medically relevant.

    User Input: {user_input}
    Related Medical Information: {pinecone_results}

    Generate the questions in a numbered list, each question on a new line:"""
    
    response = gemini_model.generate_content(prompt)
    questions = response.text.strip().split('\n')
    return [q.strip() for q in questions if q.strip()]

def generate_symptom_suggestions(user_input):
    """Suggest potentially related symptoms"""
    pinecone_results = find_match(user_input)
    
    prompt = f"""Based on the following user input about their symptoms and related medical information, 
    suggest 2-3 potentially related symptoms they might be experiencing. Present each suggestion with 
    a brief explanation of why it might be relevant.

    User Input: {user_input}
    Related Medical Information: {pinecone_results}

    Format each suggestion as:
    - [Symptom]: [Brief explanation]"""
    
    response = gemini_model.generate_content(prompt)
    suggestions = response.text.strip().split('\n')
    return [s.strip() for s in suggestions if s.strip()]

def generate_potential_diagnoses(user_input, answers):
    """Generate potential diagnoses based on user input and answers"""
    context = f"Initial concern: {user_input}\n"
    
    if 'image_analysis' in session:
        context += f"Image Analysis:\n{session['image_analysis']}\n"
    
    for i, question in enumerate(session.get('followup_questions', [])):
        if i < len(answers):
            context += f"Question {i+1}: {question}\nAnswer {i+1}: {answers[i]}\n"
    
    pinecone_results = find_match(context)
    
    prompt = f"""Based on the following patient information and related medical knowledge, provide:
    1. 2-3 potential diagnoses (list as "Possible Conditions")
    2. Recommended next steps (list as "Recommended Actions")
    3. Any warning signs to watch for (list as "Warning Signs")
    
    Patient Information:
    {context}
    
    Related Medical Knowledge:
    {pinecone_results}
    
    Structure your response with clear headings for each section."""
    
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# Routes
@app.route('/')
def hello():
    """Clear session and render main page"""
    session.clear()
    if 'image_analysis' in session:
        del session['image_analysis']
    print("Session cleared")
    return render_template('new.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image upload separately"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
            
        file = request.files['image']
        if file and allowed_file(file.filename):
            # Save to temp file
            image_data = file.read()
            
            # Analyze image
            analysis_result = analyze_image(image_data)
            if analysis_result:
                session['image_analysis'] = analysis_result
                return jsonify({
                    "status": "success",
                    "message": "Image uploaded and analyzed. Now ask me about it!",
                    "analysis": analysis_result
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Failed to analyze image"
                }), 400
                
        return jsonify({"error": "Invalid file"}), 400
        
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({"error": "Image processing failed"}), 500

@app.route('/send_message', methods=['POST'])
def send_message():
    """Handle text messages with optional image context"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        user_message = data.get('message')
        print("User Message:", user_message)
        # print({session['image_analysis']})

        # Initialize context with image analysis if available
        context_parts = []
        if 'image_analysis' in session:
            context_parts.append(f"IMAGE ANALYSIS:\n{session['image_analysis']}")
            # Don't clear here - keep for follow-up questions
        
        context_parts.append(f"USER QUERY:\n{user_message}")
        full_context = "\n\n".join(context_parts)

        # Step 1: Store the first input and generate follow-up questions
        if 'first_input' not in session:
            session['first_input'] = user_message
            session['followup_questions'] = generate_followup_questions(full_context)
            session['symptom_suggestions'] = generate_symptom_suggestions(full_context)
            session['current_question_index'] = 0
            session['answers'] = []
            
            print("First input stored:", session['first_input'])
            print("Generated follow-up questions:", session['followup_questions'])
            print("Generated symptom suggestions:", session['symptom_suggestions'])
            
            # Prepare the initial response with symptom suggestions and first follow-up question
            response = (
                "Thank you for sharing your concern. Based on what you've told me, here are some symptoms "
                "that might be related:\n\n" +
                "\n".join(session['symptom_suggestions']) +
                "\n\nTo better understand your condition, I'd like to ask: " +
                session['followup_questions'][0]
            )
            
            return jsonify({
                "response": response,
                "related_questions": []
            })

        # Step 2: Handle the follow-up questions
        if 'followup_questions' in session and session['current_question_index'] < len(session['followup_questions']):
            # Store the user's answer to the current question
            session['answers'].append(user_message)
            session['current_question_index'] += 1

            # Check if more questions remain to be asked
            if session['current_question_index'] < len(session['followup_questions']):
                next_question = session['followup_questions'][session['current_question_index']]
                return jsonify({
                    "response": next_question,
                    "related_questions": []
                })
            else:
                # All follow-up questions have been answered - generate potential diagnoses
                diagnosis_response = generate_potential_diagnoses(session['first_input'], session['answers'])
                session['diagnosis_provided'] = True
                
                # Clear image analysis after final diagnosis
                if 'image_analysis' in session:
                    del session['image_analysis']
                
                return jsonify({
                    "response": diagnosis_response,
                    "related_questions": []
                })

        # Step 3: Post-diagnosis conversation
        if 'requests' not in session:
            session['requests'] = []
        if 'responses' not in session:
            session['responses'] = []

        # Store user message and continue conversation
        session['requests'].append(user_message)
        conversation_string = get_conversation_string(session['responses'], session['requests'])
        print("Conversation String:", conversation_string)

        # Combine all context for the response
        context = f"Initial Concern: {session['first_input']}\n"
        if 'image_analysis' in session:
            context += f"Image Analysis:\n{session['image_analysis']}\n"
            
        for i, question in enumerate(session['followup_questions']):
            if i < len(session['answers']):
                context += f"Question {i+1}: {question}\nAnswer {i+1}: {session['answers'][i]}\n"
        
        if 'diagnosis_provided' in session:
            context += f"Previously provided diagnosis information\n"

        refined_query = query_refiner(context, user_message)
        print("Refined Query:", refined_query)

        pinecone_context = find_match(refined_query)
        print("Pinecone Context:", pinecone_context)

        prompt = f"Context:\n{pinecone_context}\nPatient Information:\n{context}\nCurrent Query:\n{user_message}"
        
        response = gemini_model.generate_content(prompt).text
        print("Response:", response)

        alternatives = generate_followup_questions(f"{context}\n{user_message}")
        session['responses'].append(response)

        print("Session Data:", session)

        return jsonify({
            "response": response,
            "refined_query": refined_query,
            # "related_questions": alternatives
        })

    except Exception as e:
        print(f"Error in send_message: {e}")
        return jsonify({
            "response": "Sorry, there was an error processing your request. Please try again.",
            "related_questions": []
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
