import openai
import json, re
from flask import Flask, request, render_template, jsonify
import pandas as pd

from io import StringIO
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# get OpenAI API key from json file
with open('openaicreds.json') as f:
    openaicreds = json.load(f)
    openai.api_key = openaicreds['api_key']

# Initialize the context variable to store the entire conversation history
context = []

def get_completion_from_messages(messages):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.5, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

def process_input(prompt):
    # Get system prompt
    systemMessage = open('system_prompt.txt','r').read()
    # If the context is empty, add the system prompt to the context
    if not context:
        context.append({"role": "system", "content": systemMessage})
    # Append user input to the context
    context.append({"role": "user", "content": prompt})
    output = get_completion_from_messages(context)
    # Append bot response to the context
    context.append({"role": "assistant", "content": output})
    return output

@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'data_file' in request.files:
        file = request.files['data_file']
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.txt'):
            df = pd.read_csv(file, sep='\t')
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl')
        df.to_csv('data.csv', index=False)
        return jsonify({'success': True})
    return jsonify({'error': 'Failed to upload the file'})

@app.route('/chat')
def chat_page():
    # Read the data from the "data.csv" file
    try:
        data = pd.read_csv('data.csv')
    except:
        return jsonify({'error': 'Failed to read the data'})
    
    # Display the first five rows of the data
    dataHead = data.head().to_html()

    # make an interpretation of the data set and give suggestions on how to analyse it
    data =  pd.DataFrame.to_string(data.head())
    prompt = f"I have uploaded the data set: \n ```{data}``` \n What do you believe is contained in the data? How can I analyse it."

    response = process_input(prompt)

    return render_template('index.html', dataHead=dataHead, response=response)

@app.route('/chat', methods=['POST'])
def chat():

    # Read the data from the "data.csv" file
    try:
        data = pd.read_csv('data.csv')
    except:
        return jsonify({'error': 'Failed to read the data'})

    # Display the first five rows of the data
    dataHead = data.head()
    dataHead = pd.DataFrame.to_string(dataHead)

    user_input = request.form['user_input']

    # Create prompt from user input
    with open('prompt.txt','r') as f:
        prompt = f.read()
        prompt = prompt.replace('dataHead', dataHead)
        prompt = prompt.replace('userMessage', user_input)

    response = process_input(prompt)

    # Check if response contains Python code
    code_pattern = r'```python(.*?)```'
    code_matches = re.findall(code_pattern, response, re.DOTALL)
    if code_matches:
        # Format code blocks with syntax highlighting
        response = re.sub(code_pattern, '<pre><code class="language-python">\\1</code></pre>', response, flags=re.DOTALL)
    
        # Redirect stdout to a StringIO object to capture output
        old_stdout = sys.stdout
        sys.stdout = output_buffer = StringIO()

        for match in code_matches:
            try:
                exec(match)
            except Exception as e:
                 response += f"<br><br><b>Error<b>: There was an error running the code. <br> {e}<br> Ask another question."
                 return jsonify({'user_input': user_input, 'bot_message': response})

        # Reset stdout and get captured output
        sys.stdout = old_stdout
        code_output = output_buffer.getvalue()

        context.append({"role": "user", "content": f"Running the code you have generated gives the ouput {code_output}.\
                         Interpret the output to answer {user_input}."})

        interp = get_completion_from_messages(context)

        # Append interp to response
        #response += '<br><br><b>Interpretation:</b><br>' + interp
        response += f"<br><br><b>Code Output:</b><br> {code_output} <br><br><b>Interpretation:</b><br> {interp}"

    return jsonify({'user_input': user_input, 'bot_message': response})


if __name__ == '__main__':
    app.run(debug=True)
