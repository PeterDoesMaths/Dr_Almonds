# DR Almonds (A Language Model Nailing Data Science)

Dr Almonds (a nutty data science professor) will be a Python app chat bot that uses AI to perform useful analysis on a user provided data set. Harnessing OpenAI's large language model GPT-3.5, the app will hopefully be really simple to use, even for a non-specialist! Below is a basic outline of the workflow:

1. The user uploads their data (csv, txt or xlsx file).
2. The header of the data file is extracted and the user is asked what they want from the data.
3. The user asks a question and interacts with Dr Almonds through a chat bot feature.
4. The user's messages are inserted into a prompt is then inputted into an API call to OpenAI which will
    * Produce python code to achieve the task, possibly providing an explanation of the code,
    * Run the code,
    * Provide the user with the output (this could be a figure, table, number etc.), and an interpretation.
5. At this point the user could leave a happy chappy, or if the task is not fulfilled to their liking, continue talking with Dr Almonds through the chat bot.
