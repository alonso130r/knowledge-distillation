import openai
import jsonlines

# DOCS - here's the OpenAI docs for batch https://platform.openai.com/docs/guides/batch
def batch_file(batch_name : str, model: str, messages: list[list[dict]], client: openai.Client, max_tokens : int = None):
    '''
    Parameters
    ----------
    batch_name : str
        Should be a unique batch name, which will be used for the json file name and custom request id in the batch file. 
    model : str
        Must be an OpenAI model id. Raises ValueError if it is not found in the OpenAI models. 
    messages : list[dict]
        List of messages, according to the OpenAI specification. Raises KeyError if a message does not follow the correct format. Ex (one element of the list) : {'messages' : [{'role': 'system', 'content': 'You are a helpful assistant'}, {'role': 'user','content': 'Derive the quadratic formula.'}]}
    client : openai.Client
        Synchronus OpenAI client, must be activated with API_Key. 
    
    Returns
    -------
    filename : str
        The path
    
    '''
    models = [model.id for model in client.models.list().data]
    if model not in models:
        raise ValueError(f'The model inputted ({model}) was not found in the list of models. Must be one of the following: {', '.join(models)}')
    reqs = []
    for i, msg in enumerate(messages):
        custom_id = f'{batch_name}_{i}'
        method = 'POST'
        url = '/v1/chat/completions'
        body = {
            'model' : model,
            'messages' : msg['messages'],
        }
        if msg['messages'][0]['role'] != 'system':
            raise KeyError(f'Message {i} did not follow the correct format, as it lacked a system prompt. This was the message: {msg["messages"]}. Please correct it, see examples in the function documentation.')
        if msg['messages'][1]['role'] != 'user':
            raise KeyError(f'Message {i} did not follow the correct format, as it lacked a user prompt. This was the message: {msg["messages"]}. Please correct it, see examples in the function documentation.')
        if not (msg['messages'][0]['content'] or msg['messages'][1]['content']):
            raise KeyError(f'Message {i} did not follow the correct format, as the prompt was entirely empty. This was the message: {msg["messages"]}. Please correct it, see examples in the function documentation.')
        if max_tokens:
            body['max_tokens'] = max_tokens
        reqs.append({
            'custom_id' : custom_id,
            'method' : method,
            'url' : url,
            'body' : body,
            })
    
    with jsonlines.open(f'{batch_name}.jsonl', mode='w') as writer:
        writer.write_all(reqs)
    return f'{batch_name}.jsonl'
        
    
