import os
import mimetypes
import base64
from openai import OpenAI

def image_to_data_url(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with open(image_path, "rb") as f:
        image_data = f.read()
    mime_type,_ = mimetypes.guess_type(image_path)

    if mime_type is None or not mime_type.startswith('image/'):
        mime_type = 'image/png'

    base64_encoded = base64.b64encode(image_data).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded}"


def main():
    client = OpenAI(
        base_url='https://api-inference.modelscope.cn/v1',
        api_key='ms-13172eb6-370d-4b42-996d-c8d2b830b661',
    )

    image_description = """
    The red bounding box indicates the target.\n
    These images are sampled from the same target trajectory in chronological order.\n
    """
    description_demands = """
    Please describe:\n
    1. Appearance of the target.\n
    2. Movement path.\n
    3. Behavior changes over time.\n
    4. Spatial landmarks used to identify the target.\n
    """

    output_demands = """
    Output:\n
    Chinese description\n
    English description\n
    JSON\n
    """

    prompt = f"""
    {image_description}
    {description_demands}
    {output_demands}
    """

    response = client.chat.completions.create(
        model='Qwen/Qwen3.5-35B-A3B', # ModelScope Model-Id, required
        messages=[{
            'role':'user',
            'content':[{
                'type':'text',
                'text':f"{prompt}",
            },{
                 'type':'image_url',
                 'image_url':{
                     'url':image_to_data_url('./data/DynUAV/001/48/images/frame_001171.jpg')
                 },
            },{
                 'type':'image_url',
                 'image_url':{
                     'url':image_to_data_url('./data/DynUAV/001/48/images/frame_001233.jpg')
                 }, 
            },{
                 'type':'image_url',
                 'image_url':{
                     'url':image_to_data_url('./data/DynUAV/001/48/images/frame_001292.jpg')
                 }
            },{
                 'type':'image_url',
                 'image_url':{
                     'url':image_to_data_url('./data/DynUAV/001/48/images/frame_001352.jpg')
                 }, 
            },{
                 'type':'image_url',
                 'image_url':{
                     'url':image_to_data_url('./data/DynUAV/001/48/images/frame_001565.jpg')
                 }, 
            },{
                 'type':'image_url',
                 'image_url':{
                     'url':image_to_data_url('./data/DynUAV/001/48/images/frame_001641.jpg')
                 }, 
            },{
                 'type':'image_url',
                 'image_url':{
                     'url':image_to_data_url('./data/DynUAV/001/48/images/frame_001663.jpg')
                 }, 
            },{
                 'type':'image_url',
                 'image_url':{
                     'url':image_to_data_url('./data/DynUAV/001/48/images/frame_001731.jpg')
                 }, 
            },{
                 'type':'image_url',
                 'image_url':{
                     'url':image_to_data_url('./data/DynUAV/001/48/images/frame_001780.jpg')
                 }, 
            },{
                 'type':'image_url',
                 'image_url':{
                     'url':image_to_data_url('./data/DynUAV/001/48/images/frame_001796.jpg')
                 }, 
            },{
                 'type':'image_url',
                 'image_url':{
                     'url':image_to_data_url('./data/DynUAV/001/48/images/frame_001815.jpg')
                 }, 
            },{
                 'type':'image_url',
                 'image_url':{
                     'url':image_to_data_url('./data/DynUAV/001/48/images/frame_001870.jpg')
                 }, 
            },],
        }],
        stream=False
    )
    
    description = response.choices[0].message.content
    print(description)

if __name__=='__main__':
    main()
