from glob import glob
import os

product_files = glob(os.path.join(".", "products", "*"))      
product_dict = {product_path.split("/")[-1].split(".")[-2].strip("products\\"): product_path
    for product_path in product_files}

def knowledge(transcript_text):
    knowledge = []

    if "脫髮" in transcript_text or "白髮" in transcript_text or "掉頭髮" in transcript_text or "洗髮" in transcript_text:
        file_names= [file_name for file_name in product_files if "烏髮濃專用健髮洗髮露" in file_name]
        print(file_names)

        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as file:
                content = file.read()
                new_knowledge = "推薦產品: " + content
                knowledge.append(new_knowledge)
                                

    elif "失眠" in transcript_text or "瞓唔着" in transcript_text:
        file_names= [file_name for file_name in product_files if "失眠" in file_name]
        print(file_names)

        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as file:
                content = file.read()
                new_knowledge = "推薦產品: " + content
                knowledge.append(new_knowledge)
                

    elif "感冒"in transcript_text:
        file_names= [file_name for file_name in product_files if "感冒" in file_name]
        
        print(file_names)

        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as file:
                content = file.read()
                new_knowledge = "推薦產品: " + content
                knowledge.append(new_knowledge)

    elif "咳" in transcript_text or "嗽" in transcript_text:
        file_names= [file_name for file_name in product_files if "咳" in file_name or "嗽"in file_name]
        
        print(file_names)

        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as file:
                content = file.read()
                new_knowledge = "推薦產品: " + content
                knowledge.append(new_knowledge)
                        

    elif "心"in transcript_text:
        file_names= [file_name for file_name in product_files if "心" in file_name]
        print(file_names)

        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as file:
                content = file.read()
                new_knowledge = "推薦產品: " + content
                knowledge.append(new_knowledge)
                                

    elif "病後"in transcript_text or"調理" in transcript_text:

        file_names= [file_name for file_name in product_files if "雲芝"in file_name]
        print(file_names)

        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as file:
                content = file.read()
                new_knowledge = "推薦產品: " + content
                knowledge.append(new_knowledge)

    elif "免疫力"in transcript_text or"抵抗力" in transcript_text or "身體虛弱" in transcript_text:

        file_names= [file_name for file_name in product_files if "靈芝" in file_name and "雲芝"in file_name]
        print(file_names)

        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as file:
                content = file.read()
                new_knowledge = "推薦產品: " + content
                knowledge.append(new_knowledge)
    
    elif "全面保健"in transcript_text or"延緩老化" in transcript_text:

        file_names= [file_name for file_name in product_files if "金靈芝" in file_name]
        print(file_names)

        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as file:
                content = file.read()
                new_knowledge = "推薦產品: " + content
                knowledge.append(new_knowledge)
                        
    elif "改善呼吸"in transcript_text or"補肺益腎" in transcript_text:

        file_names= [file_name for file_name in product_files if "蟲草菌絲" in file_name]
        print(file_names)

        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as file:
                content = file.read()
                new_knowledge = "推薦產品: " + content
                knowledge.append(new_knowledge)                    

    elif "鼻敏感"in transcript_text:

        file_names= [file_name for file_name in product_files if "鼻敏感" in file_name]
        print(file_names)

        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as file:
                content = file.read()
                new_knowledge = "推薦產品: " + content
                knowledge.append(new_knowledge)
                      

    elif "肝" in transcript_text:

        file_names= [file_name for file_name in product_files if "肝" in file_name]
        print(file_names)

        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as file:
                content = file.read()
                new_knowledge = "推薦產品: " + content
                knowledge.append(new_knowledge)

    elif "濃縮中藥" in transcript_text:
        file_names= [file_name for file_name in product_files if "濃縮中藥" in file_name]
        print(file_names)

        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as file:
                content = file.read()
                new_knowledge = "推薦產品: " + content
                knowledge.append(new_knowledge)

    knowledge = ', '.join(knowledge)         
    return knowledge
