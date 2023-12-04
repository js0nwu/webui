from tqdm import tqdm
from argparse import ArgumentParser
import json
import os
import gzip

parser = ArgumentParser()
#parser.add_argument("--split_file", type=str, default="./train_split_webui.json")
parser.add_argument("--root_path", type=str, default="/datasets/webui_2022-06-30/crawls")
parser.add_argument("--out_path", type=str, default="notebooks/all_data")

args = parser.parse_args()

id_list = os.listdir(args.root_path)

if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)
    
for web_id in tqdm(id_list):
    text_files = [f for f in os.listdir(os.path.join(args.root_path,web_id)) if f.endswith('.txt')]
    if not os.path.exists(os.path.join(args.out_path,web_id)):
        os.makedirs(os.path.join(args.out_path,web_id))
    for key_name in text_files:
        try:
            key = os.path.join(args.root_path, web_id, key_name)

            key_name = key_name.replace("-url.txt","")

            out_file = os.path.join(args.out_path,web_id,key_name + ".json")
            if os.path.exists(out_file):
                continue
            target = {}
            box_path = key.replace("url.txt", "box.json.gz")

            if not os.path.exists(box_path):
                continue
            with gzip.open(box_path,'rb') as box_f:
                box_f = box_f.read()
                decoded_box = box_f.decode("utf-8", errors='ignore')
                box_json=json.loads(decoded_box)

            axtree_path = key.replace("url.txt", "axtree.json.gz")
            if not os.path.exists(axtree_path):
                continue
            with gzip.open(axtree_path,'rb') as axtree_f:
                axtree_j=axtree_f.read()
                data_str = axtree_j.decode("utf-8", errors='ignore')
                axtree_json = json.loads(data_str)

            axtree_map = {}
            for i in range(len(axtree_json['nodes'])):
                node = axtree_json['nodes'][i]
                if "nodeId" in node:
                    nodeId = node["nodeId"]
                    axtree_map[nodeId] = node

            labels = []
            contentBoxes = []
            paddingBoxes = []
            borderBoxes = []
            marginBoxes = []

            for i in range(len(axtree_json['nodes'])):
                node = axtree_json['nodes'][i]
                if 'backendDOMNodeId' not in node:
                    continue
                nodeId = str(node['backendDOMNodeId'])
                # make sure is not ignored
                if node['ignored']:
                    continue

                # make sure it is on the screen
                if nodeId not in box_json:
                    continue
                if box_json[nodeId] is None:
                    continue

                # only process leaf nodes
                if len(node['childIds']) > 0:
                    continue

                elementLabels = [node['role']['value']]
                # find the top-level singleton element for the leaf node
                while "parentId" in node and node["parentId"] in axtree_map and "childIds" in axtree_map[node["parentId"]] and len(axtree_map[node["parentId"]]["childIds"]) == 1 and 'backendDOMNodeId' in axtree_map[node["parentId"]] and not axtree_map[node["parentId"]]["ignored"] and axtree_map[node["parentId"]]["role"]["value"] != "none" and axtree_map[node["parentId"]]["role"]["value"] != "generic":
                    node = axtree_map[node["parentId"]]
                    elementLabels.append(node['role']['value'])

                nodeId = str(node['backendDOMNodeId'])

                # make sure it is on the screen
                if nodeId not in box_json:
                    continue
                if box_json[nodeId] is None:
                    continue

                elementBox = box_json[nodeId]

                # content box
                x1 = elementBox['content'][0]['x']
                y1 = elementBox['content'][0]['y']
                # for some reason, the other extreme corner is the 3rd item in the content list
                x2 = elementBox['content'][2]['x']
                y2 = elementBox['content'][2]['y']
                contentBox = [x1, y1, x2, y2][:]

                # padding box
                x1 = elementBox['padding'][0]['x']
                y1 = elementBox['padding'][0]['y']
                x2 = elementBox['padding'][2]['x']
                y2 = elementBox['padding'][2]['y']
                paddingBox = [x1, y1, x2, y2][:]

                # border box
                x1 = elementBox['border'][0]['x']
                y1 = elementBox['border'][0]['y']
                x2 = elementBox['border'][2]['x']
                y2 = elementBox['border'][2]['y']
                borderBox = [x1, y1, x2, y2][:]

                # margin box
                x1 = elementBox['margin'][0]['x']
                y1 = elementBox['margin'][0]['y']
                x2 = elementBox['margin'][2]['x']
                y2 = elementBox['margin'][2]['y']
                marginBox = [x1, y1, x2, y2][:]

                labels.append(elementLabels)
                contentBoxes.append(contentBox)
                paddingBoxes.append(paddingBox)
                borderBoxes.append(borderBox)
                marginBoxes.append(marginBox)

            target["labels"] = labels
            target["contentBoxes"] = contentBoxes
            target["paddingBoxes"] = paddingBoxes
            target["borderBoxes"] = borderBoxes
            target["marginBoxes"] = marginBoxes
            target["key_name"] = key_name

            with open(out_file, "w") as f:
                json.dump(target, f)

        except Exception as e:
            print("failed", key_name, str(e))
