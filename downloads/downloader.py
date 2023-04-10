import os
import gdown
import subprocess
import glob
import shutil

DATASET_URLS = {
    "webui-all": "https://drive.google.com/drive/folders/1IGOCYjwY5wp3ZNEhxyN5bLEEJ-8M8kHg?usp=share_link",
    "webui-val": "https://drive.google.com/drive/folders/1ntEYc-VSvFOGmbgiRasALwDGzwCGnWjH?usp=share_link",
    "webui-test": "https://drive.google.com/drive/folders/1agq6S_-lyjXotPxDZVOvT78aoezGYEke?usp=share_link",
    "webui-7k": "https://drive.google.com/drive/folders/1AWj8yYMPiG--UPARdJoXT4j7J-N0VLfU?usp=share_link",
    "webui-7k-balanced": "https://drive.google.com/drive/folders/1F8W7OoMnpFGFHMK8m01r8zXb5765AB-N?usp=share_link",
    "webui-70k": "https://drive.google.com/drive/folders/1_srKdxB9Gjl02p-cEpBQ7OO2Q65I9whG?usp=share_link",
    "webui-350k": "https://drive.google.com/drive/folders/1yCEHzeWx33t6DsFt889SRnFoqtl-vTgu?usp=share_link"
}

MODEL_URLS = {

}

def download_dataset_gdown(dataset_key, tmp_path="tmp", dataset_path="ds"):
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    
    if not os.path.exists(os.path.join(tmp_path, dataset_key)):
        gdown.download_folder(DATASET_URLS[dataset_key], output=os.path.join(tmp_path, dataset_key))
    
    extract_file = glob.glob(os.path.join(tmp_path, dataset_key) + "/*.zip.001")[0]
    split_json_file = glob.glob(os.path.join(tmp_path, dataset_key) + "/*.json")[0]

    if not os.path.exists(os.path.basename(split_json_file)):
        shutil.move(split_json_file, ".")

    extract_path = os.path.join(tmp_path, "extract")
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    cmd = ['7z', 'x', extract_file, "-o" + str(extract_path)]
    sp = subprocess.Popen(cmd)
    sp.communicate()

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    dataset_ids = glob.glob(extract_path + "/*/*")

    for folder in dataset_ids:
        shutil.move(folder, os.path.join(dataset_path, os.path.basename(folder)))
    
    # delete the tmp path
    shutil.rmtree(tmp_path)
    

if __name__ == "__main__":
    download_dataset_gdown("webui-7k-balanced")