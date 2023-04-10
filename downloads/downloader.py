import os
import gdown
import subprocess
import glob
import shutil

DATASET_GDRIVE_URLS = {
    "webui-all": "https://drive.google.com/drive/folders/1IGOCYjwY5wp3ZNEhxyN5bLEEJ-8M8kHg?usp=share_link",
    "webui-val": "https://drive.google.com/drive/folders/1ntEYc-VSvFOGmbgiRasALwDGzwCGnWjH?usp=share_link",
    "webui-test": "https://drive.google.com/drive/folders/1agq6S_-lyjXotPxDZVOvT78aoezGYEke?usp=share_link",
    "webui-7k": "https://drive.google.com/drive/folders/1AWj8yYMPiG--UPARdJoXT4j7J-N0VLfU?usp=share_link",
    "webui-7k-balanced": "https://drive.google.com/drive/folders/1F8W7OoMnpFGFHMK8m01r8zXb5765AB-N?usp=share_link",
    "webui-70k": "https://drive.google.com/drive/folders/1_srKdxB9Gjl02p-cEpBQ7OO2Q65I9whG?usp=share_link",
    "webui-350k": "https://drive.google.com/drive/folders/1yCEHzeWx33t6DsFt889SRnFoqtl-vTgu?usp=share_link"
}

MODEL_GDRIVE_URLS = {
    "screenclassification": {
        "screenclassification-resnet-baseline.ckpt": "https://drive.google.com/file/d/1uBZMa5Z1lXiGGf5i4JHdhIck5gJjw22K/view?usp=share_link",
        "screenclassification-resnet-noisystudent+rico.ckpt": "https://drive.google.com/file/d/1olTBXNN4bzj32LYz06Rd2gPj4LTv079H/view?usp=share_link",
        "screenclassification-resnet-noisystudent+web7k.ckpt": "https://drive.google.com/file/d/1c9Ow1sm8tQwGmLlXX-_4rWn6opklIuR6/view?usp=share_link",
        "screenclassification-resnet-noisystudent+web7kbal.ckpt": "https://drive.google.com/file/d/1bKuieDZplPBWSLJc7NAkZZG3pP7DouQp/view?usp=share_link",
        "screenclassification-resnet-noisystudent+web70k.ckpt": "https://drive.google.com/file/d/1Mfksb3Rnp2GhCnTHLveXw4v43vpDivFd/view?usp=share_link",
        "screenclassification-resnet-noisystudent+web350k.ckpt": "https://drive.google.com/file/d/1jAVpeXV46veDq2L4RJMX8uLTim23Egl_/view?usp=share_link",
        "screenclassification-resnet-randaugment.ckpt": "https://drive.google.com/file/d/1vFm9e20GORqM5Bhxn-6F9LCZCzxEG5n5/view?usp=share_link",
        "screenclassification-vgg16-baseline.ckpt": "https://drive.google.com/file/d/13hZRlAGW9OErdjMQnBjsMklv2vyZR535/view?usp=share_link",    
    },
    "screenrecognition": {
        "screenrecognition-ssd-vins.ckpt": "https://drive.google.com/file/d/1bu3wL2PH6AHgg5-7YkEidsBLaPbs22kc/view?usp=share_link",
        "screenrecognition-vins.ckpt": "https://drive.google.com/file/d/10Id643ldFjOeOtnY2cGCtKOEDGMp_BHV/view?usp=share_link",
        "screenrecognition-web7k-vins.ckpt": "https://drive.google.com/file/d/1M3uoxLKncwf0WHLbEhbCoOytDjQT2gU9/view?usp=share_link",
        "screenrecognition-web7k.ckpt": "https://drive.google.com/file/d/1DfIz1geicHYNq3_UdT10oSjLCDgkE72t/view?usp=share_link",
        "screenrecognition-web7kbal-vins.ckpt": "https://drive.google.com/file/d/10Gb77oBa7HmQwcR2vLVTdNdoUPagEPyy/view?usp=share_link",
        "screenrecognition-web7kbal.ckpt": "https://drive.google.com/file/d/1-0TrGpDaQMrDK2Wf8A-7pnrgHnJXmdiz/view?usp=share_link",
        "screenrecognition-web70k-vins.ckpt": "https://drive.google.com/file/d/1BsOa3e9T3_HM5rGPY70K9Z4FuBIrKqVs/view?usp=share_link",
        "screenrecognition-web70k.ckpt": "https://drive.google.com/file/d/1yeCFHIfLl7taSAoYYCuECwaCZmLToKlI/view?usp=share_link",
        "screenrecognition-web350k-vins.ckpt": "https://drive.google.com/file/d/14BjYnwyWhHK8APpWLHj9J7SgoHBLjrMb/view?usp=share_link",
        "screenrecognition-web350k.ckpt": "https://drive.google.com/file/d/1SjU-yjhBXdImCmSf251EWceAH-QAef_N/view?usp=share_link",
    },
    "screensim": {
        "screensim-resnet-uda+web7k.ckpt": "https://drive.google.com/file/d/16fRllQ80tYuiFoSlrpnDOAsmD5W-sWPb/view?usp=share_link",
        "screensim-resnet-web7k.ckpt": "https://drive.google.com/file/d/1uxpcGHvceYYTxxj98bBme2QXkmgXGiQg/view?usp=share_link",
        "screensim-resnet-uda+web7kbal.ckpt": "https://drive.google.com/file/d/1CareIltu1GgKINm9XNjfUwKA3YBniklY/view?usp=share_link",
        "screensim-resnet-web7kbal.ckpt": "https://drive.google.com/file/d/133tv6-nFdm78ngn4DVMbJ0W8QEpS__aB/view?usp=share_link",
        "screensim-resnet-uda+web70k.ckpt": "https://drive.google.com/file/d/1taNDFSIUP1ThsWpkeb0Vd4V8I4exf0qM/view?usp=share_link",
        "screensim-resnet-web70k.ckpt": "https://drive.google.com/file/d/1oj32qKLVOZdwFtqosht2tWaaVA84uXA-/view?usp=share_link",
        "screensim-resnet-uda+web350k.ckpt": "https://drive.google.com/file/d/1WCofe3JUDT_AJNVLXjVxWsBurLe0wcjQ/view?usp=share_link",
        "screensim-resnet-web350k.ckpt": "https://drive.google.com/file/d/1vP7-YHkcz9BqfmKhpd_F_LlgzYENTGJG/view?usp=share_link",
    }
}

def download_dataset_gdown(dataset_key, tmp_path="tmp", dataset_path="ds"):
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    
    if not os.path.exists(os.path.join(tmp_path, dataset_key)):
        gdown.download_folder(DATASET_GDRIVE_URLS[dataset_key], output=os.path.join(tmp_path, dataset_key))
    
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
    
def download_model_gdown(model_name, model_key, model_path="checkpoints"):
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    gdown.download(MODEL_GDRIVE_URLS[model_name][model_key], output=os.path.join(model_path, model_key), fuzzy=True)

if __name__ == "__main__":
    # download_dataset_gdown("webui-7k-balanced")
    download_model_gdown("screenrecognition", "screenrecognition-web350k-vins.ckpt")