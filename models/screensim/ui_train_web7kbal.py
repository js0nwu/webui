if __name__ == "__main__":

    ARTIFACT_DIR = "checkpoints_screensim_resnet18_web7kbal"
    CHECK_INTERVAL_STEPS = 100

    import os

    if not os.path.exists(ARTIFACT_DIR):
        os.makedirs(ARTIFACT_DIR)

    from ui_datasets import *
    from ui_models import *
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import *

    from torch import nn
    import torch
    import datetime
    from pytorch_lightning.loggers import TensorBoardLogger
    import os
    
    logger = TensorBoardLogger(ARTIFACT_DIR)
    
    data = WebUISimilarityDataModule(split_file="../../downloads/balanced_7k.json")

    model = UIScreenEmbedder()

    print("***********************************")
    print("checkpoints: " + str(os.listdir(ARTIFACT_DIR)))
    print("***********************************")
    
    checkpoint_callback = ModelCheckpoint(dirpath=ARTIFACT_DIR, every_n_train_steps=CHECK_INTERVAL_STEPS, save_last=True)
    checkpoint_callback2 = ModelCheckpoint(dirpath=ARTIFACT_DIR, filename= "screensim", save_top_k=1, every_n_train_steps=CHECK_INTERVAL_STEPS, mode="max", monitor="f1")
    earlystopping_callback = EarlyStopping(monitor="f1", mode="max", patience=20)
    trainer = Trainer(
        gpus=1,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, checkpoint_callback2, earlystopping_callback],
        logger=logger,
        val_check_interval=CHECK_INTERVAL_STEPS,
        accumulate_grad_batches=4,
        limit_val_batches=50
    )
    
    if os.path.exists(os.path.join(ARTIFACT_DIR, "last.ckpt")):
        model = UIScreenClassifier.load_from_checkpoint(os.path.join(ARTIFACT_DIR, "last.ckpt"))

    trainer.fit(model, data)
