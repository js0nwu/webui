if __name__ == "__main__":

    ARTIFACT_DIR = "./checkpoints_screenrecognition_web70k"

    CHECK_INTERVAL_STEPS = 4000

    import os

    if not os.path.exists(ARTIFACT_DIR):
        os.makedirs(ARTIFACT_DIR)

    from ui_datasets import *
    from ui_models import *
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import *
    import torch
    import datetime
    from pytorch_lightning.loggers import TensorBoardLogger
    logger = TensorBoardLogger(ARTIFACT_DIR)

    data = WebUIDataModule(train_split_file = '../../downloads/train_split_web70k.json')
    
    model = UIElementDetector(num_classes=32)
    
    print("***********************************")
    print("checkpoints: " + str(os.listdir(ARTIFACT_DIR)))
    print("***********************************")
    
    
    checkpoint_callback = ModelCheckpoint(dirpath=ARTIFACT_DIR, filename= "screenrecognition",every_n_train_steps=CHECK_INTERVAL_STEPS, save_last=True)
    checkpoint_callback2 = ModelCheckpoint(dirpath=ARTIFACT_DIR, filename= "screenrecognition",monitor='mAP', mode="max", save_top_k=1)
    
    earlystopping_callback = EarlyStopping(monitor="mAP", mode="max", patience=10)

    trainer = Trainer(
        gpus=1,
        precision=16,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        callbacks=[checkpoint_callback, checkpoint_callback2, earlystopping_callback],
        val_check_interval=CHECK_INTERVAL_STEPS,
        min_epochs=10,
        logger=logger,
        limit_val_batches=0.5
    )
    
    if os.path.exists(os.path.join(ARTIFACT_DIR, "last.ckpt")):
        model = UIElementDetector.load_from_checkpoint(os.path.join(ARTIFACT_DIR, "last.ckpt"))

    trainer.fit(model, data)
