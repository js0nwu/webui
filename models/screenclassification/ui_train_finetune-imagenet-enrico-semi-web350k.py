if __name__ == "__main__":

    ARTIFACT_DIR = "checkpoints_screenclassification_enrico-webui-silver-web350k"
    CHECK_INTERVAL_STEPS = 32

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
    
    import torchvision.models as models
    
    from ui_models_extra import *
    
    logger = TensorBoardLogger(ARTIFACT_DIR)
    
    data = SilverDataModule(silver_id_list_path="../../downloads/train_split_web350k.json", K=35000)

    model = UIScreenClassifier(num_classes=20, lr=0.00005, arch="resnet50_conv", dropout=0.5)

    model_pretrained = models.resnet50(pretrained=True)
    replace_default_bn_with_custom(model_pretrained, dropout=model.hparams.dropout_block)
    replace_res_blocks_with_stochastic(model_pretrained, model.hparams.stochastic_depth_p)
    model.model = model_pretrained
    model.model.fc = nn.Sequential(nn.Dropout(model.hparams.dropout), nn.Linear(model.model.fc.in_features, model.hparams.num_classes))

    print("***********************************")
    print("checkpoints: " + str(os.listdir(ARTIFACT_DIR)))
    print("***********************************")
    
    checkpoint_callback = ModelCheckpoint(dirpath=ARTIFACT_DIR, every_n_train_steps=CHECK_INTERVAL_STEPS, save_last=True)
    checkpoint_callback2 = ModelCheckpoint(dirpath=ARTIFACT_DIR, filename= "screenclassification", save_top_k=1, every_n_train_steps=CHECK_INTERVAL_STEPS, mode="max", monitor="f1_weighted")
    earlystopping_callback = EarlyStopping(monitor="f1_weighted", mode="max", patience=100)
    trainer = Trainer(
        gpus=1,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, checkpoint_callback2, earlystopping_callback],
        logger=logger,
        val_check_interval=CHECK_INTERVAL_STEPS,
        accumulate_grad_batches=4,
    )
    
    if os.path.exists(os.path.join(ARTIFACT_DIR, "last.ckpt")):
        model = UIScreenClassifier.load_from_checkpoint(os.path.join(ARTIFACT_DIR, "last.ckpt"))

    trainer.fit(model, data)
