if __name__ == "__main__":

    ARTIFACT_DIR = "checkpoints_screenclassification_imagenet-enrico"
    CHECK_INTERVAL_STEPS = 8000

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
    
    logger = TensorBoardLogger(ARTIFACT_DIR)
    
    # data = WebUIClassificationDataModule()
    # data = WebUIReconstructionDataModule()
    data = EnricoDataModule()

    model = UIScreenClassifier(num_classes=20)
    # model = UIScreenClassifier(num_classes=32)
    # model = UIScreenSegmenter(num_classes=32)

    
    def convert_bn_to_in(model):
        for child_name, child in model.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(model, child_name, nn.InstanceNorm2d(child.num_features))
            else:
                convert_bn_to_in(child)
    
    model_pretrained = models.resnet50(pretrained=True)
    convert_bn_to_in(model_pretrained)
    model.model = model_pretrained
    
    # model_old = UIScreenClassifier.load_from_checkpoint("checkpoints_screenclassification_old/last.ckpt", strict=False)
    # model.model = model_old.model
    # model.conv_cls[-1] = nn.Conv2d(2048, 20, 3, stride=1, padding=1)
    model.model.fc = nn.Linear(model.model.fc.in_features, 20)
    # del model_old


    # for p in model.parameters():
    #     p.requires_grad = False

    # for p in model.conv_cls.parameters():
    #     p.requires_grad = True

    print("***********************************")
    print("checkpoints: " + str(os.listdir(ARTIFACT_DIR)))
    print("***********************************")
    
    checkpoint_callback = ModelCheckpoint(dirpath=ARTIFACT_DIR, every_n_train_steps=CHECK_INTERVAL_STEPS, save_last=True)
    checkpoint_callback2 = ModelCheckpoint(dirpath=ARTIFACT_DIR, filename= "screenclassification", monitor="f1_weighted", mode="max", save_top_k=1)
    earlystopping_callback = EarlyStopping(monitor="f1_weighted", mode="max", patience=20)
    # earlystopping_callback = EarlyStopping(monitor="bce", mode="min", patience=10)

    trainer = Trainer(
        gpus=1,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, checkpoint_callback2, earlystopping_callback],
        #default_root_dir="s3://sagemaker-screenclassification",
        logger=logger,
        # val_check_interval=CHECK_INTERVAL_STEPS,
        # min_epochs=10000,
        accumulate_grad_batches=2,
        min_epochs=10
    )
    
    if os.path.exists(os.path.join(ARTIFACT_DIR, "last.ckpt")):
        model = UIScreenClassifier.load_from_checkpoint(os.path.join(ARTIFACT_DIR, "last.ckpt"))

    trainer.fit(model, data)
