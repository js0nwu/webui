if __name__ == "__main__":

    ARTIFACT_DIR = "/checkpoints_screenrecognition_web350k"

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
    
    # data = ClayUIDataModule()
    # data = RicoUIDataModule()
    # data = ClayUIDataModule(one_hot_labels=True)
    
    data = WebUIDataModule(train_split_file = 'train_split_web350k.json')
    
    # for webmini
    model = UIElementDetector(num_classes=32, val_weights = [0.0000e+00, 8.5562e-03, 4.7493e-01, 2.2077e-01, 7.4514e-02, 3.7923e-02,
        5.8686e-02, 5.6902e-02, 4.0222e-04, 2.1070e-02, 8.4225e-03, 8.6766e-03,
        1.6245e-03, 3.3478e-03, 6.0092e-03, 2.1418e-03, 9.7243e-04, 2.3983e-03,
        2.1189e-03, 2.5169e-03, 3.4803e-04, 7.2677e-04, 1.0001e-03, 1.5414e-04,
        1.0525e-03, 9.5557e-04, 1.6619e-03, 1.5896e-04, 5.6419e-04, 9.6701e-04,
        2.4627e-04, 1.9208e-04]
, test_weights= [0.0000e+00, 8.5988e-03, 4.7486e-01, 2.2070e-01, 7.6354e-02, 3.7529e-02,
        5.4189e-02, 5.7368e-02, 5.6382e-04, 2.1981e-02, 7.8788e-03, 9.6592e-03,
        1.1740e-03, 3.8739e-03, 8.1307e-03, 1.7772e-03, 1.0072e-03, 2.6138e-03,
        2.0770e-03, 2.6934e-03, 5.5354e-04, 6.5838e-04, 1.1482e-03, 2.0027e-04,
        5.3886e-04, 6.6719e-04, 1.3585e-03, 2.0380e-04, 7.0742e-04, 6.9831e-04,
        1.7707e-04, 5.5207e-05])
    
    # for clay
    # model = UIElementDetector(num_classes=24)
    # model = UIElementDetector.load_from_checkpoint("checkpoint41.ckpt")
    print("***********************************")
    print("checkpoints: " + str(os.listdir(ARTIFACT_DIR)))
    print("***********************************")
    
    
    #checkpoint_s3_bucket="s3://sagemaker-screenrec/checkpoints"
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
