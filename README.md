# Fed-XCal: Federated cross-modal calibration for privacy-preserving hand biometric recognition

This is the official PyTorch implementation of **Fed-XCal**.

#### Abstract
Decentralized intelligence is essential for privacy-sensitive identity authentication, yet it faces an inherent conflict between model performance and data privacy. Data heterogeneity across distributed clients leads to significant feature space drift, which undermines the reliability of the aggregated global model. While existing methods attempt to mitigate such drift by relying on global consensus to guide local training and federated aggregation, they remain inherently passive rather than actively restoring performance. To address this challenge, we propose Fed-XCal, a Federated Cross-modal Calibration framework that actively calibrates both feature and decision spaces by using the natural consistency between different hand biometric modalities as a reliable anchor. It enables autonomous self-correction of drifted representations at the local level, while dynamically weighing predictive reliability for a metacognitive fusion of cross-modal information. Extensive evaluations on hand-based multi-modal datasets demonstrate the robustness of Fed-XCal under extreme non-IID settings. Specifically, for heterogeneous palmprint and palm vein datasets, the proposed Fed-XCal establishes substantial margins over state-of-the-art methods across both identification and verification scenarios. Furthermore, our method exhibits strong generalization when extended to other hand-based biometric modalities and training paradigms. Crucially, this cross-modal success establishes that intrinsic multi-modal consistency can effectively overcome single-view heterogeneity for next-generation distributed hand biometrics.


#### Requirements
Our codes were implemented by PyTorch and CUDA. If you wanna try our method, please first install the necessary packages as follows:
```bash
pip install -r requirements.txt
```

#### Data Preprocessing
Both the training and testing sets are loaded as .txt files. Each line in the text file represents the content of an image in the dataset. The specific format is:
Image_Path Image_Label (i.e., two columns per line, separated by a space).

You can directly run the following script to generate these text files (code refers to ./genText.py by https://github.com/Zi-YuanYang/CCNet):
```bash
python ./genText.py
```

#### Pretrained Model
To help readers use our model, we release our final model weights for the palmprint-palmvein datasets. They are located in the https://github.com/SYK-BJUT/Fed-XCal/tree/master models directory.
Models numbered 0-5 correspond to the datasets: BLUE, NIR, WHT, 700, TP (Tongji Print), and TV (Tongji Vein), respectively.

#### Training
After you prepare the training and testing texts, you can configure the framework via command-line arguments. 

* `--batch_size`: The size of batch to be used for local training. (default: `512`)
* `--epoch_num`: The number of local training epochs. (default: `3`)
* `--com`: The number of federated communication rounds. (default: `100`)
* `--temp`: The temperature value in the contrastive loss. (default: `0.07`)
* `--weight1`: The weight of cross-entropy loss when uncertainty weighting is not used. (default: `0.7`)
* `--weight2`: The weight of contrastive loss when uncertainty weighting is not used. (default: `0.15`)
* `--weight3`: The weight of Prox loss when uncertainty weighting is not used. (default: `1.0`)
* `--weight4`: The weight of Arcface loss when uncertainty weighting is not used. (default: `0.01`)
* `--weight5`: The weight of Center loss when uncertainty weighting is not used. (default: `0.1`)
* `--mu`: The intensity in the Prox loss. (default: `1e-2`)
* `--arcface_s`: The scale factor for Arcface loss. (default: `64.0`)
* `--arcface_m`: The angular margin for Arcface loss. (default: `0.5`)
* `--center_alpha`: The update rate for the center points for Center loss. (default: `0.5`)
* `--feat_dim`: The dimension of features. (default: `6144`)
* `--lr`: The initial training learning rate. (default: `0.001`)
* `--hidden_dim`: The hidden layer dimension of the weight prediction network. (default: `64`)
* `--wn_lr`: Learning rate for the weight network. (default: `0.01`)
* `--wn_weight_decay`: Weight decay for the weight network. (default: `1e-4`)
* `--wn_step_size`: Step size for weight network LR scheduler. (default: `50`)
* `--wn_gamma`: Gamma decay for weight network LR scheduler. (default: `0.8`)
* `--wn_epochs`: Number of epochs to train the weight network. (default: `500`)
* `--id_num`: The total number of identity classes across all clients. (default: `1000`)
* `--gpu_id`: The ID of the training GPU. (default: `'1'`)
* `--seed`: The random seed. (default: `42`)
* `--model_save_dir`: The path for saving single/multi-modal testing results. (default: `./Results/Saved models/`)

#### Acknowledgments
Thanks to all my cooperators, they contributed so much to this work.

#### Reference
We refer to the following repositories:
* https://github.com/Zi-YuanYang/PSFed-Palm
* https://github.com/Zi-YuanYang/CCNet