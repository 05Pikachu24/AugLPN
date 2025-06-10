Wuqing Yu, Linfeng Yang, and Xiangyu Liu. "AugLPN-NILM: Augmented lightweight parallel network for NILM embedding attention module over sequence to point." Sustainable Energy, Grids and Networks 38 (2024): 101378.

15 Feb 2025 Add: An interesting thing is that by adjusting the loss function of AugLPN, you will be able to surpass the state-of-the-art (SOTA) results reported in the paper, which implies that the model has great potential to conduct the load decomposition.
In this repository are available codes for implementation of our study.

# Requirements:
The version of python should preferably be greater than 3.7
our environment(for reference only):
    tensorflow==2.3.0
    keras==2.4.0
    scikit-learn==1.1.2

# Reference(Acknowledgement):
1. https://github.com/MingjunZhong/NeuralNetNilm
2. https://github.com/MingjunZhong/transferNILM/
3. C. Zhang, M. Zhong, Z. Wang, N. Goddard, and C. Sutton. Sequence-to-point learning with neural networks
for non-intrusive load monitoring. In Proceedings for Thirty-Second AAAI Conference on Artificial Intelligence.
AAAI Press, 2018.


# Get the paper results quickly
Some already well-trained models ('*.h5' files) are in the folder directory '/models' 
Change the file path (refer to the parameter 'param_file')  in the AugLPNNILM_test.py, and you will get the results soon.
    For example: param_file = args.trained_model_dir + '/UK_DALE'+ '/AugLPN_' + args.appliance_name + '_pointnet_model'

# ----------***Reproduce  our results***-----start-----------
# 1. Prepare training and test dataset for REDD and UK_DALE
1. REDD and UK_DALE datasets are available in (http://redd.csail.mit.edu/) and (https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2015/UK-DALE-disaggregated).
2. Put the raw data into the folder directory dataset_preprocess, and named low_freq and UK_DALE respectively.
3. Run redd_processing.py and uk_dale_processing.py to get the prepared dataset for training and test.
   (note that the preprocessing of UK_DALE dataset needs another step :put the preprocessed data in "dataset_preprocess/created_data/UK_DALE/" )
The structure of folder directory is as follows:<br>
   dataset_processing/ <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;created_data/ <br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;REDD/ <br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;UK_DALE/ <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;low_freq/ <br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;house_1/ <br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;house_2/ <br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;UK_DALE/ <br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;house_2/ <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;redd_processing.py <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ukdale_processing.py <br>

# 2. Start the training
You can run AugLPNNILM_train.py to verify the results in our paper after you have preprocessed all the dataset.
The best results('*.h5' files) will be stored in the file directory '/models'
# 3. Start the test
Change the file path (refer to the parameter 'param_file')  in the AugLPNNILM_test.py, and you will get the results soon.
    for example: param_file = args.trained_model_dir + '/AugLPN_' + args.appliance_name + '_pointnet_model'
# ------------***Reproduce  our results***-----end----------

Contact e-mail:
2113301058@st.gxu.edu.cn
