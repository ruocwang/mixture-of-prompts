conda create -n mop python=3.9
conda activate mop
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install tqdm
pip install transformers==4.38.1
pip install openai==1.14.3
pip install matplotlib
pip install rouge_score==0.1.2
pip install scikit-learn==1.4.1.post1
pip install pandas==2.2.0
## install this if you want to run OOD dataset
pip install k-means-constrained